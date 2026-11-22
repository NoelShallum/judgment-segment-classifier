// Read the last-token hidden state and seven native logits for each local excerpt.
// Compatible with the public llama.cpp b11960 C API.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs=std::filesystem;
std::string read_text(const fs::path &path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    std::ostringstream data;
    data << input.rdbuf();
    return data.str();
}
std::vector<llama_token> tokenize(const llama_vocab *v,const std::string &text,bool bos,bool control) {
    std::vector<llama_token> ids(text.size()+16);
    int n=llama_tokenize(v,text.data(),text.size(),ids.data(),ids.size(),bos,control);
    if (n<0) throw std::runtime_error("Tokenization failed");
    ids.resize(n);
    return ids;
}
void decode(llama_context *ctx,const std::vector<llama_token>& tokens,int offset,bool emit) {
    for (size_t first=0;first<tokens.size();first+=256) {
        int n=std::min<size_t>(256,tokens.size()-first);
        auto batch=llama_batch_init(n,0,1);
        batch.n_tokens=n;
        for (int j=0;j<n;++j) {
            batch.token[j]=tokens[first+j];
            batch.pos[j]=offset+first+j;
            batch.n_seq_id[j]=1;
            batch.seq_id[j][0]=0;
            batch.logits[j]=emit && first+j+1==tokens.size();
        }
        int status=llama_decode(ctx,batch);
        llama_batch_free(batch);
        if (status!=0) throw std::runtime_error("Decoder status " + std::to_string(status));
    }
}
int main(int argc,char **argv) {
    if (argc!=6) {std::cerr<<"project_hidden MODEL BACKEND_LIBRARY_DIR INPUT_DIR THREADS CONTEXT\n";return 2;}
    try {
        fs::path task=argv[3];
        ggml_backend_load_all_from_path(argv[2]);
        llama_backend_init();
        auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        auto model=llama_model_load_from_file(argv[1],mp);
        if (!model) throw std::runtime_error("Unable to load model");
        auto cp=llama_context_default_params();
        cp.n_ctx=std::stoi(argv[5]);cp.n_batch=256;cp.n_ubatch=256;
        cp.n_threads=cp.n_threads_batch=std::stoi(argv[4]);
        cp.embeddings=true;cp.pooling_type=LLAMA_POOLING_TYPE_NONE;cp.swa_full=true;
        auto ctx=llama_init_from_model(model,cp);
        if (!ctx) throw std::runtime_error("Unable to create context");
        auto vocab=llama_model_get_vocab(model);
        auto head=tokenize(vocab,read_text(task/"prefix.txt"),true,true);
        auto foot=tokenize(vocab,read_text(task/"suffix.txt"),false,true);
        std::array<int,7> choices;
        bool semantic=fs::exists(task/"choices.txt");
        std::istringstream options(semantic ? read_text(task/"choices.txt") : "");
        for (int i=0;i<7;i++) {
            std::string text(1,'A'+i);
            if (semantic && !std::getline(options,text)) throw std::runtime_error("Missing role verbalizer");
            auto ids=tokenize(vocab,text,false,false);
            if (ids.empty() || (!semantic && ids.size()!=1)) throw std::runtime_error("Invalid option tokens");
            choices[i]=ids[0];
            for (int j=0;j<i;++j) if (choices[i]==choices[j]) throw std::runtime_error("First role tokens collide");
        }
        fs::create_directories(task/"vectors");
        {std::ofstream file(task/"token-ids.txt");for (int id:choices) file<<id<<" ";}
        decode(ctx,head,0,false);
        auto mem=llama_get_memory(ctx);
        std::istringstream list(read_text(task/"manifest.txt"));
        std::string name;int count=0;int width=llama_model_n_embd_out(model);
        while (std::getline(list,name)) {
            if (name.empty()) continue;
            auto target=task/"vectors"/(name+".f32");
            if (fs::exists(target) && fs::file_size(target)==sizeof(float)*(width+7)) continue;
            auto ids=tokenize(vocab,read_text(task/"texts"/(name+".txt")),false,false);
            ids.insert(ids.end(),foot.begin(),foot.end());
            if (head.size()+ids.size()>llama_n_ctx(ctx)) throw std::runtime_error("Input exceeds configured context");
            if (!llama_memory_seq_rm(mem,0,head.size(),-1)) {
                llama_memory_clear(mem,false);
                decode(ctx,head,0,false);
            }
            decode(ctx,ids,head.size(),true);
            auto hidden=llama_get_embeddings_ith(ctx,-1);
            auto logits=llama_get_logits_ith(ctx,-1);
            if (!hidden || !logits) throw std::runtime_error("Decoder did not return both requested outputs");
            auto temp=target.string()+".partial";
            {std::ofstream file(temp,std::ios::binary);
            file.write(reinterpret_cast<const char*>(hidden),width*sizeof(float));
            for (int opt:choices) file.write(reinterpret_cast<const char*>(logits+opt),sizeof(float));
            if (!file) throw std::runtime_error("Cannot save pooled vector");}
            fs::rename(temp,target);
            ++count;
            if (count%10==0 || count==1) std::cout<<"Computed "<<count<<" excerpts in this run\n"<<std::flush;
        }
        llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
    } catch (const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
