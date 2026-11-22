#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" void role_viterbi(const float* x, const float* trans, const std::int64_t* offsets, std::size_t docs, std::int8_t* out) {
    for(std::size_t d=0;d<docs;++d) {
        auto lo=offsets[d], hi=offsets[d+1];
        float state[7],next[7];std::vector<std::int8_t> back((hi-lo)*7);
        for(int k=0;k<7;++k){
            state[k]=x[lo*7+k]+trans[13+k];
            if(hi==lo+1)state[k]+=trans[(k+3)*10+2];
        }
        for(auto t=lo+1;t<hi;++t){
            for(int k=0;k<7;++k){
                int who=0;float best=state[0]+trans[3*10+k+3];
                for(int j=1;j<7;++j){
                    float z=state[j]+trans[(j+3)*10+k+3];
                    if(z>best){who=j;best=z;}
                }
                float z=x[t*7+k];if(t==hi-1)z+=trans[(k+3)*10+2];
                next[k]=best+z;back[(t-lo)*7+k]=static_cast<std::int8_t>(who);
            }
            std::copy(next,next+7,state);
        }
        int k=static_cast<int>(std::max_element(state,state+7)-state);
        for(auto t=hi-1;t>=lo;--t){out[t]=static_cast<std::int8_t>(k);if(t>lo)k=back[(t-lo)*7+k];}
    }
}
