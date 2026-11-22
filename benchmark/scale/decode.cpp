#include <vector>
#include <cstdint>
#include <algorithm>
extern "C" void decode(const float* emission,const float* transition,const int64_t* off, int docs, int8_t* out) {
    for (int d=0;d<docs;d++){
        int a=off[d],n=off[d+1]-a;
        if (n==0) continue;
        std::vector<int8_t> trace(n*7);
        float previous[7],next[7];
        for(int c=0;c<7;c++)previous[c]=emission[a*7+c];
        for(int i=1;i<n;i++) {
            for(int c=0;c<7;c++) {
                int best=0; float value=previous[0]+transition[c];
                for(int j=1;j<7;j++){float v=previous[j]+transition[j*7+c]; if(v>value){value=v;best=j;}}
                next[c]=value+emission[(a+i)*7+c];trace[i*7+c]=best;
            }
            std::copy(next,next+7,previous);
        }
        int c=std::max_element(previous,previous+7)-previous;
        for(int i=n-1;i>=0;i--){out[a+i]=c;if(i>0)c=trace[i*7+c];}
    }
}
