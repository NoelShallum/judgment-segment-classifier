// Exact Viterbi with one state per ordered pair of rhetorical roles.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" void decode_trigrams(const double* unary, const float* edge,
                                const std::int64_t* offsets, std::size_t documents,
                                const std::int8_t* section, const float* tensor,
                                float edge_gain, float stay, float triple_gain,
                                std::int8_t* labels) {
    constexpr int K = 7, S = K*K;
    std::vector<std::int8_t> back;
    for (std::size_t doc = 0; doc < documents; ++doc) {
        const auto first=offsets[doc], last=offsets[doc+1];
        if (first==last) continue;
        if (last==first+1) {
            int best=0;
            for (int c=1;c<K;++c)
                if (unary[first*K+c]>unary[first*K+best]) best=c;
            labels[first]=best;
            continue;
        }
        back.resize((last-first)*S);
        double score[S], next[S];
        for (int a=0;a<K;++a) for (int b=0;b<K;++b) {
            const int ab=a*K+b;
            score[ab]=unary[first*K+a]+(a==b ? stay : 0.f)
                +edge[(first+1)*S+ab]*edge_gain+unary[(first+1)*K+b];
        }
        for (auto i=first+2;i<last;++i) {
            const float* block=tensor+section[i]*K*S;
            for (int b=0;b<K;++b) for (int c=0;c<K;++c) {
                const int bc=b*K+c;
                double best=-std::numeric_limits<double>::infinity();
                int from=0;
                for (int a=0;a<K;++a) {
                    double value=score[a*K+b]+(b==c ? stay : 0.f)
                        +edge[i*S+bc]*edge_gain+block[a*S+bc]*triple_gain;
                    if (value>best) { best=value; from=a; }
                }
                next[bc]=best+unary[i*K+c];
                back[(i-first)*S+bc]=from;
            }
            for (int j=0;j<S;++j) score[j]=next[j];
        }
        int prev=0, curr=0;
        for (int c=0;c<K;++c) for (int b=0;b<K;++b)
            if (score[b*K+c]>score[prev*K+curr]) { prev=b; curr=c; }
        for (auto i=last-1;i>=first+2;--i) {
            labels[i]=curr;
            const int old=back[(i-first)*S+prev*K+curr];
            curr=prev;
            prev=old;
        }
        labels[first+1]=curr;
        labels[first]=prev;
    }
}
