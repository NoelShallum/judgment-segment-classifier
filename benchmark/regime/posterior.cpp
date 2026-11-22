#include <algorithm>
#include <cmath>
#include <vector>
extern "C" void posterior(const float *obs, const float *edge, const float *start,
                          int rows, float *out) {
    constexpr int k = 7;
    std::vector<double> f(rows * k);
    double back[k], prev[k], denom;
    std::fill(back, back+k, 1.0);
    for (int i=0; i<rows; ++i) {
        denom=0;
        for (int j=0; j<k; ++j) {
            double v= i ? 0.0 : start[j];
            if (i) for (int t=0; t<k; ++t) v+=f[(i-1)*k+t]*edge[t*k+j];
            f[i*k+j]=v*obs[i*k+j]; denom+=f[i*k+j];
        }
        for (int j=0; j<k; ++j) f[i*k+j]/=std::max(denom,1e-30);
    }
    for (int i=rows-1; i>=0; --i) {
        denom=0;
        for (int j=0; j<k; ++j) {
            out[i*k+j]=f[i*k+j]*back[j]; denom+=out[i*k+j];
        }
        for (int j=0; j<k; ++j) out[i*k+j]/=std::max(denom,1e-30);
        denom=0;
        for (int j=0; j<k; ++j) {
            double v=0.;
            for (int t=0; t<k; ++t) v+=edge[j*k+t]*obs[i*k+t]*back[t];
            prev[j]=v; denom+=v;
        }
        for(int j=0; j<k; ++j) back[j]=prev[j]/std::max(denom,1e-30);
    }
}
