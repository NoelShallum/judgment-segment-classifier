#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" bool segment_path(const int32_t* spans, const double* weights,
                             std::size_t count, std::size_t n, int8_t* labels) {
    constexpr int roles = 7;
    const double unreachable = -std::numeric_limits<double>::infinity();
    std::vector<double> score((n + 1) * roles, unreachable);
    std::vector<int32_t> start((n + 1) * roles, -1), previous(start);
    for (std::size_t i = 0; i < count; ++i) {
        const int a = spans[3 * i], b = spans[3 * i + 1], k = spans[3 * i + 2];
        double before = a ? unreachable : 0.0;
        int past = -1;
        if (a) {
            for (int p = 0; p < roles; ++p) {
                if (p != k && score[a * roles + p] > before) {
                    before = score[a * roles + p];
                    past = p;
                }
            }
        }
        const auto at = b * roles + k;
        if (before + weights[i] > score[at]) {
            score[at] = before + weights[i];
            start[at] = a;
            previous[at] = past;
        }
    }
    int role = std::max_element(score.begin() + n * roles, score.end()) -
               (score.begin() + n * roles);
    int b = n;
    while (b > 0) {
        const auto at = b * roles + role;
        const int a = start[at];
        if (a < 0) return false;
        std::fill(labels + a, labels + b, role);
        b = a;
        role = previous[at];
    }
    return true;
}
