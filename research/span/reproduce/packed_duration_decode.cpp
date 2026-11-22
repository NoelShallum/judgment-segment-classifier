// Exact Viterbi with capped role duration and time-dependent role-pair weights.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" void decode_durations(
    const double* unary, const float* edge, const std::int64_t* offsets,
    std::size_t documents, std::size_t ages, const float* stop, const float* keep,
    std::int8_t* path) {
    constexpr std::size_t roles = 7;
    const auto states = roles * ages;
    const auto impossible = -std::numeric_limits<double>::infinity();
    std::vector<double> current(states), next(states);
    std::vector<std::uint16_t> parent;
    for (std::size_t d = 0; d < documents; ++d) {
        const auto begin = offsets[d];
        const auto end = offsets[d + 1];
        parent.assign((end - begin) * states, 0);
        std::fill(current.begin(), current.end(), impossible);
        for (std::size_t r = 0; r < roles; ++r) {
            current[r * ages] = unary[begin * roles + r];
        }
        for (auto row = begin + 1; row < end; ++row) {
            std::fill(next.begin(), next.end(), impossible);
            double ending[roles];
            std::size_t origins[roles];
            for (std::size_t r = 0; r < roles; ++r) {
                ending[r] = impossible;
                origins[r] = r * ages;
                for (std::size_t a = 0; a < ages; ++a) {
                    const auto i = r * ages + a;
                    const double candidate = current[i] + stop[i];
                    if (candidate > ending[r]) {
                        ending[r] = candidate;
                        origins[r] = i;
                    }
                }
            }
            const auto back = (row - begin) * states;
            const auto ep = edge + row * roles * roles;
            for (std::size_t r = 0; r < roles; ++r) {
                const auto b = r * ages;
                for (std::size_t previous = 0; previous < roles; ++previous) {
                    if (previous == r) continue;
                    const double offer = ending[previous] + ep[previous * roles + r];
                    if (offer > next[b]) {
                        next[b] = offer;
                        parent[back + b] = origins[previous];
                    }
                }
                for (std::size_t a = 0; a < ages; ++a) {
                    const auto i = b + a;
                    const auto j = b + std::min(a + 1, ages - 1);
                    const double offer = current[i] + ep[r * roles + r] + keep[i];
                    if (offer > next[j]) {
                        next[j] = offer;
                        parent[back + j] = i;
                    }
                }
                for (std::size_t a = 0; a < ages; ++a) {
                    next[b + a] += unary[row * roles + r];
                }
            }
            current.swap(next);
        }
        std::size_t best = 0;
        for (std::size_t i = 1; i < states; ++i) {
            if (current[i] + stop[i] > current[best] + stop[best]) best = i;
        }
        for (auto row = end - 1; row >= begin; --row) {
            path[row] = best / ages;
            if (row > begin) best = parent[(row - begin) * states + best];
        }
    }
}
