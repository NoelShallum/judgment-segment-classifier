// Viterbi across contiguous, independently scored judgments.
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" void decode_judgments(
    const double* emissions, const float* edge, const std::int64_t* offsets,
    std::size_t documents, const float* transition, const float* edge_scale,
    std::int8_t* path) {
    constexpr int roles = 7;
    std::vector<std::int8_t> backtrack;
    for (std::size_t document = 0; document < documents; ++document) {
        const auto begin = offsets[document];
        const auto end = offsets[document + 1];
        backtrack.resize((end - begin) * roles);
        double state[roles];
        for (int role = 0; role < roles; ++role) {
            state[role] = emissions[begin * roles + role] + transition[13 + role];
        }
        for (auto i = begin + 1; i < end; ++i) {
            double next[roles];
            for (int current = 0; current < roles; ++current) {
                double best = 0;
                int origin = 0;
                for (int previous = 0; previous < roles; ++previous) {
                    const int pair = previous * roles + current;
                    const double score = state[previous]
                        + transition[(previous + 3) * 10 + current + 3]
                        + edge[i * roles * roles + pair] * edge_scale[pair];
                    if (previous == 0 || score > best) {
                        best = score;
                        origin = previous;
                    }
                }
                backtrack[(i - begin) * roles + current] = origin;
                next[current] = best + emissions[i * roles + current];
            }
            for (int role = 0; role < roles; ++role) state[role] = next[role];
        }
        int role = 0;
        for (int candidate = 1; candidate < roles; ++candidate) {
            if (state[candidate] + transition[(candidate + 3) * 10 + 2]
                > state[role] + transition[(role + 3) * 10 + 2]) role = candidate;
        }
        for (auto i = end - 1; i >= begin; --i) {
            path[i] = role;
            if (i > begin) role = backtrack[(i - begin) * roles + role];
        }
    }
}
