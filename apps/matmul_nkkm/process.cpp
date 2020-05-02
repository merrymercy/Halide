#include <chrono>
#include <cstdio>

#include "matmul.h"
#include "matmul_auto_schedule.h"

#include "HalideBuffer.h"
#include "halide_benchmark.h"

#include "../autoscheduler/utils.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    if (argc != 1) {
        printf("Usage: %s\n", argv[0]);
        return 1;
    }

    std::vector<int> args = GetArgsFromEnv();
    int i = 0;
    const int N = args[i++];
    const int M = args[i++];
    const int K = args[i++];

    Buffer<float> mat_A(K, N);
    Buffer<float> mat_B(M, K);
    Buffer<float> output(M, N);

    // init randomly
    for (int iy = 0; iy < N; iy++) {
        for (int ix = 0; ix < K; ix++) {
            mat_A(ix, iy) = rand();
        }
    }

    for (int iy = 0; iy < K; iy++) {
        for (int ix = 0; ix < M; ix++) {
            mat_B(ix, iy) = rand();
        }
    }

    matmul(mat_A, mat_B, output);

    // Timing code

    // Manually-tuned version
    double min_t_manual = benchmark(50, 50, [&]() {
        matmul(mat_A, mat_B, output);
    });
    printf("Manually-tuned time: %gms\n", min_t_manual * 1e3);

//    // Auto-scheduled version
//    double min_t_auto = benchmark(10, 10, [&]() {
//        matmul_auto_schedule(mat_A, mat_B, output);
//    });
//    printf("Auto-scheduled time: %gms\n", min_t_auto * 1e3);

    return 0;
}
