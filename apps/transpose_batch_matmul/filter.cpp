#include <cstdio>

#include "transpose_batch_matmul.h"
#include "transpose_batch_matmul_classic_auto_schedule.h"
#include "transpose_batch_matmul_auto_schedule.h"

#include "benchmark_util.h"
#include "HalideBuffer.h"

int main(int argc, char **argv) {
    if (argc != 1) {
        printf("Usage: %s\n", argv[0]);
        return 1;
    }

    std::vector<int> args = GetArgsFromEnv();
    int i = 0;
    const int B = args[i++];
    const int N = args[i++];
    const int M = args[i++];
    const int K = args[i++];

    Halide::Runtime::Buffer<float> mat_A(K, N, B);
    Halide::Runtime::Buffer<float> mat_B(K, M, B);
    Halide::Runtime::Buffer<float> output(M, N, B);

    // init randomly
    for (int ib = 0; ib < B; ib++) {
        for (int iy = 0; iy < N; iy++) {
            for (int ix = 0; ix < K; ix++) {
                mat_A(ix, iy, ib) = (rand() % 256) / 256.0f;
            }
        }
    }

    for (int ib = 0; ib < B; ib++) {
        for (int iy = 0; iy < M; iy++) {
            for (int ix = 0; ix < K; ix++) {
                mat_B(ix, iy, ib) = (rand() % 256) / 256.0f;
            }
        }
    }

    three_way_bench(
        [&]() { transpose_batch_matmul(mat_A, mat_B, output); },
        [&]() { transpose_batch_matmul_classic_auto_schedule(mat_A, mat_B, output); },
        [&]() { transpose_batch_matmul_auto_schedule(mat_A, mat_B, output); }
    );

    return 0;
}
