#include <cstdio>

#include "mat_mul.h"
#include "mat_mul_classic_auto_schedule.h"
#include "mat_mul_auto_schedule.h"

#include "benchmark_util.h"
#include "HalideBuffer.h"

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

    Halide::Runtime::Buffer<float> mat_A(K, N);
    Halide::Runtime::Buffer<float> mat_B(M, K);
    Halide::Runtime::Buffer<float> output(M, N);

    // init randomly
    for (int iy = 0; iy < N; iy++) {
        for (int ix = 0; ix < K; ix++) {
            mat_A(ix, iy) = (rand() % 256) / 256.0f;
        }
    }

    for (int iy = 0; iy < K; iy++) {
        for (int ix = 0; ix < M; ix++) {
            mat_B(ix, iy) = (rand() % 256) / 256.0f;
        }
    }

    three_way_bench(
        [&]() { mat_mul(mat_A, mat_B, output); },
        [&]() { mat_mul_classic_auto_schedule(mat_A, mat_B, output); },
        [&]() { mat_mul_auto_schedule(mat_A, mat_B, output); }
    );

    return 0;
}
