#include <cstdio>

#include "batch_norm.h"
#include "batch_norm_classic_auto_schedule.h"
#include "batch_norm_auto_schedule.h"

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
    const int M = args[i++];
    const int N = args[i++];

    Halide::Runtime::Buffer<float> input(N, M, B);
    Halide::Runtime::Buffer<float> output(B);

    // init randomly
    input.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});

    three_way_bench(
        [&]() { batch_norm(input, output); },
        [&]() { batch_norm_classic_auto_schedule(input, output); },
        [&]() { batch_norm_auto_schedule(input, output); }
    );

    return 0;
}
