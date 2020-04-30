#include <cstdio>
#include <chrono>

#include "conv1d_nlc.h"
#include "conv1d_nlc_classic_auto_schedule.h"
#include "conv1d_nlc_auto_schedule.h"

#include "benchmark_util.h"
#include "HalideBuffer.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    std::vector<int> args = GetArgsFromEnv();
    int i = 0;
    const int N  = args[i++];
    const int L = args[i++];
    const int CI = args[i++];
    const int CO = args[i++];
    const int kernel_size = args[i++];
    const int strides     = GetArg(args, i++, 1);
    const int padding     = GetArg(args, i++, 0);
    const int dilation    = GetArg(args, i++, 1);
    const int groups      = GetArg(args, i++, 1);

    const int OL = (L + 2 * padding - (kernel_size - 1) * dilation - 1) / strides + 1;

    Buffer<float> input(CI, L, N);
    Buffer<float> filter(CO, CI / groups, kernel_size);
    Buffer<float> output(CO, OL, N);

    input.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});
    filter.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});

    three_way_bench(
        [&]() { conv1d_nlc(input, filter, output); },
        [&]() { conv1d_nlc_classic_auto_schedule(input, filter, output); },
        [&]() { conv1d_nlc_auto_schedule(input, filter, output); }
    );

    return 0;
}
