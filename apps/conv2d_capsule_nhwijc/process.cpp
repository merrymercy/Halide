#include <cstdio>
#include <chrono>

#include "conv2d_capsule_nhwijc.h"
#include "conv2d_capsule_nhwijc_classic_auto_schedule.h"
#include "conv2d_capsule_nhwijc_auto_schedule.h"

#include "benchmark_util.h"
#include "HalideBuffer.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    std::vector<int> args = GetArgsFromEnv();
    int i = 0;
    const int N  = args[i++];
    const int H  = args[i++];
    const int W  = args[i++];
    const int CI = args[i++];
    const int CO = args[i++];
    const int kernel_size  = args[i++];
    const int strides      = GetArg(args, i++, 1);
    const int padding      = GetArg(args, i++, 0);
    const int capsule_size = GetArg(args, i++, 4);

    const int KH = kernel_size;
    const int KW = kernel_size;
    const int SH = strides;
    const int SW = strides;
    const int PH = padding;
    const int PW = padding;

    const int OH = (H + 2 * PH - KH) / SH + 1;
    const int OW = (W + 2 * PW - KW) / SW + 1;

    Buffer<float> input(CI, capsule_size, capsule_size, W, H, N);
    Buffer<float> filter(CO, CI, capsule_size, capsule_size, KW, KH);
    Buffer<float> output(CO, capsule_size, capsule_size, OW, OH, N);

    input.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});
    filter.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});

    three_way_bench(
        [&]() { conv2d_capsule_nhwijc(input, filter, output); },
        [&]() { conv2d_capsule_nhwijc_classic_auto_schedule(input, filter, output); },
        [&]() { conv2d_capsule_nhwijc_auto_schedule(input, filter, output); }
    );

    return 0;
}
