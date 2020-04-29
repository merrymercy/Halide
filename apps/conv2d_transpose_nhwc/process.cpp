#include <cstdio>
#include <chrono>

#include "conv2d_transpose_nhwc.h"
#include "conv2d_transpose_nhwc_classic_auto_schedule.h"
#include "conv2d_transpose_nhwc_auto_schedule.h"

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
    const int kernel_size = args[i++];
    const int strides     = GetArg(args, i++, 1);
    const int padding     = GetArg(args, i++, 0);

    const int KH = kernel_size;
    const int KW = kernel_size;
    const int SH = strides;
    const int SW = strides;

    const int OH = (H - 1) * SH - 2 * padding + KH;
    const int OW = (W - 1) * SW - 2 * padding + KW;

    Buffer<float> input(CI, W, H, N);
    Buffer<float> filter(CO, CI, KW, KH);
    Buffer<float> output(CO, OW, OH, N);

    input.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});
    filter.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});

    three_way_bench(
        [&]() { conv2d_transpose_nhwc(input, filter, output); },
        [&]() { conv2d_transpose_nhwc_classic_auto_schedule(input, filter, output); },
        [&]() { conv2d_transpose_nhwc_auto_schedule(input, filter, output); }
    );

    return 0;
}
