#include <cstdio>
#include <chrono>

#include "depthwise_conv2d_nhwc.h"
#include "depthwise_conv2d_nhwc_classic_auto_schedule.h"
#include "depthwise_conv2d_nhwc_auto_schedule.h"

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
    const int dilation    = GetArg(args, i++, 1);
    const int factor      = GetArg(args, i++, 1);

    const int KH = kernel_size;
    const int KW = kernel_size;
    const int SH = strides;
    const int SW = strides;
    const int PH = padding;
    const int PW = padding;
    const int DH = dilation;
    const int DW = dilation;

    const int OH = (H + 2 * PH - (KH - 1) * DH - 1) / SH + 1;
    const int OW = (W + 2 * PW - (KW - 1) * DW - 1) / SW + 1;

    Buffer<float> input(CI, W, H, N);
    Buffer<float> filter(CI, KW, KH, factor);
    Buffer<float> output(CO, OW, OH, N);

    input.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});
    filter.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});

    three_way_bench(
        [&]() { depthwise_conv2d_nhwc(input, filter, output); },
        [&]() { depthwise_conv2d_nhwc_classic_auto_schedule(input, filter, output); },
        [&]() { depthwise_conv2d_nhwc_auto_schedule(input, filter, output); }
    );

    return 0;
}
