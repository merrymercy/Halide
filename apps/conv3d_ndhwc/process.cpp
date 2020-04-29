#include <cstdio>
#include <chrono>

#include "conv3d_ndhwc.h"
#include "conv3d_ndhwc_classic_auto_schedule.h"
#include "conv3d_ndhwc_auto_schedule.h"

#include "benchmark_util.h"
#include "HalideBuffer.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    std::vector<int> args = GetArgsFromEnv();
    size_t i = 0;
    const int N  = args[i++];
    const int D  = args[i++];
    const int H  = args[i++];
    const int W  = args[i++];
    const int CI = args[i++];
    const int CO = args[i++];
    const int kernel_size = args[i++];
    const int strides     = GetArg(args, i++, 1);
    const int padding     = GetArg(args, i++, 0);
    const int dilation    = GetArg(args, i++, 1);
    const int groups      = GetArg(args, i++, 1);

    const int KD = kernel_size;
    const int KH = kernel_size;
    const int KW = kernel_size;
    const int SD = strides;
    const int SH = strides;
    const int SW = strides;
    const int PD = padding;
    const int PH = padding;
    const int PW = padding;
    const int DD = dilation;
    const int DH = dilation;
    const int DW = dilation;

    const int OD = (D + 2 * PD - (KD - 1) * DD - 1) / SD + 1;
    const int OH = (H + 2 * PH - (KH - 1) * DH - 1) / SH + 1;
    const int OW = (W + 2 * PW - (KW - 1) * DW - 1) / SW + 1;

    Buffer<float> input(CI, W, H, D, N);
    Buffer<float> filter(CO, CI / groups, KW, KH, KD);
    Buffer<float> output(CO, OW, OH, OD, N);

    input.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});
    filter.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});

    three_way_bench(
        [&]() { conv3d_ndhwc(input, filter, output); },
        [&]() { conv3d_ndhwc_classic_auto_schedule(input, filter, output); },
        [&]() { conv3d_ndhwc_auto_schedule(input, filter, output); }
    );

    return 0;
}
