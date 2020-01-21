#include <cstdio>
#include <chrono>

#include "conv_layer.h"
#include "conv_layer_classic_auto_schedule.h"
#include "conv_layer_auto_schedule.h"

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
    const int KH = args[i++];
    const int KW = args[i++];
    const int SH = args[i++];
    const int SW = args[i++];
    const int PH = args[i++];
    const int PW = args[i++];

    const int OH = (H + 2 * PH - KH) / SH + 1;
    const int OW = (W + 2 * PW - KW) / SW + 1;

    Buffer<float> input(W, H, CI, N);
    Buffer<float> filter(KH, KW, CI, CO);
    Buffer<float> bias(CO);
    Buffer<float> output(OW, OH, CO, N);

    input.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});
    filter.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});
    bias.for_each_value([&](float &f) {f = (double)rand()/RAND_MAX;});

    three_way_bench(
        [&]() { conv_layer(input, filter, bias, output); },
        [&]() { conv_layer_classic_auto_schedule(input, filter, bias, output); },
        [&]() { conv_layer_auto_schedule(input, filter, bias, output); }
    );

    return 0;
}
