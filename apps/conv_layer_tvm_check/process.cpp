#include <chrono>
#include <cstdio>

#include "conv_layer.h"
#include "conv_layer_auto_schedule.h"

#include "HalideBuffer.h"
#include "halide_benchmark.h"

#include "utils.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    //Buffer<float> input(CI, W + 2, H + 2, N);
    //Buffer<float> filter(CO, 3, 3, CI);
    //Buffer<float> bias(CO);

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
        //const int groups      = GetArg(args, i++, 1);

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



    Buffer<float>  input(CI, W, H, N);
    Buffer<float>  filter(CO, KW, KH, CI);
    Buffer<float> output(CO, OW, OH, N);


    for (int c = 0; c < input.dim(3).extent(); c++) {
        for (int z = 0; z < input.dim(2).extent(); z++) {
            for (int y = 0; y < input.dim(1).extent(); y++) {
                for (int x = 0; x < input.dim(0).extent(); x++) {
                    input(x, y, z, c) = 1.0;
                }
            }
        }
    }

    for (int c = 0; c < filter.dim(3).extent(); c++) {
        for (int z = 0; z < filter.dim(2).extent(); z++) {
            for (int y = 0; y < filter.dim(1).extent(); y++) {
                for (int x = 0; x < filter.dim(0).extent(); x++) {
                    if (z == 0) {
                        filter(x, y, z, c) = 2.0;
                    } else {
                        filter(x, y, z, c) = 1.0;
                    }
                }
            }
        }
    }

    for (int i = 0; i < 4; i++) {
      std::cerr << input.dim(i).extent() << " ";
    }
    std::cerr << std::endl;

    for (int i = 0; i < 4; i++) {
      std::cerr << filter.dim(i).extent() << " ";
    }
    std::cerr << std::endl;

    for (int i = 0; i < 4; i++) {
      std::cerr << output.dim(i).extent() << " ";
    }
    std::cerr << std::endl;


// This is necessary to get the PTX compiler to do a good
// job. TODO: This should be a scheduling directive or a runtime
// function.
#ifdef _WIN32
    _putenv_s("HL_CUDA_JIT_MAX_REGISTERS", "256");
#else
    setenv("HL_CUDA_JIT_MAX_REGISTERS", "256", 1);
#endif

    conv_layer(input, filter, output);

    double sum = 0;
    for (int c = 0; c < output.dim(3).extent(); c++) {
        for (int z = 0; z < output.dim(2).extent(); z++) {
            for (int y = 0; y < output.dim(1).extent(); y++) {
                for (int x = 0; x < output.dim(0).extent(); x++) {
                    sum += output(x, y, z, c);
                }
            }
        }
    }
    std::cerr << "sum: " << sum << std::endl;
    std::cerr << "output(1, 1, 1, 0): " << output(1, 1, 1, 0) << std::endl;

    // Timing code

    // Manually-tuned version
    double min_t_manual = benchmark(10, 10, [&]() {
        conv_layer(input, filter, output);
        output.device_sync();
    });
    printf("Manually-tuned time: %gms\n", min_t_manual * 1e3);

    // Auto-scheduled version
    double min_t_auto = benchmark(10, 10, [&]() {
        conv_layer_auto_schedule(input, filter, output);
        output.device_sync();
    });
    printf("Auto-scheduled time: %gms\n", min_t_auto * 1e3);

    return 0;
}
