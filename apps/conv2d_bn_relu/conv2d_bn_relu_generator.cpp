#include <stdlib.h>
#include "Halide.h"
#include "benchmark_util.h"

namespace {

using namespace Halide;

Func pad(Func f, Expr width, Expr height) {
	std::vector<std::pair<Expr, Expr>> bounds(f.dimensions());
	bounds[1].first = 0;
	bounds[1].second = width;
	bounds[2].first = 0;
	bounds[2].second = height;
	return Halide::BoundaryConditions::constant_exterior(f, 0.0f, bounds);
}

class Conv2dBnRelu : public Halide::Generator<Conv2dBnRelu> {
public:
    Input<Buffer<float>>  input{"input", 4};
    Input<Buffer<float>>  filter{"filter", 4};
    Input<Buffer<float>>  bias{"bias", 1};
    Input<Buffer<float>> bn_scale{"bn_scale", 1};
    Input<Buffer<float>> bn_offset{"bn_offset", 1};

    Output<Buffer<float>> f_ReLU{"ReLU", 4};

    void generate() {
        std::vector<int> args = GetArgsFromEnv();
        int i = 0;
        const int N  = args[i++];
        const int H  = args[i++];
        const int W  = args[i++];
        const int CI = args[i++];
        const int CO = args[i++];
        const int kernel_size = args[i++];
        const int strides = args[i++];
        const int padding = args[i++];
        const int dilation = GetArg(args, i++, 1);

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

        Var c("c"), w("w"), h("h"), n("n");

        Func f_bn("bn"), f_conv("conv"), padded("pad");

        if (PH || PW) {
            padded = pad(input, W, H);
        } else {
            padded = input;
        }

        RDom r(0, KW, 0, KH, 0, CI);

        f_conv(c, w, h, n) = bias(c);
        f_conv(c, w, h, n) += filter(c, r.z, r.x, r.y) * padded(r.z, w * SW + r.x * DW - PW, h * SH + r.y * DH - PH, n);

        f_bn(c, w, h, n) = f_conv(c, w, h, n) * bn_scale(c);
        f_bn(c, w, h, n) += bn_offset(c);

        f_ReLU(c, w, h, n) = max(0, f_bn(c, w, h, n));

        f_ReLU.bound(c, 0, CO)
              .bound(w, 0, OW)
              .bound(h, 0, OH)
              .bound(n, 0, N);

        input.dim(0).set_bounds(0, CI).set_stride(1)
             .dim(1).set_bounds(0,  W).set_stride(CI)
             .dim(2).set_bounds(0,  H).set_stride(CI * W)
             .dim(3).set_bounds(0,  N).set_stride(CI * W * H);

        filter.dim(0).set_bounds(0, CO).set_stride(1)
              .dim(1).set_bounds(0, CI).set_stride(CO)
              .dim(2).set_bounds(0, KW).set_stride(CO * CI)
              .dim(3).set_bounds(0, KH).set_stride(CO * CI * KW);

        bias.dim(0).set_bounds(0, CO).set_stride(1);

        bn_scale.dim(0).set_bounds(0, CO).set_stride(1);

        bn_offset.dim(0).set_bounds(0, CO).set_stride(1);

        f_ReLU.dim(0).set_bounds(0, CO).set_stride(1)
              .dim(1).set_bounds(0, OW).set_stride(CO)
              .dim(2).set_bounds(0, OH).set_stride(CO * OW)
              .dim(3).set_bounds(0,  N).set_stride(CO * OW * OH);

        if (auto_schedule) {
            // Provide estimates on the input image
            input.dim(0).set_bounds_estimate(0, CI);
            input.dim(1).set_bounds_estimate(0, W);
            input.dim(2).set_bounds_estimate(0, H);
            input.dim(3).set_bounds_estimate(0, N);

            filter.dim(0).set_bounds_estimate(0, CO);
            filter.dim(1).set_bounds_estimate(0, CI);
            filter.dim(2).set_bounds_estimate(0, KW);
            filter.dim(3).set_bounds_estimate(0, KH);

            bias.dim(0).set_bounds_estimate(0, CO);

            bn_scale.dim(0).set_bounds_estimate(0, CO);

            bn_offset.dim(0).set_bounds_estimate(0, CO);

            // Provide estimates on the pipeline f_ReLU
            f_ReLU.estimate(c, 0, CO)
                  .estimate(w, 0, OW)
                  .estimate(h, 0, OH)
                  .estimate(n, 0, N);
        } else {

        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Conv2dBnRelu, conv2d_bn_relu)
