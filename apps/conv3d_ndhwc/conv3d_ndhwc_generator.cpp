#include <stdlib.h>
#include "Halide.h"
#include "benchmark_util.h"

namespace {

using namespace Halide;

Func pad(Func f, Expr width, Expr height, Expr depth) {
	std::vector<std::pair<Expr, Expr>> bounds(f.dimensions());
	bounds[1].first = 0;
	bounds[1].second = width;
	bounds[2].first = 0;
	bounds[2].second = height;
	bounds[3].first = 0;
	bounds[3].second = depth;
	return Halide::BoundaryConditions::constant_exterior(f, 0.0f, bounds);
}

class Conv3d : public Halide::Generator<Conv3d> {
public:
    Input<Buffer<float>>  input{"input", 5};
    Input<Buffer<float>>  filter{"filter", 5};

    Output<Buffer<float>> output{"output", 5};

    void generate() {
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
        const int in_channel_per_group = CI / groups;
        const int out_channel_per_group = CO / groups;

        Var c("c"), w("w"), h("h"), d("d"), n("n");

        Func f_conv("conv"), padded("pad");

        if (PD || PH || PW) {
            padded = pad(input, W, H, D);
        } else {
            padded = input;
        }

        RDom r(0, KW, 0, KH, 0, KD, 0, in_channel_per_group);

        f_conv(c, w, h, d, n) = 0.0f;
        f_conv(c, w, h, d, n) += filter(c, r.w, r.x, r.y, r.z)
            * padded(c / out_channel_per_group * in_channel_per_group + r.w,
                     w * SW + r.x * DW - PW,
                     h * SH + r.y * DH - PH,
                     d * SD + r.z * DD - PD,
                     n);
        output(c, w, h, d, n) = f_conv(c, w, h, d, n);

        output.bound(c, 0, CO)
              .bound(w, 0, OW)
              .bound(h, 0, OH)
              .bound(d, 0, OD)
              .bound(n, 0, N);

        input.dim(0).set_bounds(0, CI).set_stride(1)
             .dim(1).set_bounds(0,  W).set_stride(CI)
             .dim(2).set_bounds(0,  H).set_stride(CI * W)
             .dim(3).set_bounds(0,  D).set_stride(CI * W * H)
             .dim(4).set_bounds(0,  N).set_stride(CI * W * H * D);

        filter.dim(0).set_bounds(0, CO).set_stride(1)
              .dim(1).set_bounds(0, in_channel_per_group).set_stride(CO)
              .dim(2).set_bounds(0, KW).set_stride(CO * in_channel_per_group)
              .dim(3).set_bounds(0, KH).set_stride(CO * in_channel_per_group * KW)
              .dim(4).set_bounds(0, KD).set_stride(CO * in_channel_per_group * KW * KH);

        output.dim(0).set_bounds(0, CO).set_stride(1)
              .dim(1).set_bounds(0, OW).set_stride(CO)
              .dim(2).set_bounds(0, OH).set_stride(CO * OW)
              .dim(3).set_bounds(0, OD).set_stride(CO * OW * OH)
              .dim(4).set_bounds(0,  N).set_stride(CO * OW * OH * OD);

        if (auto_schedule) {
            // Provide estimates on the input image
            input.dim(0).set_bounds_estimate(0, CI);
            input.dim(1).set_bounds_estimate(0, W);
            input.dim(2).set_bounds_estimate(0, H);
            input.dim(3).set_bounds_estimate(0, D);
            input.dim(4).set_bounds_estimate(0, N);

            filter.dim(0).set_bounds_estimate(0, CO);
            filter.dim(1).set_bounds_estimate(0, in_channel_per_group);
            filter.dim(2).set_bounds_estimate(0, KW);
            filter.dim(3).set_bounds_estimate(0, KH);
            filter.dim(4).set_bounds_estimate(0, KD);

            // Provide estimates on the pipeline output
            output.estimate(c, 0, CO)
                  .estimate(w, 0, OW)
                  .estimate(h, 0, OH)
                  .estimate(d, 0, OD)
                  .estimate(n, 0, N);
        } else {

        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Conv3d, conv3d_ndhwc)
