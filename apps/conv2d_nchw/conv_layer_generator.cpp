#include <stdlib.h>
#include "Halide.h"
#include "benchmark_util.h"

namespace {

using namespace Halide;

Func pad(Func f, Expr width, Expr height) {
	std::vector<std::pair<Expr, Expr>> bounds(f.dimensions());
	bounds[0].first = 0;
	bounds[0].second = width;
	bounds[1].first = 0;
	bounds[1].second = height;
	return Halide::BoundaryConditions::constant_exterior(f, 0.0f, bounds);
}

class ConvolutionLayer : public Halide::Generator<ConvolutionLayer> {
public:
    Input<Buffer<float>>  input{"input", 4};
    Input<Buffer<float>>  filter{"filter", 4};
    Input<Buffer<float>>  bias{"bias", 1};

    Output<Buffer<float>> f_ReLU{"ReLU", 4};

    void generate() {
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

        Var h("h"), w("w"), c("c"), n("n");

        Func f_conv("conv"), padded("pad");;

        if (PH || PW) {
            padded = pad(input, W, H);
        } else {
            padded = input;
        }

        RDom r(0, KW, 0, KH, 0, CI);

        f_conv(w, h, c, n) = bias(c);
        f_conv(w, h, c, n) += filter(r.x, r.y, r.z, c) * padded(w * SW + r.x - PW, h * SH + r.y - PH, r.z, n);
        f_ReLU(w, h, c, n) = max(0, f_conv(w, h, c, n));

        f_ReLU.bound(w, 0, OW)
              .bound(h, 0, OH)
              .bound(c, 0, CO)
              .bound(n, 0, N);

        input.dim(0).set_bounds(0,  W).set_stride(1)
             .dim(1).set_bounds(0,  H).set_stride(W)
             .dim(2).set_bounds(0, CI).set_stride(W * H)
             .dim(3).set_bounds(0,  N).set_stride(W * H * CI);

        filter.dim(0).set_bounds(0, KW).set_stride(1)
              .dim(1).set_bounds(0, KH).set_stride(KW)
              .dim(2).set_bounds(0, CI).set_stride(KW * KH)
              .dim(3).set_bounds(0, CO).set_stride(KW * KH * CI);

        bias.dim(0).set_bounds(0, CO).set_stride(1);

        f_ReLU.dim(0).set_bounds(0, OW).set_stride(1)
              .dim(1).set_bounds(0, OH).set_stride(OW)
              .dim(2).set_bounds(0, CO).set_stride(OW * OH)
              .dim(3).set_bounds(0,  N).set_stride(OW * OH * CO);

        if (auto_schedule) {
            // Provide estimates on the input image
            input.dim(0).set_bounds_estimate(0, W);
            input.dim(1).set_bounds_estimate(0, H);
            input.dim(2).set_bounds_estimate(0, CI);
            input.dim(3).set_bounds_estimate(0, N);

            filter.dim(0).set_bounds_estimate(0, KW);
            filter.dim(1).set_bounds_estimate(0, KH);
            filter.dim(2).set_bounds_estimate(0, CI);
            filter.dim(3).set_bounds_estimate(0, CO);

            bias.dim(0).set_bounds_estimate(0, CO);

            // Provide estimates on the pipeline f_ReLU
            f_ReLU.estimate(w, 0, OW)
                  .estimate(h, 0, OH)
                  .estimate(c, 0, CO)
                  .estimate(n, 0, N);
        } else {

        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ConvolutionLayer, conv_layer)
