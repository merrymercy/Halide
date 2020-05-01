#include "Halide.h"
#include "utils.h"

namespace {

using namespace Halide;

Func pad(Func f, Expr width, Expr height) {
    Halide::Region bounds(f.dimensions());
    bounds[1].min = 0;
    bounds[1].extent = width;
    bounds[2].min = 0;
    bounds[2].extent = height;
    return Halide::BoundaryConditions::constant_exterior(f, 0.0f, bounds);
}

class Conv2d : public Halide::Generator<Conv2d> {
public:
    Input<Buffer<float>>  input{"input", 4};
    Input<Buffer<float>>  filter{"filter", 4};

    Output<Buffer<float>> output{"output", 4};

    void generate() {
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

        Var c("c"), w("w"), h("h"), n("n");

        Func f_conv_trans("conv_trans"), padded("pad");

        const int bpad_top  = KH - 1 - padding;
        const int bpad_left = KW - 1 - padding;
        const int PH = (bpad_top + SH - 1) / SH;
        const int PW = (bpad_left + SW - 1) / SW;

        padded = pad(input, W, H);

        const int BH = (SH - bpad_top % SH) % SH;
        const int BW = (SW - bpad_left % SW) % SW;

        RDom r(0, CI, 0, KW, 0, KH);

        output(c, w, h, n) = 0.0f;
        output(c, w, h, n) += filter(c, r.x, KW - 1 - r.y, KH - 1 - r.z)
            * padded(r.x, (w + r.y + BW) / SW - PW, (h + r.z + BH) / SH - PH, n);

        output.bound(c, 0, CO)
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

        output.dim(0).set_bounds(0, CO).set_stride(1)
              .dim(1).set_bounds(0, OW).set_stride(CO)
              .dim(2).set_bounds(0, OH).set_stride(CO * OW)
              .dim(3).set_bounds(0,  N).set_stride(CO * OW * OH);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Conv2d, demo)
