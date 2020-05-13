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
        const int dilation    = GetArg(args, i++, 1);
        const int groups      = GetArg(args, i++, 1);

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

        Func padded("pad");

        if (PH || PW) {
            padded = pad(input, W, H);
        } else {
            padded = input;
        }

        RDom r(0, CI / groups, 0, KW, 0, KH);

        Func func("func");
        func(c, w, h, n) = 0.0f;
        func(c, w, h, n) += filter(c, r.y, r.z, r.x)
            * padded(c / (CO / groups) * (CI / groups) + r.x, w * SW + r.y * DW - PW, h * SH + r.z * DH - PH, n);
        output(c, w, h, n) = func(c, w, h, n);

        output.bound(c, 0, CO)
              .bound(w, 0, OW)
              .bound(h, 0, OH)
              .bound(n, 0, N);

        input.dim(0).set_bounds(0, CI).set_stride(1)
             .dim(1).set_bounds(0,  W).set_stride(CI)
             .dim(2).set_bounds(0,  H).set_stride(CI * W)
             .dim(3).set_bounds(0,  N).set_stride(CI * W * H);

        filter.dim(0).set_bounds(0, CO).set_stride(1)
              .dim(1).set_bounds(0, KW).set_stride(CO)
              .dim(2).set_bounds(0, KH).set_stride(CO * KW)
              .dim(3).set_bounds(0, CI / groups).set_stride(CO * KH * KW);

        output.dim(0).set_bounds(0, CO).set_stride(1)
              .dim(1).set_bounds(0, OW).set_stride(CO)
              .dim(2).set_bounds(0, OH).set_stride(CO * OW)
              .dim(3).set_bounds(0,  N).set_stride(CO * OW * OH);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Conv2d, demo)
