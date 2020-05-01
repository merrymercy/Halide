#include "Halide.h"
#include "utils.h"

namespace {

using namespace Halide;

Func pad(Func f, Expr width, Expr height) {
    Halide::Region bounds(f.dimensions());
    bounds[3].min = 0;
    bounds[3].extent = width;
    bounds[4].min = 0;
    bounds[4].extent = height;
    return Halide::BoundaryConditions::constant_exterior(f, 0.0f, bounds);
}

class Conv2d : public Halide::Generator<Conv2d> {
public:
    Input<Buffer<float>>  input{"input", 6};
    Input<Buffer<float>>  filter{"filter", 6};

    Output<Buffer<float>> output{"output", 6};

    void generate() {
        std::vector<int> args = GetArgsFromEnv();
        int i = 0;
        const int N  = args[i++];
        const int H  = args[i++];
        const int W  = args[i++];
        const int CI = args[i++];
        const int CO = args[i++];
        const int kernel_size  = args[i++];
        const int strides      = GetArg(args, i++, 1);
        const int padding      = GetArg(args, i++, 0);
        const int capsule_size = GetArg(args, i++, 4);

        const int KH = kernel_size;
        const int KW = kernel_size;
        const int SH = strides;
        const int SW = strides;
        const int PH = padding;
        const int PW = padding;

        const int OH = (H + 2 * PH - KH) / SH + 1;
        const int OW = (W + 2 * PW - KW) / SW + 1;

        Var c("c"), cap_j("cap_j"), cap_i("cap_i"), w("w"), h("h"), n("n");

        Func f_conv("conv"), padded("pad");

        if (PH || PW) {
            padded = pad(input, W, H);
        } else {
            padded = input;
        }

        RDom r(0, KW, 0, KH, 0, capsule_size, 0, CI);

        output(c, cap_j, cap_i, w, h, n) = 0.0f;
        output(c, cap_j, cap_i, w, h, n) += filter(c, r.w, cap_j, r.z, r.x, r.y)
            * padded(r.w, r.z, cap_i, w * SW + r.x - PW, h * SH + r.y - PH, n);

        output.bound(c, 0, CO)
              .bound(cap_j, 0, capsule_size)
              .bound(cap_i, 0, capsule_size)
              .bound(w, 0, OW)
              .bound(h, 0, OH)
              .bound(n, 0, N);

        input.dim(0).set_bounds(0, CI).set_stride(1)
             .dim(1).set_bounds(0, capsule_size).set_stride(CI)
             .dim(2).set_bounds(0, capsule_size).set_stride(CI * capsule_size)
             .dim(3).set_bounds(0,  W).set_stride(CI * capsule_size * capsule_size)
             .dim(4).set_bounds(0,  H).set_stride(CI * capsule_size * capsule_size * W)
             .dim(5).set_bounds(0,  N).set_stride(CI * capsule_size * capsule_size * W * H);

        filter.dim(0).set_bounds(0, CO).set_stride(1)
              .dim(1).set_bounds(0, CI).set_stride(CO)
              .dim(2).set_bounds(0, capsule_size).set_stride(CO * CI)
              .dim(3).set_bounds(0, capsule_size).set_stride(CO * CI * capsule_size)
              .dim(4).set_bounds(0, KW).set_stride(CO * CI * capsule_size * capsule_size)
              .dim(5).set_bounds(0, KH).set_stride(CO * CI * capsule_size * capsule_size * KW);

        output.dim(0).set_bounds(0, CO).set_stride(1)
              .dim(1).set_bounds(0, capsule_size).set_stride(CO)
              .dim(2).set_bounds(0, capsule_size).set_stride(CO * capsule_size)
              .dim(3).set_bounds(0, OW).set_stride(CO * capsule_size * capsule_size)
              .dim(4).set_bounds(0, OH).set_stride(CO * capsule_size * capsule_size * OW)
              .dim(5).set_bounds(0,  N).set_stride(CO * capsule_size * capsule_size * OW * OH);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Conv2d, demo)
