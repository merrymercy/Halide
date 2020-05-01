#include "Halide.h"
#include "utils.h"

namespace {

using namespace Halide;

Func pad(Func f, Expr length) {
    Halide::Region bounds(f.dimensions());
	bounds[1].min = 0;
	bounds[1].extent = length;
    return Halide::BoundaryConditions::constant_exterior(f, 0.0f, bounds);
}

class Conv1d : public Halide::Generator<Conv1d> {
public:
    Input<Buffer<float>>  input{"input", 3};
    Input<Buffer<float>>  filter{"filter", 3};

    Output<Buffer<float>> output{"output", 3};

    void generate() {
        std::vector<int> args = GetArgsFromEnv();
        int i = 0;
        const int N  = args[i++];
        const int L = args[i++];
        const int CI = args[i++];
        const int CO = args[i++];
        const int kernel_size = args[i++];
        const int strides     = GetArg(args, i++, 1);
        const int padding     = GetArg(args, i++, 0);
        const int dilation    = GetArg(args, i++, 1);
        const int groups      = GetArg(args, i++, 1);

        const int OL = (L + 2 * padding - (kernel_size - 1) * dilation - 1) / strides + 1;

        Var c("c"), l("l"), n("n");

        Func padded("pad");;

        if (padding) {
            padded = pad(input, L);
        } else {
            padded = input;
        }

        RDom r(0, CI / groups, 0, kernel_size);

        output(c, l, n) = 0.0f;
        output(c, l, n) += filter(c, r.y, r.x)
            * padded(c / (CO / groups) * (CI / groups) + r.x, l * strides + r.y * dilation - padding, n);

        output.bound(c, 0, CO)
              .bound(l, 0, OL)
              .bound(n, 0, N);

        input.dim(0).set_bounds(0, CI).set_stride(1)
             .dim(1).set_bounds(0,  L).set_stride(CI)
             .dim(2).set_bounds(0,  N).set_stride(CI * L);

        filter.dim(0).set_bounds(0, CO).set_stride(1)
              .dim(1).set_bounds(0, kernel_size).set_stride(CO)
              .dim(2).set_bounds(0, CI / groups).set_stride(CO * kernel_size);

        output.dim(0).set_bounds(0, CO).set_stride(1)
              .dim(1).set_bounds(0, OL).set_stride(CO)
              .dim(2).set_bounds(0,  N).set_stride(CO * OL);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Conv1d, demo)
