#include "Halide.h"
#include "utils.h"

namespace {

class BatchNorm : public Halide::Generator<BatchNorm> {
public:
    std::vector<int> args = GetArgsFromEnv();
    int i = 0;
    const int B = args[i++];
    const int M = args[i++];
    const int N = args[i++];

    Input<Buffer<float>>    input{"input", 3};
    Output<Buffer<float>>   output{"output", 1};

    void generate() {
        Var x("x"), y("y"), b("b");

        // Algorithm
        RDom k(0, N, 0, M);

        Func matrix_mul("matrix_mul");

        matrix_mul(b) = 0.0f;
        matrix_mul(b) += input(k.x, k.y, b) * input(k.x, k.y, b);

        output(b) = sqrt(matrix_mul(b));

        output.bound(b, 0, B);

        input.dim(0).set_bounds(0, N).set_stride(1)
             .dim(1).set_bounds(0, M).set_stride(N)
             .dim(2).set_bounds(0, B).set_stride(N * M);

        output.dim(0).set_bounds(0, B).set_stride(1);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(BatchNorm, demo)
