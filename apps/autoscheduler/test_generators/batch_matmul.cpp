#include "Halide.h"

#include "utils.h"

namespace {

class BatchMatmul : public Halide::Generator<BatchMatmul> {
public:
    std::vector<int> args = GetArgsFromEnv();
    int i = 0;
    const int B = args[i++];
    const int N = args[i++];
    const int M = args[i++];
    const int K = args[i++];

    Input<Buffer<float>>    input_a{"input_a", 3};
    Input<Buffer<float>>    input_b{"input_b", 3};

    Output<Buffer<float>>   output{"output", 3};

    void generate() {
        Var x("x"), y("y"), b("b");

        // Algorithm
        RDom k(0, K);

        output(x, y, b) = 0.0f;
        output(x, y, b) += input_a(k, y, b) * input_b(x, k, b);

        output.bound(x, 0, M)
              .bound(y, 0, N)
              .bound(b, 0, B);

        input_a.dim(0).set_bounds(0, K).set_stride(1)
               .dim(1).set_bounds(0, N).set_stride(K)
               .dim(2).set_bounds(0, B).set_stride(K * N);

        input_b.dim(0).set_bounds(0, M).set_stride(1)
               .dim(1).set_bounds(0, K).set_stride(M)
               .dim(2).set_bounds(0, B).set_stride(M * K);

        output.dim(0).set_bounds(0, M).set_stride(1)
              .dim(1).set_bounds(0, N).set_stride(M)
              .dim(2).set_bounds(0, B).set_stride(M * N);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(BatchMatmul, demo)
