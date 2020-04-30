#include "Halide.h"
#include "benchmark_util.h"

namespace {

class TransposeBatchMatMul : public Halide::Generator<TransposeBatchMatMul> {
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

        Func matrix_trans("matrix_trans");
        Func matrix_mul("matrix_mul");

        matrix_trans(x, y, b) = input_b(y, x, b);

        matrix_mul(x, y, b) += input_a(k, y, b) * matrix_trans(x, k, b);

        output(x, y, b) = matrix_mul(x, y, b);

        output.bound(x, 0, M)
              .bound(y, 0, N)
              .bound(b, 0, B);

        input_a.dim(0).set_bounds(0, K).set_stride(1)
               .dim(1).set_bounds(0, N).set_stride(K)
               .dim(2).set_bounds(0, B).set_stride(K * N);

        input_b.dim(0).set_bounds(0, K).set_stride(1)
               .dim(1).set_bounds(0, M).set_stride(K)
               .dim(2).set_bounds(0, B).set_stride(K * M);

        output.dim(0).set_bounds(0, M).set_stride(1)
              .dim(1).set_bounds(0, N).set_stride(M)
              .dim(2).set_bounds(0, B).set_stride(M * N);

        // Estimates
        {
            input_a.dim(0).set_bounds_estimate(0, K)
                   .dim(1).set_bounds_estimate(0, N)
                   .dim(2).set_bounds_estimate(0, B);

            input_b.dim(0).set_bounds_estimate(0, K)
                   .dim(1).set_bounds_estimate(0, M)
                   .dim(2).set_bounds_estimate(0, B);

            output.estimate(x, 0, M)
                  .estimate(y, 0, N)
                  .estimate(b, 0, B);
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(TransposeBatchMatMul, transpose_batch_matmul)
