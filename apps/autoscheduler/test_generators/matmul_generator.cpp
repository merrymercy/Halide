#include "Halide.h"
#include <iostream>
#include <functional>
#include <string>
#include <vector>
#include <stdlib.h>

#include "../autoscheduler/utils.h"

namespace {

class MatMul : public Halide::Generator<MatMul> {
public:
    std::vector<int> args = GetArgsFromEnv();
    int i = 0;
    const int N = args[i++];
    const int M = args[i++];
    const int K = args[i++];

    Input<Buffer<float>>    input_a{"input_a", 2};
    Input<Buffer<float>>    input_b{"input_b", 2};
    Output<Buffer<float>>   output{"output", 2};

    void generate() {
        Var x("x"), y("y");

        // Algorithm
        RDom k(0, K);

        Func matmul("matrix_mul");
        matmul(x, y) += input_a(k, y) * input_b(x, k);
        output(x, y) = matmul(x, y);

        // Schedule
        if (!auto_schedule) {
            Var xi("xi"), yi("yi"), yii("yii"), xii("xii"), xy("xy");

            output.split(x, x, xi, 16)
                  .split(y, y, yi, 16)
                  .reorder(xi, yi, x, y)
                  .parallel(y);

            matmul.compute_at(output, x);

            matmul.update(0)
                  .reorder(x, y, k)
                  .vectorize(x);
        }

        output.bound(x, 0, M)
              .bound(y, 0, N);

        input_a.dim(0).set_bounds(0, K).set_stride(1)
               .dim(1).set_bounds(0, N).set_stride(K);

        input_b.dim(0).set_bounds(0, M).set_stride(1)
               .dim(1).set_bounds(0, K).set_stride(M);

        output.dim(0).set_bounds(0, M).set_stride(1)
              .dim(1).set_bounds(0, N).set_stride(M);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(MatMul, demo)
