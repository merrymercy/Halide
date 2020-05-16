#include "Halide.h"
#include "utils.h"

namespace {

class TransposeBatchMatMul : public Halide::Generator<TransposeBatchMatMul> {
public:
    std::vector<int> args = GetArgsFromEnv();
    int i = 0;
    const int batch = args[i++];
    const int seq_len = args[i++];
    const int n_head = args[i++];
    const int n_dim = args[i++];

    Input<Buffer<float>>    query{"query", 4};
    Input<Buffer<float>>    value{"value", 4};
    Output<Buffer<float>>   output{"output", 4};

    void generate() {

        // Algorithm

        Func query_T("query_T");
        Func value_T("value_T");
        Func matmul("matmul");
        RDom k(0, n_dim);
        Var b("b"), h("h"), l("l"), d("d"), i("i"), j("j");

        query_T(d, l, h, b) = query(d, h, l, b);
        value_T(l, d, h, b) = value(d, h, l, b);

        matmul(j, i, h, b) += query_T(k, i, h, b) * value_T(j, k, h, b);
        output(j, i, h, b) = matmul(j, i, h, b);

        output.bound(j, 0, seq_len)
              .bound(i, 0, seq_len)
              .bound(h, 0, n_head)
              .bound(b, 0, batch);

        query.dim(0).set_bounds(0, n_dim).set_stride(1)
             .dim(1).set_bounds(0, n_head).set_stride(n_dim)
             .dim(2).set_bounds(0, seq_len).set_stride(n_dim * n_head)
             .dim(3).set_bounds(0, batch).set_stride(n_dim * n_head * seq_len);

        value.dim(0).set_bounds(0, n_dim).set_stride(1)
             .dim(1).set_bounds(0, n_head).set_stride(n_dim)
             .dim(2).set_bounds(0, seq_len).set_stride(n_dim * n_head)
             .dim(3).set_bounds(0, batch).set_stride(n_dim * n_head * seq_len);

        output.dim(0).set_bounds(0, seq_len).set_stride(1)
              .dim(1).set_bounds(0, seq_len).set_stride(seq_len)
              .dim(2).set_bounds(0, n_head).set_stride(seq_len * seq_len)
              .dim(3).set_bounds(0, batch).set_stride(seq_len * seq_len * n_head);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(TransposeBatchMatMul, demo)
