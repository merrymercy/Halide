import numpy as np

import tvm
from common import matmul_nkkm, measure_schedule


N = 256
A, B, C = matmul_nkkm(N, N, N)

s = tvm.create_schedule([C.op])
y, x = s[C].op.axis
k, = s[C].op.reduce_axis
CC = s.cache_write(C, 'local')

y, yi = s[C].split(y, 16)
x, xi = s[C].split(x, 16)
s[C].reorder(y, x, yi, xi)
#xy = s[C].fuse(y, x)
#s[C].parallel(xy)
#s[CC].compute_at(s[C], xy)

s[C].parallel(y)
s[CC].compute_at(s[C], x)

y, x = s[CC].op.axis
k, = s[CC].op.reduce_axis
s[CC].reorder(k, y, x)
s[CC].vectorize(x)

print(tvm.lower(s, [A, B, C], simple_mode=True))
costs = measure_schedule(s, [A, B, C], 'llvm -mcpu=core-avx2')
print("%.4f ms" % (np.mean(costs) * 1e3))

