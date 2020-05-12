import numpy as np

import tvm
import topi

N, H, W, CI, CO, KH, SH, PH = 1, 56, 56, 3, 64, 3, 2, 1
KW = KH
SW = SH
PW = PH

A = tvm.placeholder((N, H, W, CI), name='A')
B = tvm.placeholder((KH, KW, CI, CO), name='B')
C = topi.nn.conv2d_nhwc(A, B, stride=(SH, SW), padding=(PH, PW), dilation=1)

s = tvm.create_schedule([C.op])
func = tvm.build(s, [A, B, C], target='llvm')


a_np = np.ones(topi.get_const_tuple(A.shape)).astype(np.float32)
b_np = np.ones(topi.get_const_tuple(B.shape)).astype(np.float32)
b_np[:,0,:,:] = 2
c_np = np.ones(topi.get_const_tuple(C.shape)).astype(np.float32)

a_tvm = tvm.nd.array(a_np)
b_tvm = tvm.nd.array(b_np)
c_tvm = tvm.nd.array(c_np)

print(c_np.sum())

func(a_tvm, b_tvm, c_tvm)
c_np = c_tvm.asnumpy()

print(c_np.sum())
print(c_np[0,1,1,1])

