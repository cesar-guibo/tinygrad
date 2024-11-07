from tinygrad import Tensor
from tinygrad.dtype import dtypes

#for item in uniform1.matmul(uniform2).schedule_with_vars()[0]:
#    print(item)
#    print()
#
#a = Tensor.rand(100, 500)
#for item in a.schedule_with_vars()[0]:
#    print(item)
#    print()

a = Tensor([512 - i for i in range(512)], device="GPU")
c = Tensor([i for i in range(50000)], device="GPU")
b = Tensor([[i] for i in range(50000)], device="GPU")

x = a.flip(axis=0).cumsum(axis=0).flip(axis=0)
print(x.lazydata.lbs[0].st)
print(x.numpy())

