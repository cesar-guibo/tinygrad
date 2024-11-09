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

a = Tensor([[i for i in range(100)] for _ in range(50000)], device="GPU")
c = Tensor([i for i in range(50000)], device="GPU")
b = Tensor([[i] for i in range(50000)], device="GPU")

#print((a.sum(axis=0)).numpy())
print((a.cumsum(axis=0)).numpy())

