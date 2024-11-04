from tinygrad import Tensor

#uniform1 = Tensor.uniform(100, 500, low=0, high=10)
#uniform2 = Tensor.uniform(500, 100, low=0, high=10)

#for item in uniform1.matmul(uniform2).schedule_with_vars()[0]:
#    print(item)
#    print()
#
#a = Tensor.rand(100, 500)
#for item in a.schedule_with_vars()[0]:
#    print(item)
#    print()

a = Tensor([[i for i in range(100)] for _ in range(5000)], device="GPU")
b = Tensor([[i] for i in range(50000)], device="GPU")

#print((a + b.expand(a.shape)).numpy())
#print((a.sum(axis=0)).numpy())
print((a.cumsum(axis=0)).numpy())