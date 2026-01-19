from functools import partial
import matplotlib.pyplot as plt

def pmf(n,p):
    return p*(1-p)**(n-1)

pmf1=partial(pmf,p=0.5)
N=list(range(1000))
pmf_list=list(map(pmf1,N))

print(pmf_list[10])
print(pmf_list[15])
# plt.plot(N,pmf_list)
# plt.show()
