import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def mean(lst):
    return sum(lst)/len(lst)

def gaussian(x,mu,sigma):
    p=1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))
    return p

# define two samples
sample1 = [2, 4, 6, 8, 10]
sample2 = [1, 3, 5, 7, 9]

sample1=np.random.normal(1.0,0.2,1000) # two normally distributed samples with slight mean diff
sample2=np.random.normal(1.01,0.2,1000)

# perform two-sample t-test
t_stat, p_value = stats.ttest_ind(sample1, sample2)

# diff=mean(sample1)-mean(sample2)
diff=sample1.mean()-sample2.mean()

print(f"the difference between the sample means is: {diff}")
print(f"tstat and p value are {t_stat} {p_value}")
# print results
if p_value < 0.05:
    print("The difference between the sample means is significant.")
else:
    print("The difference between the sample means is not significant.")

count, bins, ignored = plt.hist(sample1, 30, density=True)
print(bins)
plt.plot(bins, gaussian(bins,1.0,0.2), linewidth=2, color='r')
plt.show()
