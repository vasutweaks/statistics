import numpy as np
import matplotlib.pyplot as plt
import math

def mean(lst):
    return sum(lst)/len(lst)

def std(numbers):
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std_dev = math.sqrt(variance)
    return std_dev

# Define the population
population = np.random.normal(loc=50, scale=10, size=100000)

# Define the sample size and number of samples to take
sample_size = 50
num_samples = 1000

# Take num_samples samples of size sample_size from the population
samples = [np.random.choice(population, size=sample_size) for _ in range(num_samples)]

# Compute the mean of each sample
sample_means = [np.mean(sample) for sample in samples]

# Plot the distribution of sample means
plt.hist(sample_means, bins=30, alpha=0.5, density=True)
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.title('Distribution of Sample Means (n={})'.format(sample_size))
# plt.show()

print(f"mean of sample means is: {mean(sample_means)}")
print(f"mean of population is: {population.mean()}") # this is close to above mean

standard_error_of_means=std(sample_means)
std_population=population.std()
std_by=std_population/math.sqrt(sample_size)
print(f"standard_error_of_means is: {standard_error_of_means}")
print(f"std of population devided by sqrt of sample size is: {std_by}")
