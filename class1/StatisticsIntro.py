import numpy as np
import pandas as pd
from scipy.stats import binom, poisson, norm, uniform, expon
from scipy.stats import ttest_1samp, ttest_ind, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Example dataset
data = [10, 20, 20, 30, 40, 50, 60, 70, 80, 90]

# Mean
mean = np.mean(data)
print("Mean:", mean)

# Median
median = np.median(data)
print("Median:", median)

# Mode
mode = pd.Series(data).mode()[0]
print("Mode:", mode)

# Variance
variance = np.var(data)
print("Variance:", variance)

# Standard Deviation
std_dev = np.std(data)
print("Standard Deviation:", std_dev)

# Range
range_ = np.ptp(data)
print("Range:", range_)

# Interquartile Range (IQR)
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
print("Interquartile Range (IQR):", IQR)

# Binomial Distribution
n, p = 10, 0.5
binomial = binom.pmf(5, n, p)
print("Binomial PMF for k=5:", binomial)

# Poisson Distribution
lambda_ = 3
poisson_dist = poisson.pmf(5, lambda_)
print("Poisson PMF for k=5:", poisson_dist)

# Normal Distribution
mean, std_dev = 0, 1
normal_dist = norm.pdf(0, mean, std_dev)
print("Normal PDF for x=0:", normal_dist)

# Uniform Distribution
uniform_dist = uniform.pdf(0.5)
print("Uniform PDF for x=0.5:", uniform_dist)

# Exponential Distribution
lambda_ = 1
exponential_dist = expon.pdf(1, scale=1/lambda_)
print("Exponential PDF for x=1:", exponential_dist)

# One-sample t-test
sample_data = [2.3, 2.5, 2.8, 2.9, 3.0, 3.2, 3.3]
population_mean = 3.0
t_stat, p_value = ttest_1samp(sample_data, population_mean)
print("One-sample t-test p-value:", p_value)

# Two-sample t-test
sample_data1 = [2.3, 2.5, 2.8, 2.9, 3.0, 3.2, 3.3]
sample_data2 = [3.1, 3.3, 3.4, 3.5, 3.7, 3.8, 4.0]
t_stat, p_value = ttest_ind(sample_data1, sample_data2)
print("Two-sample t-test p-value:", p_value)

# Chi-square test
observed = [[10, 10, 20], [20, 20, 20]]
chi2_stat, p_value, dof, expected = chi2_contingency(observed)
print("Chi-square test p-value:", p_value)

# Data for visualization
data = np.random.normal(0, 1, 1000)

# Histogram
plt.hist(data, bins=30, edgecolor='k', alpha=0.7)
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Density Plot
sns.kdeplot(data, shade=True)
plt.title('Density Plot')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# Box Plot
sns.boxplot(data)
plt.title('Box Plot')
plt.xlabel('Value')
plt.show()
