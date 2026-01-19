import pandas as pd
import numpy as np
import math

# 1. Load data
df = pd.read_csv('weight-height.csv')
male = df[df['Gender'] == 'Male']['Height'].values
female = df[df['Gender'] == 'Female']['Height'].values

# 2. Sample sizes
n_m = len(male)
n_f = len(female)

# 3. Sample means
mean_m = male.mean()
mean_f = female.mean()

# 4. Sample variances (unbiased)
var_m = male.var(ddof=1)
var_f = female.var(ddof=1)

# 5. Standard error of difference (Welch)
se = math.sqrt(var_m / n_m + var_f / n_f)

# 6. t-statistic
t_stat = (mean_m - mean_f) / se

# 7. Welch–Satterthwaite degrees of freedom
df_num = (var_m / n_m + var_f / n_f) ** 2
df_den = (var_m**2) / (n_m**2 * (n_m - 1)) + (var_f**2) / (n_f**2 * (n_f - 1))
df_welch = df_num / df_den

# 8. Approximate two-sided p-value using normal approximation
# (valid here due to very large df)
p_value = math.erfc(abs(t_stat) / math.sqrt(2))

mean_m, mean_f, var_m, var_f, t_stat, df_welch, p_value
