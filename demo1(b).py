import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom,poisson,norm,expon
#poisson
lambda_param = 5
k = 3
prob_3= poisson.pmf(lambda_param,k)
print(f"probab of 3: {prob_3}")


n=10
p=0.6
k_success = 7
prob_7= binom.pmf(k_success,n,p)
print(f"probab of 7: {prob_7}")

exp_samp= np.random.exponential(scale=2, size=1000)
plt.figure(figsize=(8,6))
plt.hist(exp_samp, bins=30, density=True, alpha=0.6, color='green')
x_exp = np.linspace(0, 10, 100)
plt.plot(x_exp, expon.pdf(x_exp,scale=2), 'r-', lw=2, label="exponential distubution")
plt.xlabel('value')
plt.ylabel('prob')
plt.legend()
plt.grid(True)
plt.show()
