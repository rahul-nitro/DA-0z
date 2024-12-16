total_p_outcomes = 6
favourable_ot = 1
probaility_4= favourable_ot / total_p_outcomes 
print(f"pobabilty of rolling a {probaility_4}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,poisson,binom,expon

mean = 50
std_dev = 10
samples = np.random.normal(mean,std_dev,1000)
plt.figure(figsize=(8,6))
plt.hist(samples,bins=30,density=True,alpha=0.6,color='blue')
x = np.linspace(mean-4*std_dev , mean+4*std_dev,100)
plt.plot(x,norm.pdf(x,mean,std_dev),'r-',lw=2,label="normal Distribution")
plt.title("Normal Distribution Example(Quality Control)")
plt.xlabel('values')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
