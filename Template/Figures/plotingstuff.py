import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

#Plotting parameters
plt.xlim([0, 1])
plt.ylim([0, 1])

#Gaussian parameters
mu = 0.3
variance = 0.3
sigma = math.sqrt(variance)
#sigma = 0.25

#Model parameters
v=0.4
K=0.4
a=0.90

#axes definition
theta_i = np.linspace(0, 1, 100)
x = np.linspace(-1/2,1,100)
K_list = [1-K]*100

#show x=0 axe
#plt.axvline(x=0, color='k')

#belief probability (function with 100 points)
qit = (0.5*v)/(0.5*v+(a*theta_i+(1-a)*(1-theta_i))*(1-v))

#distribution of prior opinions (2 different options)
plt.plot(x, stats.norm.pdf(x, mu, sigma),label=r'Gaussian pdf of $\theta_i$')
#plt.plot(theta_i, stats.norm.pdf(theta_i, mu, sigma))
#block curve
#plt.plot(theta_i,qit,"--",label=r'$U_{i}(B)$')
#send curve
#plt.plot(theta_i,1-qit,"-.",label=r'$U_{i}(S)$')
#check threshold
#plt.plot(theta_i,K_list,":",label=r'$U_{i}(C)$')


#bounds
idx = np.argwhere(np.diff(np.sign((1-qit) - K))).flatten()
plt.axvline((theta_i[idx]), 0, 1,linestyle="dashed",label=r'low bound')

idx2 = np.argwhere(np.diff(np.sign(qit - K))).flatten()
plt.axvline((theta_i[idx2]), 0, 1,linestyle="dotted",label=r'high bound')

#sections
section1 = np.arange(0,theta_i[idx],1/20.)
plt.fill_between(section1,stats.norm.pdf(section1,mu,sigma),alpha=0.35,label="Block region",color="red",edgecolor="b")

section2 = np.arange(theta_i[idx2],1,1/20.)
plt.fill_between(section2,stats.norm.pdf(section2,mu,sigma),alpha=0.35,label="Send region",color="royalblue",edgecolor="g")

section3 = np.arange(theta_i[idx],theta_i[idx2],1/50.)
plt.fill_between(section3,stats.norm.pdf(section3,mu,sigma),alpha=0.35,label="Check region",color="green",edgecolor="g")

#axes labels
plt.xlabel(r'prior opinion $\theta_i$')
#plt.ylabel('y axis label')

plt.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2,borderaxespad=0, frameon=False)
#plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
plt.legend()
plt.savefig('NameThisSample1.pdf')
plt.show()
