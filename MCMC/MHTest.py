from matplotlib import pyplot as plt
from numpy import mean
from numpy.random import uniform, normal
from scipy.stats import beta
import seaborn as sns

from proposal_distribution import Proposal_Distribution
from target_distribution import Target_Distribution
from MCMC import Metropolis_Hastings as MH

n = 0
# n represents the number of samples which comes from the first normal distribution.
xs = []
# sampling from 0.2*N(0, 1) + 0.8*N(5, 1), append it to xs.
for i in range(30):
    u = uniform(0, 1)
    if u < 0.8:
        x = normal(5, 1)
        xs.append(x)
    else:
        x = normal(0, 1)
        xs.append(x)
        n += 1


# the posterior is proportional to p^n*(1-p)^(30 - n). That is beta distribution
class TD(Target_Distribution):
    @staticmethod
    def pdf(x):
        return beta.pdf(x, n + 1, 30 - n + 1)


class PD(Proposal_Distribution):
    @staticmethod
    def pdf(x, y):
        return 1

    @staticmethod
    def sampling():
        return uniform(0, 1)


if __name__ == '__main__':
    mh = MH(burning_period=100,
            end_period=5000,
            x0=0.5,
            target_distribution=TD,
            proposal_distribution=PD)

    samples = mh.run()

    print(len(samples))

    sns.set()

    plt.figure(dpi=400)
    ax1 = plt.subplot(121)
    ax1.plot(samples)
    ax1.set_xlabel("time")
    ax1.set_ylabel("samples")

    ax2 = plt.subplot(122)
    ax2.hist(samples,
             bins=40, histtype='stepfilled',
             density=True, alpha=0.3, color='orange')
    ax2.set_xlabel('p')
    ax2.set_ylabel('density')
    sns.despine(right=True, top=True)
    plt.title("MCMC")
    plt.show()

    # calculate the expectation of samples
    print("p estimation:", mean(samples))
