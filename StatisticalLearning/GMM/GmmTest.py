from random import seed

import matplotlib.pyplot as plt
from numpy import concatenate, sqrt, zeros_like, pi, exp, power as pow, linspace
from numpy.random import normal
import seaborn as sns

from GMM import Gmm

if __name__ == '__main__':
    alpha = [1 / 6, 1 / 3, 1 / 2]
    mean = [1.5, -1, 0]
    sigma2 = [0.05, 0.2, 0.1]

    n = 600

    normal_pdf = lambda x, mean, sigma2: (1 / sqrt(2 * pi * sigma2)) * exp(-0.5 * pow(x - mean, 2) / sigma2)

    f = lambda x: alpha[0] * normal_pdf(x, mean[0], sigma2[0]) \
                  + alpha[1] * normal_pdf(x, mean[1], sigma2[1]) \
                  + alpha[2] * normal_pdf(x, mean[2], sigma2[2])

    seed(42)

    group1 = normal(loc=mean[0], scale=sqrt(sigma2[0]), size=int(n * alpha[0]))
    group2 = normal(loc=mean[1], scale=sqrt(sigma2[1]), size=int(n * alpha[1]))
    group3 = normal(loc=mean[2], scale=sqrt(sigma2[2]), size=int(n * alpha[2]))

    data = concatenate([group1, group2, group3], axis=0)

    gmm = Gmm(cluster=3,
              threshold=1e-4,
              log_likelihood_threshold=1e-1,
              alpha=None,  # alpha,
              mean=None,  # mean,
              sigma2=None  # sigma2
              )

    gmm_alpha, gmm_mean, gmm_sigma2 = gmm.fit(data)
    print("alpha(estimation): ", gmm_alpha)
    print("mean(estimation): ", gmm_mean)
    print("sigma2(estimation)", gmm_sigma2)

    gmm_f = lambda x: gmm_alpha[0] * normal_pdf(x, gmm_mean[0], gmm_sigma2[0]) \
                      + gmm_alpha[1] * normal_pdf(x, gmm_mean[1], gmm_sigma2[1]) \
                      + gmm_alpha[2] * normal_pdf(x, gmm_mean[2], gmm_sigma2[2])

    x = linspace(-3, 3, 200)

    sns.set(style='whitegrid')
    plt.figure(dpi=400)
    ax = plt.gca()
    ax.scatter(group1, zeros_like(group1), label='g1', alpha=0.5)
    ax.scatter(group2, zeros_like(group2), label='g2', alpha=0.5)
    ax.scatter(group3, zeros_like(group3), label='g3', alpha=0.5)
    ax.plot(x, normal_pdf(x, mean[0], sigma2[0]),
            label="$N(\mu_1, \sigma_1)$",
            alpha=0.5)
    ax.plot(x, normal_pdf(x, mean[1], sigma2[1]),
            label="$N(\mu_2, \sigma_2)$",
            alpha=0.3)
    ax.plot(x, normal_pdf(x, mean[2], sigma2[2]),
            label="$N(\mu_3, \sigma_3)$",
            alpha=0.3)
    ax.plot(x, f(x), label='ture mixture pdf')
    ax.plot(x, gmm_f(x), label='gmm')
    ax.set_xlabel("$x$")
    sns.despine()
    plt.legend()
    plt.title("$Gaussian\quad Mixture\quad Model$")
    plt.show()


    print(gmm.predict(group1))
    print(gmm.predict(group2))
    print(gmm.predict(group3))
