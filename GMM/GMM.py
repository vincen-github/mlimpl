from numpy import ones, ndarray, var, ones_like, exp, power as pow, pi, sqrt, asarray, concatenate, log, zeros, argmax
from numpy.linalg import norm
from numpy.random import uniform

import logging

from typing import List

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s -  %(asctime)s - %(filename)s -  %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)


class DimensionalError(Exception):
    pass


class Gmm(object):
    """
    Gaussian Mixture Model(GMM) - single dimension version

    @author: vincen

    @NorthWestern University——CHINA Mathematics faculty statistic

    @Github: https://github.com/vincen-github/Machine-Learning-Code

    @MyEmail : vincen.nwu@gmail.com

    @Reference: Expectation Maximization —— A/Prof Richard Yi Da Xu Yida.Xu@uts.edu.au

    ------------------------------------------------

    Example:
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
    """

    def __init__(self, cluster: int,
                 threshold: (float, int),
                 log_likelihood_threshold: (float, int),
                 alpha: (List[float], ndarray) = None,
                 mean: (List[float], ndarray) = None,
                 sigma2: (List[float], ndarray) = None) -> None:
        # cluster represent the number of gaussian density.
        if not isinstance(cluster, int):
            raise TypeError
        if cluster < 0:
            raise ValueError
        self.cluster = cluster

        if not isinstance(threshold, (int, float)):
            raise TypeError
        if not threshold > 0:
            raise ValueError
        self.threshold = threshold

        if not isinstance(log_likelihood_threshold, (int, float)):
            raise TypeError
        if not log_likelihood_threshold > 0:
            raise ValueError
        self.log_likelihood_threshold = log_likelihood_threshold

        self.alpha = alpha
        self.mean = mean
        self.sigma2 = sigma2

    @staticmethod
    def normal_pdf(x, mean, sigma2):
        return (1 / sqrt(2 * pi * sigma2)) * exp(- pow(x - mean, 2) / (2 * sigma2))

    def log_likelihood(self, X, alpha, mean, sigma2):
        res = 0
        for i in range(len(X)):
            likelihood = 0
            for l in range(self.cluster):
                likelihood += alpha[l] * self.normal_pdf(X[i], mean=mean[l], sigma2=sigma2[l])
            res += log(likelihood)
        return res

    def fit(self, X: (list, ndarray)) -> tuple:
        X = asarray(X)
        nrow = len(X)

        # initial alpha, mean and sigma2
        self.alpha = ones(self.cluster) / self.cluster
        # generate mean randomly if mean isn't passed by user(i.e mean = None).
        if self.mean is None:
            self.mean = uniform(low=X.min(), high=X.max(), size=self.cluster)
        if self.sigma2 is None:
            # initial sigma2 as the variance of overall data
            self.sigma2 = var(X) * ones_like(self.mean)

        iterNum = 0
        while True:
            log_likelihood_old = self.log_likelihood(X, alpha=self.alpha,
                                                     mean=self.mean, sigma2=self.sigma2)
            logger.info(
                """iterNum - {} \n
                alpha(estimation) : {} \n
                mean(estimation) : {} \n
                sigma2(estimation): {} \n
                log-likelihood: {}""".format(iterNum,
                                             self.alpha,
                                             self.mean,
                                             self.sigma2,
                                             log_likelihood_old))
            iterNum += 1
            alpha_old, mean_old, sigma_old = self.alpha.copy(), self.mean.copy(), self.sigma2.copy()

            # update alpha, mean, sigma2
            responsibility_prob = zeros(shape=(nrow, self.cluster))
            # responsibility_prob[i][l] = α^(g)_l*N(x_i|μ^(g)_l, σ^(g)_l) / Σ^k_{j=1} α^(g)_j * N(x_i|μ^(g)_j, σ^(g)_j)
            # it is a matrix which has nrow lines and self.cluster columns
            for i in range(nrow):
                for l in range(self.cluster):
                    # compute numerator
                    responsibility_prob[i][l] = self.alpha[l] * self.normal_pdf(X[i], mean=self.mean[l],
                                                                                sigma2=self.sigma2[l])
                # Divide each row by the corresponding denominator -> Σ^k_{j=1} α^(g)_j * N(x_i|μ^(g)_j, σ^(g)_j)
                responsibility_prob[i] = responsibility_prob[i] / responsibility_prob[i].sum()

            # update alpha, mean, sigma2
            for l in range(self.cluster):
                # αl = 1/n * Σ^n_{i = 1} pil
                self.alpha[l] = responsibility_prob[:, l].mean()
                # μl = (Σ^n_{i = 1} x_i*pil) / (Σ^n_{i = 1} pil)
                self.mean[l] = X @ responsibility_prob[:, l] / responsibility_prob[:, l].sum()
                # sigma_l = (Σ^n_{i = 1}【(xi - mu_l)^2 * pil】) / (Σ^n_{i = 1} pil)
                self.sigma2[l] = pow(X - self.mean[l], 2) @ responsibility_prob[:, l] / responsibility_prob[:, l].sum()

            # Determine whether the termination conditions are met.
            # calculate the norm of parameters
            para_norm = norm(
                concatenate([self.alpha, self.mean, self.sigma2]) - concatenate([alpha_old, mean_old, sigma_old]))
            # calculate log likelihood of parameters after updating
            log_likelihood = self.log_likelihood(X, alpha=self.alpha, mean=self.mean, sigma2=self.sigma2)
            if para_norm < self.threshold:
                logger.info("‖para - para_old‖ < threshold, optimize success.")
                return self.alpha, self.mean, self.sigma2
            if log_likelihood - log_likelihood_old < self.log_likelihood_threshold:
                logger.info("|log-likelihood - log-likelihood| < log_likelihood_threshold, optimize success.")
                return self.alpha, self.mean, self.sigma2

    def predict(self, X: (list, ndarray)) -> ndarray:
        # Determine which type of sample the incoming sample belongs to with p(z|x, θ).
        res = []
        for i in range(len(X)):
            # p(z = l|x, θ) = αl * N(x|μl, σ^2l) / Σ^k_{l = 1} αl*N(x|μl,σ^2l)
            # we can omit denominator because it is identical for a determinate sample
            category = argmax(self.alpha * self.normal_pdf(X[i], self.mean, self.sigma2))
            res.append(category)
        return asarray(res)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: (list, ndarray)):
        if alpha is None:
            self._alpha = alpha
        else:
            alpha = asarray(alpha)
            if not isinstance(alpha, (list, ndarray)):
                raise TypeError
            if alpha.sum() != 1 or not (0 <= alpha.all() <= 1):
                raise ValueError
            if len(alpha) != self.cluster:
                raise DimensionalError
            self._alpha = alpha

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean: (list, ndarray)):
        if mean is None:
            self._mean = mean
        else:
            if not isinstance(mean, (list, ndarray)):
                raise TypeError
            if len(mean) != self.cluster:
                raise DimensionalError
            self._mean = mean

    @property
    def sigma2(self):
        return self._sigma

    @sigma2.setter
    def sigma2(self, sigma2: (list, ndarray)):
        if sigma2 is None:
            self._sigma = sigma2
        else:
            sigma2 = asarray(sigma2)
            if not isinstance(sigma2, (list, ndarray)):
                raise TypeError
            if not (sigma2.all() > 0):
                raise ValueError
            if len(sigma2) != self.cluster:
                raise DimensionalError
            self._sigma = sigma2
