# @author vincen
# @date 3:30 pm Thursday, 18 March 2021 time in Shanxi xi'an
from numpy.random import uniform
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s -  %(asctime)s - %(filename)s -  %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)


class rejection_sampling():
    """
    class rejection_sampling():
    =========================
        The rejection-sampling is a classical sampling method with allows one to sample
        from a distribution which is difficult or impossible to simulate by an inverse transformation.
        Instead, draws are token from an instrumental density (also called as proposal density) and
        accepted with a carefully chosen probability.The resulting draw is a draw from the target density.
    ---------------------
    Parameters:
        1._proposal_rv
            Random variables that are easy to sample.You'd better pass a common r.v like scipy.normal,scipy.chi as so on.
            u need to pass a object meet the following two conditions:
                Ⅰ. can get the density function use calling pdf
                Ⅱ. can sampling a random number use calling rvs method.
            i suggest u pass a object associated with scipy.stats.do as it you only need to select the type of proposal distribution
            instead of taking care of implements of above two methods.
            All distributions will have location (L) and Scale (S) parameters along with any shape parameters needed, the names for the shape parameters will vary.
            Standard form for the distributions will be given where loc = 0 and scale = 1.
            web address of distribution in scipy is https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html
        2._target_density
            The density function of distribution which you want to sampling from.
        3._constant
            Constant used to make _target_density/(c*_proposal_rv.pdf) less than or equal to 1.
            u must correctly set this parameter.
        i'm already set mods of above three parameters as private but use getter/setter decorate the get/set method of them.
        use instance.parameter_name get/set the value of corresponding parameter.
    ------------------
    Example:
        target_density = lambda x: 6 * x * (1 - x) if 0 <= x <= 1 else 0
        constant = 3 / 2
        from scipy.stats import uniform as scipy_uniform

        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        plt.figure(dpi=400)
        sample_num = (100, 1000, 5000, 20000)
        sns.set(style="whitegrid")
        for i in range(1, 5, 1):
            ax = plt.subplot(2, 2, i)
            x_arr = np.linspace(0, 1.1, 100)

            rvs = rejection_sampling(target_density=target_density,
                                     proposal_rv=scipy_uniform,
                                     constant=constant
                                     ).generate(size=sample_num[i - 1])

            # transform rvs from generator to list
            rvs = list(rvs)

            y = [target_density(x) for x in x_arr]
            plt.plot(x_arr, y, c="darkblue")
            plt.hist(rvs, alpha=.7, color='steelblue', density=True, bins=80)
            plt.title("sample number = {}".format(sample_num[i - 1]))
            plt.xlabel("x")
            sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
    """
    _target_density = None
    _proposal_rv = None
    _constant = None

    def __init__(self, target_density, proposal_rv, constant):
        self.target_density = target_density
        self.proposal_rv = proposal_rv
        self.constant = constant

    @property
    def target_density(self):
        return self._target_density

    @target_density.setter
    def target_density(self, density):
        self._target_density = density

    @property
    def proposal_rv(self):
        return self._proposal_rv

    @proposal_rv.setter
    def proposal_rv(self, proposal_rv):
        pdf_attr = getattr(proposal_rv, "pdf", None)
        if callable(pdf_attr):
            rvs_attr = getattr(proposal_rv, "rvs", None)
        else:
            raise NotImplementedError
        if callable(rvs_attr):
            self._proposal_rv = proposal_rv
        else:
            raise NotImplementedError

    @property
    def constant(self):
        return self._constant

    @constant.setter
    def constant(self, constant):
        if not isinstance(constant, (float, int)):
            raise TypeError
        elif constant <= 0:
            raise ValueError
        self._constant = constant

    def generate(self, size=1):
        self.accept_num = 0

        proposal_pdf = self.proposal_rv.pdf

        proposal_rvs = self.proposal_rv.rvs(size=size)
        uniform_rvs = uniform(size=size)
        for rvs in zip(uniform_rvs, proposal_rvs):
            if rvs[0] < self.target_density(rvs[1]) / (self.constant * proposal_pdf(rvs[1])):
                yield rvs[1]
                self.accept_num += 1

        logger.info("The generation is complete, acceptance rate of it is {}...".format(self.accept_num / size))


if __name__ == "__main__":
    target_density = lambda x: 6 * x * (1 - x) if 0 <= x <= 1 else 0
    constant = 3 / 2
    from scipy.stats import uniform as scipy_uniform

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    plt.figure(dpi=400)
    sample_num = (100, 1000, 5000, 20000)
    sns.set(style="whitegrid")
    for i in range(1, 5, 1):
        ax = plt.subplot(2, 2, i)
        x_arr = np.linspace(0, 1.1, 100)

        rvs = rejection_sampling(target_density=target_density,
                                 proposal_rv=scipy_uniform,
                                 constant=constant
                                 ).generate(size=sample_num[i - 1])

        # transform rvs from generator to list
        rvs = list(rvs)

        y = [target_density(x) for x in x_arr]
        plt.plot(x_arr, y, c="darkblue")
        plt.hist(rvs, alpha=.7, color='steelblue', density=True, bins=80)
        plt.title("sample number = {}".format(sample_num[i - 1]))
        plt.xlabel("x")
        sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()
