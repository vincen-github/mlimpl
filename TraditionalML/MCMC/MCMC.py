from numpy.random import uniform

from proposal_distribution import Proposal_Distribution
from target_distribution import Target_Distribution


class Metropolis_Hastings(object):

    def __init__(self,
                 burning_period: int,
                 end_period: int,
                 x0,
                 target_distribution,
                 proposal_distribution
                 ) -> None:
        self._target_distribution = target_distribution
        self._proposal_distribution = proposal_distribution
        self.x0 = x0
        self._burning_period = burning_period
        self._end_period = end_period

    def run(self):
        samples = []
        x = self.x0
        for i in range(self._end_period):
            x_star = self.proposal_distribution.sampling()
            u = uniform(0, 1)
            accept_prob = min(1, (self.target_distribution.pdf(x_star) * self.proposal_distribution.pdf(x_star, x))
                              / (self.target_distribution.pdf(x) * self.proposal_distribution.pdf(x, x_star)))
            if u < accept_prob:
                x = x_star
            if i > self.burning_period:
                samples.append(x)
        return samples

    @property
    def burning_period(self):
        return self._burning_period

    @burning_period.setter
    def burning_period(self, burning_period: int):
        if not isinstance(int):
            raise TypeError
        if burning_period <= 0:
            raise ValueError
        self._burning_period = burning_period

    @property
    def end_period(self):
        return self._end_period

    @end_period.setter
    def end_period(self, end_period: int) -> None:
        if not isinstance(end_period, int):
            raise TypeError
        if end_period <= 0:
            raise ValueError
        self._end_period = end_period

    @property
    def target_distribution(self):
        return self._target_distribution

    @target_distribution.setter
    def target_distribution(self, target_distribution):
        if not issubclass(target_distribution, Target_Distribution):
            raise TypeError
        self._target_distribution = target_distribution

    @property
    def proposal_distribution(self):
        return self._proposal_distribution

    @proposal_distribution.setter
    def proposal_distribution(self, proposal_distribution):
        if not issubclass(proposal_distribution, Proposal_Distribution):
            raise TypeError
        self._proposal_distribution = proposal_distribution
