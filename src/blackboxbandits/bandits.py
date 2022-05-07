"""Provides implementations of various bandit algorithms in a generic context.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class AbstractMultiBandit(ABC):
    """Abstract base class for multi-bandits.
    
    Any instance will repeatedly choose a fixed number of arms to pull
    and then observe the rewards for each of those arms, receiving the maximum
    of those rewards.
    
    Parameters
    ----------
    A : int
        Number of arms available to choose from. Arms are subsequently referred
        to by their zero-indexed number.
    T : int
        Arm budget at each round, i.e. how many arms the bandit is allowed to
        pull.
    n : int
        Number of rounds expected overall. Used by some children.

    Attributes
    ----------
    Same as parameters, plus:
    self.round : int
        The current round, between `0` and `n`.
    self.expecting_observation : bool
        Whether or not the bandit is currently waiting to receive the rewards
        for the arms it just chose.
    """

    def __init__(self, A: int, T: int, n: int):
        self.A = A
        self.T = T
        self.n = n
        self.round = 0
        self.expecting_observation = False

    @abstractmethod
    def select_arms(self) -> List[int]:
        """Get arms selected by the bandit at the current round.

        Must only be called when the bandit isn't currently waiting for rewards.

        Returns
        -------
        List[int]
            A list of length `T` of the arms chosen by the bandit to pull at
            this round. Identified by their zero-indexed number.
        """
        assert self.round < self.n, \
            "Didn't expect this many rounds."
        assert not self.expecting_observation, \
            "Must observe rewards before choosing more arms"
        
        self.expecting_observation = True

    @abstractmethod
    def observe_rewards(self, arms: List[int], rewards: List[float]) -> None:
        """Provide the bandit with the rewards for the arms just selected.

        Must only be called after calling `select_arms` to get the chosen arms.

        Parameters
        ----------
        arms : List[int]
            A list of the arms whose rewards are being provided. Should match
            the output of the last call to `select_arms()`.
        rewards : List[float]
            A list of the rewards at this round for the arms in `arms`, in the
            corresponding order. Rewards must be in the interval [0,1].
        """
        assert self.expecting_observation, \
            "Must choose arms before observing rewards"
        assert len(arms) == len(rewards), "Must have a reward for each arm"
        assert all(0 <= arm < self.A for arm in arms), "Invalid arms provided"
        assert all(0 <= reward <= 1 for reward in rewards), \
            "Rewards not in range [0, 1]"

        self.expecting_observation = False
        self.round += 1


class AbstractFPML(AbstractMultiBandit):
    """Abstract base class for the full-feedback parts of
    Follow the Perturbed Multiple Leaders.

    Extends `AbstractMultiBandit`.
    
    Parameters
    ----------
    Same as `AbstractMultiBandit`, plus:
    epsilon : float
        The rate of the i.i.d. exponential perturbations applied to cumulative
        rewards at each round.

    Attributes
    ----------
    Same as parameters.
    """

    def __init__(self, A: int, T: int, n: int, epsilon: float):
        super().__init__(A, T, n) # The value of T here is meaningless
        self.epsilon = epsilon
        self._cum_est_rewards = np.zeros(A)

    def select_arms(self, _num: int, _store=True) -> List[int]:
        """Implements corresponding method in `AbstractMultiBandit`; but returns
        the top `_num` perturbed leaders, not `T`.
        """
        if _store:
            super().select_arms()
        perturbations = np.random.exponential(scale=1/self.epsilon, size=self.A)
        perturbed_cum_est_rewards = self._cum_est_rewards + perturbations
        leaderboard = np.argsort(perturbed_cum_est_rewards)
        leaders = leaderboard[-_num:] if _num > 0 else np.array([])
        return leaders.tolist()

    def _full_feedback_observation(self, rewards: List[float]) -> None:
        assert len(rewards) == self.A, "Must provide rewards for all arms"
        self._cum_est_rewards += np.array(rewards)


class FPMLFixed(AbstractFPML):
    """Implementation of the Follow the Perturbed Multiple Leaders algorithm
    with partial feedback and a fixed number of uniform exploration arms chosen
    at each round.

    Implements `AbstractMultiBandit`.

    Parameters
    ----------
    Same as `AbstractMultiBandit` plus:
    S : int, optional
        How many of the `T` arms at each round to set aside for exploration.
        Defaults to 1.

    Attributes
    ----------
    Same as parameters, plus:
    epsilon : float
        The rate of the i.i.d. exponential perturbations applied to cumulative
        rewards at each round.
    """

    def __init__(self, A: int, T: int, n: int, S: int = 1):
        self.S = S
        super().__init__(A, T, n, epsilon=S/A*(np.log(A)/n)**(1/(T-S+1)))
        # Value of epsilon from Thm 3.9

    def select_arms(self) -> List[int]:
        """Implements corresponding method in `AbstractMultiBandit`.
        """
        leaders = super().select_arms(_num=self.T-self.S)
        self._explore_arms = np.random.choice(np.arange(self.A),
                                              size=self.S,
                                              replace=False).tolist()
        return list(set(leaders+self._explore_arms))

    def observe_rewards(self, arms: List[int], rewards: List[float]) -> None:
        """Implements corresponding method in `AbstractMultiBandit`.
        """
        super().observe_rewards(arms, rewards)
        estimates = np.zeros(self.A)
        for arm, reward in zip(arms, rewards):
            if arm in self._explore_arms:
                estimates[arm] = reward*self.A/self.S
        super()._full_feedback_observation(estimates.tolist())


class FPMLProb(AbstractFPML):
    """Implementation of the Follow the Perturbed Multiple Leaders algorithm
    with partial feedback and probabilistic exploration.

    Implements `AbstractMultiBandit`.

    Parameters
    ----------
    Same as `AbstractMultiBandit`, plus:
    gamma : float
        The probability with which to replace individual chosen arms with
        a uniformly sampled exploration arm.

    Attributes
    ----------
    Same as parameters, plus:
    epsilon : float
        The rate of the i.i.d. exponential perturbations applied to cumulative
        rewards at each round.
    """

    def __init__(self, A: int, T: int, n: int, gamma: float, epsilon: Optional[float] = None):
        self.gamma = gamma
        eps = gamma*T/A*(np.log(A)/n)**(1/(T-gamma*T+1)) if epsilon is None else epsilon
        super().__init__(A, T, n, epsilon=eps)
        # Value of epsilon adapted from Thm 3.9

    def select_arms(self) -> List[int]:
        """Implements corresponding method in `AbstractMultiBandit`.
        """
        S = np.random.binomial(n=self.T, p=self.gamma)
        leaders = super().select_arms(_num=self.T-S)
        self._explore_arms = np.random.choice(np.arange(self.A),
                                              size=S,
                                              replace=False).tolist()
        return list(set(leaders+self._explore_arms))

    def observe_rewards(self, arms: List[int], rewards: List[float]) -> None:
        """Implements corresponding method in `AbstractMultiBandit`.
        """
        super().observe_rewards(arms, rewards)
        estimates = np.zeros(self.A)
        for arm, reward in zip(arms, rewards):
            if arm in self._explore_arms:
                estimates[arm] = reward*self.A/(self.gamma*self.T)
        super()._full_feedback_observation(estimates.tolist())


class FPMLWithGR(AbstractFPML):
    """Implementation of the Follow the Perturbed Multiple Leaders algorithm
    with partial feedback and geometric resampling for reward estimation.

    Implements `AbstractMultiBandit`.

    Parameters
    ----------
    Same as `AbstractMultiBandit`, plus:
    gamma : float, optional
        Float between 0 and 1 indicating the tendency to explore uniformly;
        defaults to 0, in which case no explicit exploration is performed.

    Attributes
    ----------
    Same as parameters, plus:
    epsilon : float
        The rate of the i.i.d. exponential perturbations applied to cumulative
        rewards at each round.
    """

    def __init__(self, A: int, T: int, n: int,
                 gamma: float=0, epsilon: Optional[float] = None):
        self.gamma = gamma
        self._M = int(np.ceil(np.sqrt(A*(n/np.log(A))**(T/(T+1)))))
        eps = np.sqrt(1/A*(n/np.log(A))**((T-2)/(T+1))) \
              if epsilon is None else epsilon
        super().__init__(A, T, n, epsilon=eps)
        # Values from Prop 3.11
        # TODO: Update these for gamma>0 case

    def select_arms(self, _store=True) -> List[int]:
        """Implements corresponding method in `AbstractMultiBandit`.
        """
        S = np.random.binomial(n=self.T, p=self.gamma)
        leaders = super().select_arms(_num=self.T-S, _store=_store)
        explore = np.random.choice(np.arange(self.A), size=S, replace=False).tolist()
        selected = list(set(leaders+explore))
        if _store:
            self._selected = selected
        return selected

    def observe_rewards(self, arms: List[int], rewards: List[float]) -> None:
        """Implements corresponding method in `AbstractMultiBandit`.
        """
        super().observe_rewards(arms, rewards)
        assert arms == self._selected, \
            "Rewards must be provided for the chosen arms"
        estimates = np.ones(self.A)
        estimates[arms] -= (1-np.array(rewards)) * self._geometric_resample(arms)
        super()._full_feedback_observation(estimates.tolist())

    def _geometric_resample(self, arms: List[int]) -> np.array:
        geom = np.full(len(arms), self._M)
        for k in range(1, self._M):
            resampled_arms = self.select_arms(_store=False)
            for i, arm in enumerate(arms):
                if arm in resampled_arms:
                    geom[i] = min(geom[i], k)
        return geom


class StreeterFPML(AbstractMultiBandit):
    """Implementation of the modified Streeter algorithm with multi-bandit
    FPML as a sub-routine.

    Implements `AbstractMultiBandit`.

    Parameters
    ----------
    Same as `AbstractMultiBandit` plus:
    T_1 : int
        The arm-budget for each internal instance of FPML.
    T_2 : int
        The number of internal instances of FPML.
    gamma : float
        The propensity to explore at each round, float between 0 and 1.
    gr : bool, optional
        Whether to use geometric resampling in the internal FPML instances.
        Defaults to False.

    Attributes
    ----------
    Same as parameters.

    Notes
    -----
    The parameters `T_1` and `T_2` must multiply to `T`.
    """

    def __init__(self, A: int, T: int, T_1: int, T_2: int, n: int,
                 gamma: float, gr: bool = False, epsilon: Optional[float] = None):
        assert T_1 * T_2 == T, "Time parameters must multiply to total budget"
        self.T_1 = T_1
        self.T_2 = T_2
        self.gamma = gamma
        self.gr = gr
        fpml_class = FPMLWithGR if gr else FPMLProb
        self._internal_fpml_instances = [fpml_class(A, T_1, n, gamma, epsilon)
                                         for _ in range(T_2)]
        super().__init__(A, T, n)

    def select_arms(self) -> List[int]:
        """Implements corresponding method in `AbstractMultiBandit`.
        """
        super().select_arms()
        self._grouped_arms = [self._internal_fpml_instances[t].select_arms()
                             for t in range(self.T_2)]
        return [arm for arms in self._grouped_arms for arm in arms]

    def observe_rewards(self, arms: List[int], rewards: List[float]) -> None:
        """Implements corresponding method in `AbstractMultiBandit`.
        """
        super().observe_rewards(arms, rewards)
        assert arms == [arm for arms in self._grouped_arms for arm in arms], \
            "Observed rewards must be for the chosen arms"

        # Re-group rewards by internal FPML instance
        i = 0; grouped_rewards = []
        for arms in self._grouped_arms:
            grouped_rewards.append(rewards[i:i+len(arms)])
            i += len(arms)
        
        # Create modified 'greedy' rewards
        max_rewards = [max(rewards) for rewards in grouped_rewards]
        grouped_modified_rewards = []
        for t, these_rewards in enumerate(grouped_rewards):
            cum_max = 0 if t == 0 else max(max_rewards[:t])
            modified_rewards = [max(reward, cum_max) - cum_max
                                for reward in these_rewards]
            grouped_modified_rewards.append(modified_rewards)

        # Feed back to internal FPML instances
        for t in range(self.T_2):
            self._internal_fpml_instances[t].observe_rewards(
                self._grouped_arms[t], grouped_modified_rewards[t]
            )
        
        self._grouped_arms = None


class Exp3(AbstractMultiBandit):
    """Implementation of the standard Exp3 algorithm.

    Implements `AbstractMultiBandit`, but cannot recommend
    more than one arm per round.

    Parameters
    ----------
    A : int
        Number of arms available to choose from. Arms are subsequently referred
        to by their zero-indexed number.
    n : int
        Number of arms expected overall. Not actually used.
    gamma : Optional[float], optional
        Exploration parameter in [0,1] determining the probability of selecting
        an arm uniformly at each round. Chooses a good value if not provided.

    Attributes
    ----------
    Same as parameters.
    """

    def __init__(self, A: int, n: int, gamma: Optional[float] = None):
        self.gamma = gamma if gamma is not None \
                           else np.sqrt(A*np.log(A))/(2/3*n*(np.e-1)) # Check this
        self._weights = np.ones(A)
        super().__init__(A=A, T=1, n=n)

    def select_arms(self) -> List[int]:
        """Implements the corresponding method from `AbstractMultiBandit`.

        Will always return a list of length 1.
        """
        super().select_arms()
        prob_dist = (1-self.gamma) * self._weights / self._weights.sum() \
                + self.gamma / self.A
        self._arm = np.random.choice(np.arange(self.A), p=prob_dist)
        self._prob_chosen = prob_dist[self._arm]
        return [self._arm]

    def observe_rewards(self, arms: List[int], rewards: List[int]) -> None:
        """Implements the corresponding method from `AbstractMultiBandit`.

        The provided lists are required to be of length 1.
        """
        super().observe_rewards(arms, rewards)
        assert len(arms) == 1 and len(rewards) == 1, \
            "Only one arm and reward should be provided."
        assert arms[0] == self._arm, "Arm doesn't match the selected arm."
        reward = rewards[0]
        est_reward = reward / self._prob_chosen
        self._weights[self._arm] *= np.exp(self.gamma * est_reward / self.A)


class Streeter(AbstractMultiBandit):
    """Implementation of the standard Streeter online greedy algorithm
    (for unit-time actions) under partial feedback.
    
    Implements `AbstractMultiBandit`.

    Parameters
    ----------
    Same as parent class plus:
    gamma : Optional[float], optional
        Exploration probability for underlying Exp3 instances. Chooses a good
        value if not used.

    Attributes
    ----------
    Same as parameters.
    """
    
    def __init__(self, A: int, T: int, n: int, gamma: Optional[float]=None):
        self.gamma = gamma
        self._internal_exp3_instances = [Exp3(A=A, n=n, gamma=gamma)
                                         for _ in range(T)]
        super().__init__(A, T, n)

    def select_arms(self) -> List[int]:
        """Implements the corresponding method from `AbstractMultiBandit`.
        """
        super().select_arms()
        self._arms = [exp3.select_arms()[0]
                      for exp3 in self._internal_exp3_instances]
        return self._arms

    def observe_rewards(self, arms: List[int], rewards: List[int]) -> None:
        """Implements the corresponding method from `AbstractMultiBandit`.
        """
        super().observe_rewards(arms, rewards)
        for t, exp3 in enumerate(self._internal_exp3_instances):
            r = rewards[0] if t == 0 else max(rewards[:(t+1)]) - max(rewards[:t])
            exp3.observe_rewards([arms[t]], [r])