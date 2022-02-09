"""Provides implementations of various bandit algorithms in a generic context.
"""

from abc import ABC, abstractmethod
from typing import List
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


class FPML(AbstractMultiBandit):
    """Implementation of the Follow the Perturbed Multiple Leaders algorithm
    with partial feedback.

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
        self.epsilon = S/A*(np.log(A)/n)**(1/(T-S+1)) # Value from Thm 3.9
        super().__init__(A, T, n)
        self._cum_est_rewards = np.zeros(A)

    def select_arms(self) -> List[int]:
        """Implements corresponding method in `AbstractMultiBandit`.
        """
        super().select_arms()
        perturbations = np.random.exponential(scale=1/self.epsilon, size=self.A)
        perturbed_cum_est_rewards = self._cum_est_rewards + perturbations
        leaderboard = np.argsort(perturbed_cum_est_rewards)
        leaders = leaderboard[-(self.T-self.S):]
        explore_arms = np.random.choice(leaderboard, size=self.S, replace=False)
        chosen_arms = np.unique(np.concatenate([leaders, explore_arms]))
        self._last_explored_arms = explore_arms
        return chosen_arms

    def observe_rewards(self, arms: List[int], rewards: List[float]) -> None:
        """Implements corresponding method in `AbstractMultiBandit`.
        """
        super().observe_rewards(arms, rewards)
        for arm, reward in zip(arms, rewards):
            estimate = reward*self.A/self.S \
                       if arm in self._last_explored_arms \
                       else 0
            self._cum_est_rewards[arm] += estimate
        self._last_explored_arms = None


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
    S : int, optional
        How many of the `T` arms at each round to set aside for exploration.
        Defaults to 1.

    Attributes
    ----------
    Same as parameters.

    Notes
    -----
    The parameters `T_1` and `T_2` must multiply to `T`.
    """

    def __init__(self, A: int, T: int, T_1: int, T_2: int, n: int, S: int = 1):
        assert T_1 * T_2 == T, "Time parameters must multiply to total budget"
        self.T_1 = T_1
        self.T_2 = T_2
        self.S = S
        self._internal_fpml_instances = [FPML(A=A, T=T_1, S=S, n=n)
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


# class Streeter(AbstractMultiBandit):
#     pass