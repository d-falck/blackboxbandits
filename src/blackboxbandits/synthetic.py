"""Tools for running bandits on synthetic data
(unrelated to black-box optimization). Similar to the
`meta` submodule but for synthetic data."""
from abc import ABC, abstractmethod
from typing import List, Type
import itertools

import pandas as pd
import numpy as np

from .bandits import AbstractMultiBandit

class AbstractEnvironment(ABC):
    """Abstract interface for a potentially randomised synthetic
    reward environment.
    
    Parameters
    ----------
    n : int
        Number of rounds.
    """
    def __init__(self, n: int):
        self.n = n

    @abstractmethod
    def generate_rewards(self) -> pd.DataFrame:
        """Generate rewards for this environment.
        """

class Synth1Environment(AbstractEnvironment):
    """Randomised environment for the first synthetic environment
    in the experimental write-up (the anticorrelated one).
    """
    def generate_rewards(self) -> pd.DataFrame:
        df = pd.DataFrame(index=range(self.n), columns=range(1,9))
        for round in range(self.n):
            type = "A" if np.random.binomial(1,0.5) == 1 else "B"
            means = ([0.8,0.7,0.6,0.5,0,0,0,0]
                     if type == "A"
                     else [0,0,0,0,0.4,0.3,0.2,0.1])
            df.loc[round,:] = [np.random.binomial(1,mean) for mean in means]
        return df
class Synth2Environment(AbstractEnvironment):
    """Randomised environment for the second synthetic environment
    in the experimental write-up (the correlated one).
    """
    def generate_rewards(self) -> pd.DataFrame:
        df = pd.DataFrame(index=range(self.n), columns=range(1,9))
        for round in range(self.n):
            type = "A" if np.random.binomial(1,0.5) == 1 else "B"
            means = ([0.8,0.7,0.6,0.5,0,0,0,0]
                     if type == "A"
                     else [0,0,0,0,0.8,0.7,0.6,0.5])
            df.loc[round,:] = [np.random.binomial(1,mean) for mean in means]
        return df

class AbstractAlgorithm(ABC):
    """Represents an algorithm for to run in a synthetic enviroment.
    Equivalent to `AbstractMetaOptimizer` but for synthetic environments.
    """
    @abstractmethod
    def __init__(self):
        self._has_run = False

    @abstractmethod
    def run(self, rewards: pd.DataFrame) -> None:
        """Run this algorithm on a given environment instance.
        
        Parameters
        ----------
        rewards : pd.DataFrame
            Dataframe of rewards, with arms as columns and rounds as rows.
        """
        self._n, self._A = rewards.shape
        self._arms = rewards.columns
        self._has_run = True

    def get_results(self) -> List[float]:
        assert self._has_run, "Must run algorithm before getting results."
        return self._scores

    @abstractmethod
    def get_history(self) -> List[List[int]]:
        """Gets the arms chosen at each round.
        """
        assert self._has_run, "Must run before getting history."

class BestFixedTArms(AbstractAlgorithm):
    """An algorithm which runs the best possible fixed combination
    of T arms on the presented environment.
    
    Implements `AbstractAlgorithm`.
    
    Parameters
    ----------
    T : int
        The number of arms to include in the fixed set.
        
    Attributes
    ----------
    Same as parameters, plus:
    best_subset : List[int]
        A list of the indices of the arms in the optimal T-subset
        found.
    """
    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.best_subset = None

    def run(self, rewards: pd.DataFrame) -> None:
        """Implements corresponding method from `AbstractAlgorithm`.
        """
        super().run(rewards)

        subsets = list(itertools.combinations(self._arms.to_list(), self.T))

        scores = [rewards.loc[:,subset].max(axis=1).sum() for subset in subsets]
        best_idx = np.array(scores).argmax()
        self.best_subset = subsets[best_idx]

        self._scores = rewards.loc[:,self.best_subset].max(axis=1).to_list()
        self._arms_chosen = self.best_subset

    def get_history(self) -> List[List[int]]:
        super().get_history()
        return [self._arms_chosen for _ in range(self._n)]


class TopTBestArms(AbstractAlgorithm):
    """An algorithm which runs the top T individually best
    possible arms together on the presented environment.
    
    Implements `AbstractAlgorithm`.
    
    Parameters
    ----------
    T : int
        The number of arms to include in the fixed set.
        
    Attributes
    ----------
    Same as parameters, plus:
    top_T : List[int]
        A list of the indices of the arms in the T_subset used.
    """
    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.top_T = None

    def run(self, rewards: pd.DataFrame) -> None:
        """Implements corresponding method from `AbstractAlgorithm`.
        """
        super().run(rewards)

        leaderboard = rewards.sum(axis=0).argsort().iloc[::-1].to_list()
        leaders = leaderboard[:self.T]
        self.top_T = self._arms[leaders].to_list()

        self._scores = rewards.loc[:,self.top_T].max(axis=1).to_list()
        self._arms_chosen = self.top_T

    def get_history(self) -> List[List[int]]:
        super().get_history()
        return [self._arms.to_list() for _ in range(self._n)]


class BanditAlgorithm(AbstractAlgorithm):
    """An algorithm which runs a specified multi-bandit algorithm on
    the environment presented.
    
    Parameters
    ----------
    bandit_type : Type[bandits.AbstractMultiBandit]
        The bandit class to be used on this environment.
    T : int
        The number of arms the bandit is allowed to pull at each round.
    **bandit_kwargs : dict, optional
        Keyword parameters for the bandit class constructor.

    Attributes
    ----------
    Same as parameters.
    """
    def __init__(self, bandit_type: Type[AbstractMultiBandit],
                 T: int, **bandit_kwargs):
        super().__init__()
        self.bandit_type = bandit_type
        self.T = T
        self.bandit_kwargs = bandit_kwargs

    def run(self, rewards: pd.DataFrame):
        super().run(rewards)

        bandit = self.bandit_type(A=self._A, T=self.T,
                                  n=self._n, **self.bandit_kwargs)

        self._scores = []
        self._arms_chosen = []
        for round in range(self._n):
            arm_indices = bandit.select_arms()
            arms = self._arms[arm_indices].to_list()
            feedback = rewards.loc[round, arms].to_list()
            bandit.observe_rewards(arm_indices, feedback)
            self._scores.append(max(feedback))
            self._arms_chosen.append(arms)

    def get_history(self) -> List[List[int]]:
        super().get_history()
        return self._arms_chosen