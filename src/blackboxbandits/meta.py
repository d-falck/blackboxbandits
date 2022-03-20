"""Provides implementations of various meta-optimizers, i.e. algorithms to
choose combinations of single black-box optimizers to run on each task in a
sequence of optimization tasks.
"""

from abc import ABC, abstractmethod
import pandas as pd
from .bandits import AbstractMultiBandit
from typing import Type, Optional, List
import itertools


class AbstractMetaOptimizer(ABC):
    """Abstract base class for a meta-optimizer.
    """

    def __init__(self):
        self._has_run = False

    @abstractmethod
    def run(self, data: pd.DataFrame,
            function_order: Optional[List[str]] = None) -> None:
        """Run the meta-optimizer on pre-calculated performance data for a set
        of underlying optimization algorithms.

        Parameters
        ----------
        data : pd.DataFrame
            A dataframe containing the normalized scores for each of a set of
            black-box optimizers on a sequence of optimization tasks.
        function_order : Optional[List[str]], optional
            A list of the function names appearing in `data` specifying the
            order in which the tasks should be presented to the meta-optimizer.
            Defaults to None, in which case the order in the dataframe is used.
        """
        self._optimizers = data.index.get_level_values("optimizer") \
                               .drop_duplicates().to_series()
        self._functions = data.index.get_level_values("function") \
                              .drop_duplicates().to_list()

        if function_order is not None:
            assert set(function_order) == set(self._functions), \
                "Functions in order list doesn't match the functions provided."
            self._functions = function_order
        
        self._A = self._optimizers.size
        self._n = len(self._functions)

        self._has_run = True

    def get_results(self) -> pd.DataFrame:
        """Get the results from the meta-opzimizer as a dataframe.
        
        Returns
        -------
        pd.DataFrame
            A dataframe of the normalized score of the meta-optimizer on
            each task in the sequence of tasks.
        """
        assert self._has_run, "Must run before getting results."
        results = pd.DataFrame({
            "visible_score": self._scores_visible,
            "generalization_score": self._scores_generalization
        }, index=self._functions)
        return results


class BestFixedTAlgos(AbstractMetaOptimizer):
    """A meta-optimizer which runs the best possible fixed combination of
    T algorithms on the presented tasks.

    Implements `AbstractMetaOptimizer`.

    Parameters
    ----------
    T : int
        The number of optimizers to include in the fixed set.

    Attributes
    ----------
    Same as parameters, plus:
    best_subset : List[str]
        A list of the names of the underlying optimizers in the optimal
        T-subset found.
    """

    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.best_subset = None

    def run(self, data: pd.DataFrame,
            function_order: Optional[List[str]] = None) -> None:
        """Implements corresponding method from `AbstractMetaOptimizer`.
        """
        super().run(data, function_order)

        relevant_subsets = list(itertools.combinations(
            self._optimizers.to_list(),
            self.T
        ))

        best_scores_visible = [-1]*self._n
        for subset in relevant_subsets:
            scores_visible = []
            scores_generalization = []
            for func in self._functions:
                relevant_data = data.loc[pd.IndexSlice[subset, func],:]
                visible_rewards = relevant_data["visible"]["score"].to_list()
                generalization_rewards = relevant_data \
                                         ["generalization"]["score"].to_list()
                scores_visible.append(max(visible_rewards))
                scores_generalization.append(max(generalization_rewards))

            if sum(scores_visible) > sum(best_scores_visible):
                best_scores_visible = scores_visible
                best_scores_generalization = scores_generalization
                best_subset = subset

        self.best_subset = best_subset
        self._scores_visible = best_scores_visible
        self._scores_generalization = best_scores_generalization
        self._arms = ",".join([str(self._optimizers.to_list().index(opt)) \
                                  for opt in best_subset])
        
    def get_history(self) -> pd.DataFrame:
        assert self._has_run, "Must run before getting history."
        history = [self._arms for _ in self._functions]
        history = pd.DataFrame({
            "arms": history
        }, index=self._functions)
        return history


class TopTBestAlgos(AbstractMetaOptimizer):
    """A meta-optimizer which runs the top T individually best
    best possible algorithms together on the presented tasks.

    Implements `AbstractMetaOptimizer`.

    Parameters
    ----------
    T : int
        The number of optimizers to include in the fixed set.

    Attributes
    ----------
    Same as parameters, plus:
    top_T : List[str]
        A list of the names of the underlying optimizers in the T-subset used.
    """

    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.top_T = None

    def run(self, data: pd.DataFrame,
            function_order: Optional[List[str]] = None) -> None:
        """Implements corresponding method from `AbstractMetaOptimizer`.
        """
        super().run(data, function_order)

        # First run each optimizer individually
        scores = pd.DataFrame(index=self._functions, columns=self._optimizers)
        for func in self._functions:
            rewards = data.loc[pd.IndexSlice[self._optimizers,func],:]
            scores.loc[func, :] = rewards["visible"]["score"].to_list()

        leaderboard = scores.mean().sort_values(ascending=False)
        self.top_T = leaderboard.index.to_list()[:self.T]

        # Then run the top T of them together
        scores_visible = []
        scores_generalization = []
        for func in self._functions:
            rewards = data.loc[pd.IndexSlice[self.top_T,func],:]
            visible = rewards["visible"]["score"].to_list()
            generalization = rewards["generalization"]["score"].to_list()
            scores_visible.append(max(visible))
            scores_generalization.append(max(generalization))

        self._scores_visible = scores_visible
        self._scores_generalization = scores_generalization
        self._arms = ",".join([str(self._optimizers.to_list().index(opt)) \
                                  for opt in self.top_T])
        
    def get_history(self) -> pd.DataFrame:
        assert self._has_run, "Must run before getting history."
        history = [self._arms for _ in self._functions]
        history = pd.DataFrame({
            "arms": history
        }, index=self._functions)
        return history


class BanditMetaOptimizer(AbstractMetaOptimizer):
    """A meta-optimizer which runs a specified multi-bandit algorithm on the
    tasks presented.

    Parameters
    ----------
    bandit_type : Type[bandits.AbstractMultiBandit]
        The bandit class to be used by this optimizer on the tasks.
    T : int
        The number of optimizers the bandit is allowed to run on each task.
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

    def run(self, data: pd.DataFrame,
            function_order: Optional[List[str]] = None) -> None:
        """Implements corresponding method in `AbstractMultiBandit`.
        """
        super().run(data, function_order)

        bandit = self.bandit_type(A=self._A, T=self.T,
                                  n=self._n, **self.bandit_kwargs)

        self._scores_visible = []
        self._scores_generalization = []
        self._arms_chosen = []
        for func in self._functions:
            arm_indices = bandit.select_arms()
            optimizers_chosen = self._optimizers[arm_indices].to_list()
            rewards = data.loc[pd.IndexSlice[optimizers_chosen,func],:]
            visible_rewards = rewards["visible"]["score"] \
                              [optimizers_chosen].to_list()
            generalization_rewards = rewards["generalization"]["score"] \
                                     [optimizers_chosen].to_list()
            bandit.observe_rewards(arm_indices, visible_rewards)

            self._scores_visible.append(max(visible_rewards))
            self._scores_generalization.append(max(generalization_rewards))
            self._arms_chosen.append(",".join(map(str,arm_indices)))
        
    def get_history(self) -> pd.DataFrame:
        assert self._has_run, "Must run before getting history."
        history = pd.DataFrame({
            "arms": self._arms_chosen
        }, index=self._functions)
        return history