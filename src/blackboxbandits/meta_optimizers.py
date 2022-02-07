from abc import ABC, abstractmethod
import pandas as pd
from .bandits import AbstractMultiBandit
from typing import Type, Optional, List
import itertools


class AbstractMetaOptimizer(ABC):

    def __init__(self):
        self.run = False

    @abstractmethod
    def run(self, data: pd.DataFrame, function_order: Optional[List[str]]) -> None:
        self.optimizers = data.index.get_level_values("optimizer") \
                               .drop_duplicates().to_series()
        self.functions = data.index.get_level_values("function") \
                              .drop_duplicates().to_list()

        if function_order is not None:
            assert set(function_order) == set(self.functions), \
                "Functions in order list doesn't match the functions provided."
            self.functions = function_order
        
        self.A = self.optimizers.size
        self.n = len(self.functions)

        self.run = True

    def get_results(self) -> pd.DataFrame:
        assert self.run, "Must run before getting results."
        results = pd.DataFrame({
            "visible_score": self.scores_visible,
            "generalization_score": self.scores_generalization
        }, index=self.functions)
        return results


class BestFixedTAlgos(AbstractMetaOptimizer):

    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def run(self, data: pd.DataFrame, function_order: Optional[List[str]]) -> None:
        super().run(data, function_order)

        relevant_subsets = list(itertools.combinations(self.optimizers.to_list(), self.T))

        best_scores_visible = [-1]*self.n
        for subset in relevant_subsets:
            scores_visible = []
            scores_generalization = []
            for func in self.functions:
                relevant_data = data.loc[pd.IndexSlice[subset, func],:]
                visible_rewards = relevant_data["visible"]["score"].to_list()
                generalization_rewards = relevant_data["generalization"]["score"].to_list()
                scores_visible.append(max(visible_rewards))
                scores_generalization.append(max(generalization_rewards))

            if sum(scores_visible) > sum(best_scores_visible):
                best_scores_visible = scores_visible
                best_scores_generalization = scores_generalization
                best_subset = subset

        self.best_subset = best_subset
        self.scores_visible = best_scores_visible
        self.scores_generalization = best_scores_generalization


class BanditMetaOptimizer(AbstractMetaOptimizer):

    def __init__(self, bandit_type: Type[AbstractMultiBandit], T: int, **bandit_kwargs):
        super().__init__()
        self.bandit_type = bandit_type
        self.T = T
        self.bandit_kwargs = bandit_kwargs

    def run(self, data: pd.DataFrame, function_order: Optional[List[str]]) -> None:
        super().run(data, function_order)

        bandit = self.bandit_type(self.A, self.T, self.n, **self.bandit_kwargs)

        self.scores_visible = []
        self.scores_generalization = []
        for func in self.functions:
            arm_indices = bandit.select_arms()
            optimizers_chosen = self.optimizers[arm_indices].to_list()
            rewards = data.loc[pd.IndexSlice[optimizers_chosen,func],:]
            visible_rewards = rewards["visible"]["score"].to_list()
            generalization_rewards = rewards["generalization"]["score"].to_list()
            bandit.observe_rewards(arm_indices, visible_rewards)

            self.scores_visible.append(max(visible_rewards))
            self.scores_generalization.append(max(generalization_rewards))