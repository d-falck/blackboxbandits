from abc import ABC, abstractmethod
from bayesmark.abstract_optimizer import AbstractOptimizer
import numpy as np
from typing import Type, List


class AbstractStatefulOptimizer(ABC):

    @abstractmethod
    def new_task(self):
        raise NotImplementedError()

    @abstractmethod
    def suggest(self, n_suggestions):
        raise NotImplementedError()

    @abstractmethod
    def observe(self, X, y):
        raise NotImplementedError()


class MakeStateful(AbstractStatefulOptimizer):

    def __init__(self, optimizer_class: Type[AbstractOptimizer], api_config, **kwargs):
        self.optimizer_class = optimizer_class
        self.api_config = api_config
        self.kwargs = kwargs
        self.new_task()

    def new_task(self):
        # Fresh instance of optimizer
        self.optimizer = self.optimizer_class(self.api_config, **self.kwargs)

    def suggest(self, n_suggestions):
        return self.optimizer.suggest(n_suggestions)

    def observe(self, X, y):
        self.optimizer.observe(X, y)


class FPMLBanditOptimizer(AbstractStatefulOptimizer):

    def __init__(self,
                 arms: List[AbstractStatefulOptimizer],
                 num_tasks: int,
                 perturbation_rate: float):
        self.arms = arms
        self.A = len(arms)
        self.n = num_tasks
        self.epsilon = perturbation_rate

        self.round = 0
        self.rewards = np.empty((0,self.A))

        self.new_task()

    def new_task(self): 
        