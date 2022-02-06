from abc import ABC, abstractmethod
from typing import List
import numpy as np


class MultiBandit(ABC):

    def __init__(self, A: int, T: int, n: int):
        self.A = A
        self.T = T
        self.n = n
        self.round = 0

    @abstractmethod
    def select_arms(self) -> List[int]:
        assert self.round < self.n, "Didn't expect this many rounds."
        assert not self.expecting_observation, "Must observe rewards before choosing more arms"
        
        self.expecting_observation = True

    @abstractmethod
    def observe_rewards(self, arms: List[int], rewards: List[float]) -> None:
        assert self.expecting_observation, "Must choose arms before observing rewards"
        assert all(0 <= arm < self.A for arm in arms), "Invalid arms provided"
        assert all(0 <= reward <= 1 for reward in rewards), "Rewards not in range [0, 1]"

        self.expecting_observation = False
        self.round += 1


class FPML(MultiBandit):

    def __init__(self, A: int, T: int, n: int, S: int):
        self.S = S
        self.epsilon = S/A*(np.log(A)/n)**(1/(T-S+1)) # Value from Thm 3.9
        super().__init__(self, A, T, n)
        self.cum_est_rewards = np.zeros(A)

    def select_arms(self) -> List[int]:
        super().select_arms()
        perturbations = np.random.exponential(scale=1/self.epsilon, size=self.A)
        perturbed_cum_est_rewards = self.cum_est_rewards + perturbations
        leaderboard = np.argsort(perturbed_cum_est_rewards)
        leaders = leaderboard[-(self.T-self.S):]
        explore_arms = np.random.choice(leaderboard, size=self.S, replace=False)
        chosen_arms = np.unique(np.concatenate([leaders, explore_arms]))
        self.last_explored_arms = explore_arms
        return chosen_arms

    def observe_rewards(self, arms: List[int], rewards: List[float]) -> None:
        super().observe_rewards(rewards)
        for arm, reward in zip(arms, rewards):
            estimate = reward*self.A/self.S if arm in self.last_explored_arms else 0
            self.cum_est_rewards[arm] += estimate
        self.last_explored_arms = None


class StreeterFPML(MultiBandit):

    def __init__(self, A: int, T: int, T_1: int, T_2: int, S: int, n: int):
        assert T_1 * T_2 == T, "Time parameters must multiply to total budget"
        self.internal_fpml_instances = [
            FPML(A=A, T=T_1, S=S, n=n) for _ in range(T_2)
        ]
        super().__init__(A, T, n)

    def select_arms(self) -> List[int]:
        super().select_arms()
        self.grouped_arms = [
            self.internal_fpml_instances[t].select_arms()
            for t in range(self.T_2)
        ]
        return [arm for arms in self.grouped_arms for arm in arms]

    def observe_rewards(self, arms: List[int], rewards: List[float]) -> None:
        super().select_arms(arms, rewards)
        assert arms == [arm for arms in self.grouped_arms for arm in arms], \
            "Observed rewards must be for the chosen arms"

        i = 0; grouped_rewards = []
        for arms in self.grouped_arms:
            grouped_rewards.append([rewards[j] for j in range(i,i+len(arms))])
            i += len(arms)

        for t in range(self.T_2):
            self.internal_fpml_instances[t].observe_rewards(
                self.grouped_arms[t], self.grouped_rewards[t]
            )
        
        self.grouped_arms = None
