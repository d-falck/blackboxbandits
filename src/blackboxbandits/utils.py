import pandas as pd
from ipywidgets import interact
from typing import List

def filter_results(results: pd.DataFrame, budget: int):
    """Filters a meta-comparison summary dataframe by budget.
    
    Parameters
    ----------
    results : pd.DataFrame
        The summary output by a `compare.MetaOptimizerComparison` object.
    budget : int
        The algorithm-budget to filter by.

    Returns
    -------
    pd.DataFrame
        A dataframe containing only the results of meta-algorithms with the
        specified time budget, sorted by visible score.
    """
    n = budget
    valid = lambda x: _is_budgeted_bandit(n)(x) or (n == 1 and not "_" in x[-7:])
    return results[results.index.to_series().apply(valid)] \
            .sort_values("visible_score", ascending=False)

def _endings(n) -> List[str]:
    factors = [(i, n//i) for i in range(1,n+1) if n % i == 0]
    return [f"_{n}"] + [f"_{a}x{b}" for a,b in factors]

def _is_budgeted_bandit(n):
    return lambda x: any(map(lambda end: x.endswith(end), _endings(n)))

def visualise_by_budget(results: pd.DataFrame, max=7) -> None:
    """Shows an interactive widget to explore a summary dataframe by budget.
    
    Parameters
    ----------
    results : pd.DataFrame
        The summary output by a `compare.MetaOptimizerComparison` object.
    max : int
        The maximum time budget mentioned in the results dataframe.
    """
    def f(n=1):
        return filter_results(results, n)
    interact(f, n=(1,max))