import pandas as pd
from .meta_optimizers import AbstractMetaOptimizer
import os
import subprocess
from typing import List, Optional
from bayesmark.serialize import XRSerializer
import numpy as np

class BaseOptimizerComparison:

    def __init__(self,
                 optimizers: List[str],
                 classifiers: List[str],
                 datasets: List[str],
                 metrics: List[str],
                 num_calls: int,
                 num_repetitions: int,
                 db_root: str):
        
        self.optimizers = optimizers
        self.classifiers = classifiers
        self.datasets = datasets
        self.metrics = metrics
        self.num_calls = num_calls
        self.num_repetitions = num_repetitions
        self.db_root = db_root
        self.dbid: Optional[str] = None

    def run(self) -> None:
        launcher_args = {
            "-dir": self.db_root,
            "-o": " ".join(self.optimizers),
            "-d": " ".join(self.datasets),
            "-c": " ".join(self.classifiers),
            "-m": " ".join(self.metrics),
            "-n": str(self.num_calls),
            "-r": str(self.num_repetitions)
        }
        # Run Bayesmark experiment launcher
        launcher_command = "bayesmark-launch " + " ".join(
            [key + " " + value for key, value in launcher_args.items()]
        )
        launcher_command += " -v"
        output = subprocess.run(launcher_command,
                                shell=True,
                                capture_output=True)

        # Get DBID
        stderr = output.stderr.decode("utf-8")
        dbid_start_index = stderr.find("Supply --db") + len("Supply --db") + 1
        dbid_end_index = stderr.find("to append to this experiment") - 1
        assert dbid_start_index != -1 and dbid_end_index != -1
        self.dbid = stderr[dbid_start_index:dbid_end_index]

        # Aggregate results
        subprocess.run(
            f'bayesmark-agg -dir "{self.db_root}" -b "{self.dbid}"',
            shell=True
        )

        # Analyse results
        subprocess.run(
            f'bayesmark-anal -dir "{self.db_root}" -b "{self.dbid}"',
            shell=True
        )

    def get_dbid(self) -> int:
        assert self.dbid is not None, "Must run comparison first."
        return self.dbid

    def get_results(self) -> pd.DataFrame:
        return self.get_results(self.dbid, self.db_root)

    @classmethod
    def get_results_for_dbid(cls, dbid: str, db_root: str) -> pd.DataFrame:
        # Read data saved by Bayesmark
        abs_db_root = os.path.abspath(db_root)
        saved_eval = XRSerializer.load_derived(db_root=abs_db_root,
                                               db=dbid,
                                               key="eval")
        eval = saved_eval[0].to_dataframe()
        saved_baseline = XRSerializer.load_derived(db_root=abs_db_root,
                                                   db=dbid,
                                                   key="baseline")
        baseline = saved_baseline[0].to_dataframe()

        # Massage into nice format
        baseline = baseline[["mean","best"]] \
            .groupby(level=["objective","function"]).min()
        baseline = baseline.unstack(level="objective")
        baseline.columns = ["visible_baseline", "generalization_baseline",
                            "visible_opt", "generalization_opt"]
        eval = eval.droplevel("suggestion")
        eval = eval.reorder_levels(["optimizer", "function",
                                    "study_id", "iter"])
        eval = eval.sort_values(eval.index.names)
        eval = eval.groupby(level=["optimizer","function","study_id"]).min()
        eval = eval.rename(columns={"_visible_to_opt": "visible"})
        eval = eval.rename(columns=lambda x: x + "_achieved")
        data = eval.join(baseline, on="function")

        # Calculate normalized scores
        data["generalization_score"] = cls._constrain(
            1 - (data["generalization_achieved"] - data["generalization_opt"]) \
            / (data["generalization_baseline"] - data["generalization_opt"])
        )
        data["visible_score"] = cls._constrain(
            1 - (data["visible_achieved"] - data["visible_opt"]) \
            / (data["visible_baseline"] - data["visible_opt"])
        )

        data = data.reindex(sorted(data.columns), axis=1)
        data.columns = data.columns.str.split("_", expand=True)

        return data

    @staticmethod
    def _constrain(series):
        return np.maximum(np.minimum(series, 1), 0)

class MetaOptimizerComparison:

    def __init__(self,
                 meta_optimizers: List[AbstractMetaOptimizer],
                 base_optimizers: List[str],
                 classifiers: List[str],
                 datasets: List[str],
                 metrics: List[str],
                 num_calls: int,
                 num_repetitions: int,
                 db_root: str):
        self.meta_optimizers = meta_optimizers
        self.base_optimizers = base_optimizers
        self.classifiers = classifiers
        self.datasets = datasets
        self.metrics = metrics
        self.num_calls = num_calls
        self.num_repetitions = num_repetitions
        self.db_root = db_root
        
        self.dbid: Optional[str] = None

        self.meta_comparison_completed = False

    def run_base_comparison(self) -> None:
        base_comparison = BaseOptimizerComparison(
            self.base_optimizers,
            self.classifiers,
            self.datasets,
            self.metrics,
            self.num_calls,
            self.num_repetitions,
            self.db_root)
        base_comparison.run()
        self.dbid = base_comparison.get_dbid()
        self.base_comparison_data = BaseOptimizerComparison \
            .get_results(self.dbid, self.db_root)

    def load_base_comparison(self, dbid: str) -> None:
        self.dbid = dbid
        self.base_comparison_data = BaseOptimizerComparison \
            .get_results(self.dbid, self.db_root)

    def run_meta_comparison(self):
        assert self.dbid is not None, "Must run or load base comparison first."
        self.meta_comparison_completed = True
        raise NotImplementedError()

    def results(self):
        raise NotImplementedError()

    def get_dbid(self):
        raise NotImplementedError()