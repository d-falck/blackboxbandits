import pandas as pd
from .meta_optimizers import AbstractMetaOptimizer
import os
import subprocess
from typing import List, Optional
from bayesmark.serialize import XRSerializer

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
        launcher_command = "bayesmark-launch " + " ".join([key + " " + value for key, value in launcher_args.items()])
        launcher_command += " -v"
        output = subprocess.run(launcher_command, shell=True, capture_output=True)

        # Get DBID
        stderr = output.stderr.decode("utf-8")
        dbid_start_index = stderr.find("Supply --db") + len("Supply --db") + 1
        dbid_end_index = stderr.find("to append to this experiment") - 1
        assert dbid_start_index != -1 and dbid_end_index != -1
        self.dbid = stderr[dbid_start_index:dbid_end_index]

        # Aggregate results
        subprocess.run(f'bayesmark-agg -dir "{self.db_root}" -b "{self.dbid}"', shell=True, capture_output=True)

        # Analyse results
        analysis_output = subprocess.run(f'bayesmark-anal -dir "{self.db_root}" -b "{self.dbid}"', shell=True, capture_output=True)
        self.results = analysis_output.stdout.decode("utf-8")

    def get_dbid(self) -> int:
        assert self.dbid is not None, "Must run comparison first."
        return self.dbid

    @staticmethod
    def get_results(dbid: str, db_root: str) -> pd.DataFrame:
        abs_db_root = os.path.abspath(db_root)
        agg_data = XRSerializer.load_derived(db_root=abs_db_root, db=dbid, key="perf")
        return agg_data[0].to_dataframe()

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
        self.base_comparison_data = BaseOptimizerComparison.get_results(self.dbid, self.db_root)

    def load_base_comparison(self, dbid: str) -> None:
        self.dbid = dbid
        self.base_comparison_data = BaseOptimizerComparison.get_results(self.dbid, self.db_root)

    def run_meta_comparison(self):
        assert self.dbid is not None, "Must run or load base comparison first."
        self.meta_comparison_completed = True
        raise NotImplementedError()

    def results(self):
        raise NotImplementedError()

    def get_dbid(self):
        raise NotImplementedError()