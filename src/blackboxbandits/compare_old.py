import subprocess
from typing import List, Optional

class OptimizerComparison:

    DB_ROOT = "./data"

    def __init__(self,
        optimizers: List[str],
        classifiers: List[str],
        datasets: List[str],
        metrics: List[str],
        num_calls: int,
        num_repetitions: int
    ):
        self.optimizers = optimizers
        self.classifiers = classifiers
        self.datasets = datasets
        self.metrics = metrics
        self.num_calls = num_calls
        self.num_repetitions = num_repetitions
        self.dbid: Optional[str] = None

    def run(self) -> None:
        launcher_args = {
            "-dir": self.DB_ROOT,
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
        subprocess.run(f'bayesmark-agg -dir "{self.DB_ROOT}" -b "{self.dbid}"', shell=True, capture_output=True)

    def analyse(self) -> str:
        assert self.dbid is not None, "Must run comparison before analysing"
        analysis_output = subprocess.run(f'bayesmark-anal -dir "{self.DB_ROOT}" -b "{self.dbid}"', shell=True, capture_output=True)
        return analysis_output.stdout.decode("utf-8")