"""Provides interfaces to run experiments comparing black-box optimizers and
combinations of them (which we call meta-optimizers) on various ML tasks.
"""

from xml.dom.minidom import parseString
import pandas as pd
from .meta import AbstractMetaOptimizer
from .bandits import AbstractMultiBandit
import os
import subprocess
from typing import List, Optional, Dict, Tuple
from bayesmark.serialize import XRSerializer
import numpy as np
from datetime import datetime
from tempfile import mkdtemp
from multiprocess import Pool, Lock
import datetime as dt
import itertools


class BaseOptimizerComparison:
    """Interface for comparing standard black-box optimizers on ML tasks.

    Parameters
    ----------
    optimizers : List[str]
        List of standard optimizer names known to Bayesmark.
    classifiers : List[str]
        List of sklearn classification methods known to Bayesmark.
    datasets : List[str]
        List of classification/regression datasets known to Bayesmark.
    metrics : List[str]
        List of loss functions known to Bayesmark. Must include at least one
        regression and one classification loss function if `datasets` includes
        both types of task.
    num_calls : int
        Number of function evaluations allowed by each optimizer on each task.
    num_repetitions : int
        Number of times to repeat the entire experiment for reliability.
    db_root : str
        Path to root folder in which a folder for this experiment's data will
        be created.
    datasets_root : Optional[str], optional
        Path to directory containing csv files for referenced datasets. Defaults
        to None.
    parallel : bool, optional
        Whether to run the experiments as a pool of tasks across multiple worker
        threads or not. Defaults to False.
    num_workers : Optional[int], optional
        Number of worker processes to use if parallelisation is enabled. If not
        specified defaults to the number of cpu cores.

    Attributes
    ----------
    Same as parameters.
    """

    def __init__(self,
                 optimizers: List[str],
                 classifiers: List[str],
                 datasets: List[str],
                 metrics: List[str],
                 num_calls: int,
                 num_repetitions: int,
                 db_root: str,
                 datasets_root: Optional[str] = None,
                 parallel: bool = False,
                 num_workers: Optional[int] = None):
        
        self.optimizers = optimizers
        self.classifiers = classifiers
        self.datasets = datasets
        self.metrics = metrics
        self.num_calls = num_calls
        self.num_repetitions = num_repetitions
        self.db_root = db_root
        self.datasets_root = datasets_root
        self.parallel = parallel
        self.num_workers = num_workers
        self._dbid: Optional[str] = None

    def run(self) -> None:
        """Run the comparison experiment defined by this class.

        May take a while, depending on whether parallelisation is enabled.

        Notes
        -----

        If parallelisation is enabled, this generates and uses a temporary
        file `jobs.txt` in the current directory.
        """
        launcher_args = {
            "-dir": self.db_root,
            "-o": " ".join(self.optimizers),
            "-d": " ".join(self.datasets),
            "-c": " ".join(self.classifiers),
            "-m": " ".join(self.metrics),
            "-n": str(self.num_calls),
            "-r": str(self.num_repetitions)
        }
        if self.datasets_root is not None:
            launcher_args["-dr"] = self.datasets_root
        if self.parallel: # Will create list of independent commands to run
            # Approximate number of indep. experiments in this comparison
            launcher_args["-nj"] = str(len(self.optimizers) * len(self.datasets) \
                                 * len(self.classifiers) * self.num_repetitions)

            # Generate dbid for whole batch
            folder_prefix = datetime.utcnow().strftime("bo_%Y%m%d_%H%M%S_")
            exp_subdir = mkdtemp(prefix=folder_prefix, dir=self.db_root)
            self._dbid = os.path.basename(exp_subdir)
            launcher_args["-b"] = self._dbid

            # Setup dbid folder
            for name in ["derived", "log", "eval", "time", "suggest_log"]:
                os.mkdir(os.path.join(exp_subdir, name))

        # Run Bayesmark experiment launcher
        launcher_command = "bayesmark-launch " + " ".join(
            [key + " " + value for key, value in launcher_args.items()]
        )
        launcher_command += " -v"
        p = subprocess.Popen(launcher_command,
                             stderr=subprocess.PIPE, shell=True)
        while p.poll() is None:
            l = p.stderr.readline().decode("utf-8")
            print(l)
            if "Supply --db" in l:
                dbid_string = l

        # If parallel we now need to run the generated commands
        if self.parallel:
            self._run_parallel_commands(self._dbid)
        else:
            # Get DBID
            dbid_start_index = dbid_string.find("Supply --db") \
                                + len("Supply --db") + 1
            dbid_end_index = dbid_string.find("to append to this experiment") - 1
            assert dbid_start_index != -1 and dbid_end_index != -1
            self._dbid = dbid_string[dbid_start_index:dbid_end_index]

        # Aggregate results
        os.system(f'bayesmark-agg -dir "{self.db_root}" -b "{self._dbid}"')

        # Analyse results (will compute baselines which we'll use)
        os.system(f'bayesmark-anal -dir "{self.db_root}" -b "{self._dbid}"')

    def get_dbid(self) -> int:
        """Get the unique DBID of this comparison.
        
        This is the name of the folder this comparison's data is saved in.
        Must run comparison first.

        Returns
        -------
        int
            The DBID of this experiment.
        """
        assert self._dbid is not None, "Must run comparison first."
        return self._dbid

    def get_results(self) -> pd.DataFrame:
        """Get the results of this comparison as a dataframe.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the results of this comparison.
        """
        return self.get_results_for_dbid(self._dbid, self.db_root)

    def _run_parallel_commands(self, dbid: int) -> None:
        with open("./jobs.txt","r") as f:
            jobs = f.readlines()[2:]
        cmd_start_idx = jobs[0].find("bayesmark-exp")
        job_commands = {job[:cmd_start_idx-1]: job[cmd_start_idx:].strip()
                        for job in jobs}
        
        l = Lock() # We'll make `lock' global among pooled processes
        pool = Pool(self.num_workers, initializer=self._pool_init, initargs=(l,)) \
               if self.num_workers is not None \
               else Pool(initializer=self._pool_init, initargs=(l,))
        print(f"Starting processing {len(job_commands)} jobs.")
        pool.map(self._process_individual_command, job_commands.items(), chunksize=1)
        pool.close()
        pool.join()
        print("Finished processing all jobs.")

    def _process_individual_command(self, job: Tuple[str, str]):
        with lock:
            print("Starting job", job[0], "at", dt.datetime.now().isoformat())
        try:
            subprocess.run(job[1], shell=True)
        except:
            with lock:
                print(f"Job {job[0]} failed")
        else:
            with lock:
                print("Finished job", job[0], "at", dt.datetime.now().isoformat())
            
    def _pool_init(self, l):
        global lock
        lock = l

    @classmethod
    def get_results_for_dbid(cls, dbid: str, db_root: str) -> pd.DataFrame:
        """Get the results of a particular comparison specified by its DBID.

        Parameters
        ----------
        dbid : str
            The DBID of the comparison to get results for.
        db_root : str
            Path to the directory containing the folder for
            the relevant comparison.
        
        Returns
        -------
        pd.DataFrame
            A dataframe containing the result of the specified comparison.
        """
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

        return data.dropna() # TODO: may want to remove dropna

    @staticmethod
    def _constrain(series):
        return np.maximum(np.minimum(series, 1), 0)


class MetaOptimizerComparison:
    """Interface for comparing *combinations* of standard black-box
    optimizers on ML tasks.

    Do not use the default constructor, use `from_base_comparison_setup` or
    `from_precomputed_base_comparison`.

    Parameters
    ----------
    meta_optimizers : Dict[str, meta.AbstractMetaOptimizer]
        Dictionary whose values are the instantiated meta-optimizers to include
        in this comparison (these will choose combinations of the base optimizers
        defined below) and whose keys are the names used to refer to them.
    db_root : str
        Path to root folder in which a folder for this experiment's data will
        be created.
    parallel_meta : bool, optional
        Whether to run the meta experiments as a pool of tasks across multiple
        worker threads or not. Defaults to False.
    num_workers : Optional[int], optional
        Number of worker processes to use if parallelisation is enabled. If not
        specified defaults to the number of cpu cores.

    Attributes
    ----------
    Same as parameters of whichever constructor used.
    """

    def __init__(self,
                 meta_optimizers: Dict[str, AbstractMetaOptimizer],
                 db_root: str,
                 parallel_meta: bool = False,
                 num_meta_repetitions: int = 1):
        self.meta_optimizers = meta_optimizers
        self.db_root = db_root
        self.parallel_meta = parallel_meta
        self.num_meta_repetitions = num_meta_repetitions
        
        self._meta_comparison_completed = False
        self._dbid: Optional[str] = None
        self._base_comparison_info_ready = False

    @classmethod
    def from_base_comparison_setup(cls,
                                   meta_optimizers: Dict[str, AbstractMetaOptimizer],
                                   base_optimizers: List[str],
                                   classifiers: List[str],
                                   datasets: List[str],
                                   metrics: List[str],
                                   num_calls: int,
                                   num_repetitions: int,
                                   db_root: str,
                                   datasets_root: Optional[str] = None,
                                   parallel_base: bool = False,
                                   parallel_meta: bool = False,
                                   num_workers: Optional[int] = None,
                                   num_meta_repetitions: int = 1):
        """Construct from base comparison setup info.
        
        Parameters
        ----------
        meta_optimizers : Dict[str, meta.AbstractMetaOptimizer]
            Dictionary whose values are the instantiated meta-optimizers to include
            in this comparison (these will choose combinations of the base optimizers
            defined below) and whose keys are the names used to refer to them.
        base_optimizers : List[str]
            List of standard optimizer names known to Bayesmark to be used as the
            base optimizers for this comparison.
        classifiers : List[str]
            List of sklearn classification methods known to Bayesmark.
        datasets : List[str]
            List of classification/regression datasets known to Bayesmark.
        metrics : List[str]
            List of loss functions known to Bayesmark. Must include at least one
            regression and one classification loss function if `datasets` includes
            both types of task.
        num_calls : int
            Number of function evaluations allowed by each optimizer on each task.
        num_repetitions : int
            Number of times to repeat the entire experiment for reliability.
        db_root : str
            Path to root folder in which a folder for this experiment's data will
            be created.
        datasets_root : Optional[str], optional
            Path to directory containing csv files for referenced datasets. Defaults
            to None.
        parallel_base : bool, optional
            Whether to run the base experiments as a pool of tasks across multiple
            worker threads or not. Defaults to False.
        parallel_meta : bool, optional
            Whether to run the meta experiments as a pool of tasks across multiple
            worker threads or not. Defaults to False.
        num_workers : Optional[int], optional
            Number of worker processes to use if parallelisation is enabled. If not
            specified defaults to the number of cpu cores.
        num_meta_repetitions : int, optional
            Number of times to re-run the meta-optimizers on each base optimizer
            study; averaging over meta-optimizer randomness. Defaults to 1.
        """

        instance = cls(meta_optimizers, db_root, parallel_meta, num_meta_repetitions)

        instance.base_optimizers = base_optimizers
        instance.classifiers = classifiers
        instance.datasets = datasets
        instance.metrics = metrics
        instance.num_calls = num_calls
        instance.num_repetitions = num_repetitions
        instance.datasets_root = datasets_root
        instance.parallel_base = parallel_base
        instance.num_workers = num_workers

        instance._base_comparison_info_ready = True

    @classmethod
    def from_precomputed_base_comparison(cls,
                                         dbid: str,
                                         meta_optimizers: Dict[str, AbstractMetaOptimizer],
                                         db_root: str,
                                         parallel_meta: bool = False,
                                         num_meta_repetitions: int = 1) -> None:
        """Construct by loading data from a previously run comparison of the base
        optimizers for this experiment.

        Parameters
        ----------
        dbid : str
            The DBID for the saved data to be loaded.
        meta_optimizers : Dict[str, meta.AbstractMetaOptimizer]
            Dictionary whose values are the instantiated meta-optimizers to include
            in this comparison (these will choose combinations of the base optimizers
            defined below) and whose keys are the names used to refer to them.
        db_root : str
            Path to root folder in which a folder for this experiment's data will
            be created.
        parallel_meta : bool, optional
            Whether to run the meta experiments as a pool of tasks across multiple
            worker threads or not. Defaults to False.
        num_workers : Optional[int], optional
            Number of worker processes to use if parallelisation is enabled. If not
            specified defaults to the number of cpu cores.
        """
        instance = cls(meta_optimizers, db_root, parallel_meta, num_meta_repetitions)
        instance._dbid = dbid
        instance._base_comparison_data = BaseOptimizerComparison \
            .get_results_for_dbid(instance._dbid, instance.db_root)
        return instance

    def run_base_comparison(self) -> None:
        """Run the base optimizers on the relevant tasks for this experiment.

        May take a while.
        """
        assert self._base_comparison_info_ready, "Must construct using `from_base_comparison_setup`."

        base_comparison = BaseOptimizerComparison(
            self.base_optimizers,
            self.classifiers,
            self.datasets,
            self.metrics,
            self.num_calls,
            self.num_repetitions,
            self.db_root,
            self.datasets_root,
            self.parallel_base,
            self.num_workers)
        base_comparison.run()
        self._dbid = base_comparison.get_dbid()
        self._base_comparison_data = base_comparison.get_results()

    def run_meta_comparison(self):
        """Run the comparison of meta-optimizers over the base optimizers.

        Must have first run or loaded the base optimizer comparison.
        """
        assert self._dbid is not None, "Must run or load base comparison first."
        self._meta_comparison_completed = True

        if self.parallel_meta:
            pool = Pool(self.num_workers) \
                   if self.num_workers is not None \
                   else Pool()
            all_results = pool.map(self._single_meta_run, list(range(self.num_meta_repetitions)))
        else:
            all_results = [self._single_meta_run() for _ in range(self.num_meta_repetitions)]
        
        results = pd.concat(all_results).groupby(level=0).mean() # TODO: Check this is right
        self._all_meta_results = results

    def full_results(self) -> pd.DataFrame:
        """Get the full results of this meta-comparison as a dataframe.
        
        Must have run the meta-comparison first.
        
        Returns
        -------
        pd.DataFrame
            A dataframe containing the average visible and generalization scores
            for each base and each meta optimizer, on each task.
        """
        assert self._meta_comparison_completed == True, \
             "Must complete comparison before getting results"
        base_results = self._base_comparison_data.xs("score", level=1, axis=1) \
                                        .rename(columns=lambda x: x+"_score")
        return pd.concat([base_results, self._all_meta_results]) \
                .groupby(["optimizer", "function"]).mean() \
                    [["visible_score", "generalization_score"]]

    def summary(self) -> pd.DataFrame:
        """Get summarized results of this meta-comparison as a dataframe.
        
        Must have run the meta-comparison first.
        
        Returns
        -------
        pd.DataFrame
            A dataframe containing visible and generalization scores
            for each base and each meta optimizer averaged over all tasks.
        """
        return self.full_results().groupby(["optimizer"]).mean()

    def get_dbid(self):
        """Get the unique DBID associated with the base optimizer comparison
        for this experiment.

        Must run or load base comparison first.

        Returns
        -------
        int
            The DBID of this base comparison for this experiment.
        """
        assert self._dbid is not None, "Must run or load base comparison first."
        return self._dbid

    def _single_meta_run(self) -> pd.DataFrame:
        all_results = []
        for rep in range(self.num_repetitions):
            results = []
            for meta_optimizer in self.meta_optimizers.values():
                comp_data = self._base_comparison_data.xs(rep, level="study_id")
                meta_optimizer.run(comp_data)
                results.append(meta_optimizer.get_results())
            all_results.append(pd.concat(results,
                                         keys=self.meta_optimizers.keys()))

        all_results = pd.concat(all_results, keys=list(range(self.num_repetitions)))
        all_results.index.rename(["study_id", "optimizer", "function"], inplace=True)
        all_results = all_results.reorder_levels(["optimizer", "function", "study_id"])
        all_results = all_results.sort_values(all_results.index.names)

        return all_results


class SyntheticBanditComparison:
    """Class implementing comparison of bandit algorithms on synthetic rewards.

    Parameters
    ----------
    rewards : pd.DataFrame
        A dataframe of float rewards in [0,1], with columns representing different
        actions and rows representing different rounds.
    bandits : List[AbstractMultiBandit]
        A dict of initialized bandit objects for comparison on these rewards,
        keyed by names for reference.
    best_fixed_budgets : List[int]
        A list of integers action budgets for which to evaluate the best fixed
        action set in hindsight (as well as running the given bandits).
    parallel: bool, optional
        Whether to use multiple processes for evaluation. Defaults to False.
    num_workers : int, optional
        How many worker processes to use, if `parallel` is True. If not specified,
        defaults to the number of available cores.
    num_repetitions : int, optional
        How many times to repeat the evaluation for reliability. Defaults to 1.

    Attributes
    ----------
    Same as parameters.
    """

    def __init__(self,
                 rewards: pd.DataFrame,
                 bandits: Dict[str, AbstractMultiBandit],
                 best_fixed_budgets: List[int],
                 parallel: bool = False,
                 num_workers: Optional[int] = None,
                 num_repetitions: int = 1):
        self.rewards = rewards
        self.bandits = bandits
        self.best_fixed_budgets = best_fixed_budgets
        self.parallel = parallel
        self.num_workers = num_workers
        self.num_repetitions = num_repetitions

        self.n = rewards.shape[0]
        self.A = rewards.shape[1]
        self.arms = rewards.columns.to_numpy()
        alg_names = list(bandits.keys()) + [f"best_fixed_{T}" for T in best_fixed_budgets]
        self._results = pd.DataFrame(0, index=rewards.index, columns=alg_names)
        self._has_run = False

        assert all(bandit.n == self.n and bandit.A == self.A for bandit in bandits), \
            "Given bandits must be set up for the right number of arms and rounds."

    def run(self) -> None:
        # Run bandits
        for round, rewards in self.rewards.iterrows():
            for bandit_name, bandit in self.bandits.items():
                arm_indices = bandit.select_arms()
                arms = self.arms[arm_indices]
                feedback = rewards[arms]
                bandit.observe_rewards(arm_indices, feedback.to_list())
                self._results[round, bandit_name] = feedback.max()

        # Compute best fixed-in-hindsight action sets
        for T in self.best_fixed_budgets:
            subsets = list(itertools.combinations(self.arms.tolist(), T))
            best_subset_rewards = np.full(self.n, -1)
            for subset in subsets:
                subset_rewards = self.rewards[:,subset].max(axis=1).to_numpy()
                if subset_rewards.sum() > best_subset_rewards.sum():
                    best_subset_rewards = subset_rewards
            self._results[:, f"best_fixed_{T}"] = best_subset_rewards

        self._has_run = True

    def get_results(self) -> pd.DataFrame:
        assert self._has_run, "Must run first."
        return self._results