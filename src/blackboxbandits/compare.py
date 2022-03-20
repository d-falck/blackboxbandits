"""Provides interfaces to run experiments comparing black-box optimizers and
combinations of them (which we call meta-optimizers) on various ML tasks.
"""

import pandas as pd
import os
import subprocess
from typing import List, Optional, Dict, Tuple
from bayesmark.serialize import XRSerializer
import numpy as np
from datetime import datetime
from tempfile import mkdtemp
from multiprocessing import Pool, Lock
import datetime as dt
import time
from . import utils
from . import synthetic
from . import meta

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
    num_repetitions : int
        Number of times to repeat the entire base experiment for reliability.
    db_root : str
        Path to root folder in which a folder for this experiment's data will
        be created.
    parallel_meta : bool, optional
        Whether to run the meta experiments as a pool of tasks across multiple
        worker threads or not. Defaults to False.
    num_workers : Optional[int], optional
        Number of worker processes to use if parallelisation is enabled. If not
        specified defaults to the number of cpu cores.
    num_meta_repetitions : int, optional
        Number of times to re-run the meta-optimizers on each base optimizer
        study; averaging over meta-optimizer randomness. Defaults to 1.
    alternative_order : bool, optional
        ONLY VALID FOR THE PENN DATASET BASE COMPARISON. Changes the problem order
        to interleave MLP with lasso, instead of all MLP then all lasso.

    Attributes
    ----------
    Same as parameters of whichever constructor used.
    """

    def __init__(self,
                 meta_optimizers: Dict[str, meta.AbstractMetaOptimizer],
                 num_repetitions: int,
                 db_root: str,
                 parallel_meta: bool = False,
                 num_workers: Optional[int] = None,
                 num_meta_repetitions: int = 1,
                 alternative_order: bool = False):
        self.meta_optimizers = meta_optimizers
        self.num_repetitions = num_repetitions
        self.db_root = db_root
        self.parallel_meta = parallel_meta
        self.num_workers = num_workers
        self.num_meta_repetitions = num_meta_repetitions
        self.alternative_order = alternative_order
        
        self._meta_comparison_completed = False
        self._dbid: Optional[str] = None
        self._base_comparison_info_ready = False
        self._order = None

    @classmethod
    def from_base_comparison_setup(cls,
                                   meta_optimizers: Dict[str, meta.AbstractMetaOptimizer],
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
                                   num_meta_repetitions: int = 1,
                                   alternative_order: bool = False):
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
        alternative_order : bool, optional
            ONLY VALID FOR THE PENN DATASET BASE COMPARISON. Changes the problem order
            to interleave MLP with lasso, instead of all MLP then all lasso.
        """

        instance = cls(meta_optimizers, num_repetitions, db_root, parallel_meta,
                       num_workers, num_meta_repetitions, alternative_order)

        instance.base_optimizers = base_optimizers
        instance.classifiers = classifiers
        instance.datasets = datasets
        instance.metrics = metrics
        instance.num_calls = num_calls
        instance.datasets_root = datasets_root
        instance.parallel_base = parallel_base

        instance._base_comparison_info_ready = True

    @classmethod
    def from_precomputed_base_comparison(cls,
                                         dbid: str,
                                         meta_optimizers: Dict[str, meta.AbstractMetaOptimizer],
                                         db_root: str,
                                         parallel_meta: bool = False,
                                         num_workers: Optional[int] = None,
                                         num_meta_repetitions: int = 1,
                                         alternative_order: bool = False) -> None:
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
        num_meta_repetitions : int, optional
            Number of times to re-run the meta-optimizers on each base optimizer
            study; averaging over meta-optimizer randomness. Defaults to 1.
        alternative_order : bool, optional
            ONLY VALID FOR THE PENN DATASET BASE COMPARISON. Changes the problem order
            to interleave MLP with lasso, instead of all MLP then all lasso.
        """
        data = BaseOptimizerComparison.get_results_for_dbid(dbid, db_root)
        num_repetitions = len(data.index.unique(level="study_id").to_list())
        instance = cls(meta_optimizers, num_repetitions, db_root, parallel_meta,
                       num_workers, num_meta_repetitions, alternative_order)
        instance._dbid = dbid
        instance._base_comparison_data = data
        instance._calculate_order()
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
        self._calculate_order()

    def run_meta_comparison(self):
        """Run the comparison of meta-optimizers over the base optimizers.

        Must have first run or loaded the base optimizer comparison.
        """
        assert self._dbid is not None, "Must run or load base comparison first."
        results = [self._single_meta_run(i) for i in range(self.num_meta_repetitions)]
        self.meta_results = pd.concat(results, keys=range(len(results)), names=["meta_rep"])
        
        self._meta_comparison_completed = True

    def full_results(self) -> pd.DataFrame:
        """Get mean and std (over meta-repetitions) of the individual meta-optimizer
        performances (for each problem separately).
        
        Must have run the meta-comparison first. Averages are taken over base study
        repetitions first.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The first dataframe contains the mean validation and test scores, the second
            the standard deviations.
        """
        assert self._meta_comparison_completed == True, \
             "Must complete comparison before getting results"
        
        # Avg over base studies
        samples = self.meta_results.drop(columns="arms").groupby(["optimizer", "function", "meta_rep"]).mean() \
            [["visible_score", "generalization_score"]]
        
        # Stats over meta repetitions
        mean = samples.groupby(["optimizer", "function"]).mean()
        std = samples.groupby(["optimizer", "function"]).std()
        
        # Get base result means to add
        base_results = self._base_comparison_data.xs("score", level=1, axis=1) \
                                        .rename(columns=lambda x: x+"_score") \
                                        .groupby(["optimizer", "function"]).mean() \
                                        [["visible_score", "generalization_score"]]
        mean = pd.concat([base_results, mean])
        
        return mean, std

    def summary(self) -> pd.DataFrame:
        """Get mean and std (over meta-repetitions) of the average meta-optimizer
        scores (over all problems).
        
        Must have run the meta-comparison first. Averages are taken over base study
        repetitions first.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The first dataframe contains the mean validation and test scores, the second
            the standard deviations.
        """
        
        # Avg over base studies and problems
        samples = self.meta_results.drop(columns="arms").groupby(["optimizer", "meta_rep"]).mean() \
            [["visible_score", "generalization_score"]]
        
        # Stats over meta repetitions
        mean = samples.groupby(["optimizer"]).mean()
        std = samples.groupby(["optimizer"]).std()
        
        # Get base result means to add
        base_results = self._base_comparison_data.xs("score", level=1, axis=1) \
                                        .rename(columns=lambda x: x+"_score") \
                                        .groupby(["optimizer"]).mean() \
                                        [["visible_score", "generalization_score"]]
        mean = pd.concat([base_results, mean])
        
        return mean, std

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

    def _single_meta_run(self, i) -> pd.DataFrame:
        to_print = f"Starting meta-comparison, repetition {i+1} of {self.num_meta_repetitions}"
        print(to_print)
        print("-"*len(to_print))
        start = time.time()
        
        global lock
        lock = Lock()
        if self.parallel_meta:
            with Pool(self.num_workers) as pool:
                results = pool.map(self._process_meta_optimizer,
                                   self.meta_optimizers, chunksize=1)
        else:
            results = map(self._process_meta_optimizer,
                          self.meta_optimizers)
        
        results = list(map(list, zip(*results))) # Transpose
        results = [pd.concat(dfs, keys=self.meta_optimizers.keys()) for dfs in results]

        results = pd.concat(results, keys=list(range(self.num_repetitions)))
        results.index.rename(["study_id", "optimizer", "function"], inplace=True)
        results = results.reorder_levels(["optimizer", "function", "study_id"])
        results = results.sort_values(results.index.names)
        
        end = time.time()
        elapsed = end - start
        print(f"Finished meta-comparison in {elapsed} seconds")
        return results

    def _process_meta_optimizer(self, name):
        np.random.seed()
        meta_optimizer = self.meta_optimizers[name]
        results = []
        for rep in range(self.num_repetitions):
            with lock:
                print(f"Running meta-optimizer {name} on base " \
                      f"study {rep+1} of {self.num_repetitions}")
            comp_data = self._base_comparison_data.xs(rep, level="study_id")
            meta_optimizer.run(comp_data, function_order=self._order)
            res = meta_optimizer.get_results()
            res["arms"] = meta_optimizer.get_history()["arms"]
            results.append(res)
        return results

    def _calculate_order(self):
        assert self._dbid is not None
        if self.alternative_order:
            functions = self._base_comparison_data.index.unique("function").to_list()
            mlp = list(filter(lambda x: x.startswith("MLP"), functions))
            lasso = list(filter(lambda x: x.startswith("lasso"), functions))
            order = utils.interleave_lists(mlp, lasso)
            self._order = order


class SyntheticComparison:
    """Class implementing comparison of algorithms on synthetic rewards.

    Parameters
    ----------
    environment : synthetic.AbstractEnvironment
        An environment specification to generate synthetic rewards.
    algos : Dict[str, synthetic.AbstractAlgorithm]
        A dict of named algorithms to run on the generated synthetic rewards.
    parallel: bool, optional
        Whether to use multiple processes for evaluation. Defaults to False.
    num_workers : int, optional
        How many worker processes to use, if `parallel` is True. If not specified,
        defaults to the number of available cores.
    num_repetitions : int, optional
        How many times to repeat the evaluation for reliability. Repetitions are
        over randomness in algorithms and in the environment. Defaults to 1.

    Attributes
    ----------
    Same as parameters.
    """

    def __init__(self,
                 environment: synthetic.AbstractEnvironment,
                 algos: Dict[str, synthetic.AbstractAlgorithm],
                 parallel: bool = False,
                 num_workers: Optional[int] = None,
                 num_repetitions: int = 1):
        self.environment = environment
        self.algos = algos
        self.parallel = parallel
        self.num_workers = num_workers
        self.num_repetitions = num_repetitions
        self._has_run = False

    def run(self) -> None:
        """Run the comparison of algorithms.
        """
        results = [self._single_run(rep) for rep in range(self.num_repetitions)]
        self.results = pd.concat(results,
                                 keys=range(len(results)),
                                 names=["rep"])
        self._has_run = True

    def full_results(self) -> pd.DataFrame:
        """Get mean and std (over trials) of individual algorithm
        performances on each round separately.
        
        Must have run the comparison first.
        
        Returns
        -------
        pd.DataFrame
        """
        assert self._has_run, "Must run first."
        samples = self.results.loc[:,"score"] # Series

        mean = samples.groupby(["algo", "round"]).mean()
        std = samples.groupby(["algo", "round"]).std()
        return pd.DataFrame({"mean": mean, "std": std})

    def summary(self) -> pd.DataFrame:
        """Get mean and std (over trials) of average algorithm
        scores (over all rounds).
        
        Must have run the comparison first.
        
        Returns
        -------
        pd.DataFrame
        """
        assert self._has_run, "Must run first."
        samples = self.results.loc[:,"score"].groupby(["algo","rep"]).mean()

        mean = samples.groupby("algo").mean()
        std = samples.groupby("algo").std()
        return pd.DataFrame({"mean": mean, "std": std})

    def _single_run(self, rep):
        to_print = f"Starting trial {rep+1} of {self.num_repetitions}"
        print(to_print)
        print("-"*len(to_print))
        start = time.time()

        self._rewards = self.environment.generate_rewards()

        global lock
        lock = Lock()
        if self.parallel:
            with Pool(self.num_workers) as pool:
                results = pool.map(self._process_algo, self.algos, chunksize=1)
        else:
            results = map(self._process_algo, self.algos)

        results = pd.concat(results, keys=self.algos.keys(), names=["algo"])

        end = time.time()
        elapsed = end - start
        print(f"Finished trial in {elapsed} seconds")
        return results

    def _process_algo(self, name):
        np.random.seed()
        with lock:
            print(f"Running algorithm {name}")
        algo = self.algos[name]
        algo.run(self._rewards)
        scores = algo.get_results()
        history = list(map(lambda x: ",".join(map(str,x)), algo.get_history()))
        df = pd.DataFrame({"score": scores, "choice": history},
                            index=self._rewards.index)
        df.index.name = "round"
        return df