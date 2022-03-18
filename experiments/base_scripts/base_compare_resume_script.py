from blackboxbandits import compare
import pmlb
from multiprocess import Pool, Lock

dataset_names = list(map(lambda x: "clf-"+x.replace("_","-"), pmlb.classification_dataset_names)) \
                + list(map(lambda x: "reg-"+x.replace("_","-"), pmlb.regression_dataset_names))

base_comparison = compare.BaseOptimizerComparison(
    optimizers=["HyperOpt", "OpenTuner-BanditA", "OpenTuner-GA",
                "OpenTuner-GA-DE", "PySOT", "RandomSearch",
                "Scikit-GBRT-Hedge", "Scikit-GP-Hedge", "Scikit-GP-LCB"],
    classifiers=["MLP-adam","lasso"],
    datasets = dataset_names,
    metrics=["mse", "nll"],
    num_calls=20,
    num_repetitions=1,
    db_root = "./saved_results",
    datasets_root="./penn_datasets",
    parallel = True
)

base_comparison._dbid = dbid = "bo_20220228_124924_b7rkpeqg"

with open("./jobs_remaining.txt","r") as f:
    jobs = f.readlines()
    cmd_start_idx = jobs[0].find("bayesmark-exp")
    job_commands = {job[:cmd_start_idx-1]: job[cmd_start_idx:].strip()
                    for job in jobs}

    l = Lock() # We'll make `lock' global among pooled processes
    pool = Pool(base_comparison.num_workers, initializer=base_comparison._pool_init, initargs=(l,)) \
           if base_comparison.num_workers is not None \
           else Pool(initializer=base_comparison._pool_init, initargs=(l,))
    print(f"Starting processing {len(job_commands)} jobs.")
    pool.map(base_comparison._process_individual_command, job_commands.items(), chunksize=1)
    pool.close()
    pool.join()
    print("Finished processing all jobs.")

with open("key_dbids.txt", "a") as f:
  f.write("\n"+dbid)
