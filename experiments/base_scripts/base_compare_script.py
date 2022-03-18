from blackboxbandits import compare
import pmlb

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

base_comparison.run()
dbid = base_comparison.get_dbid()

with open("key_dbids.txt", "a") as f:
  f.write("\n"+dbid)