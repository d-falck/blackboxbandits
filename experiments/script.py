from blackboxbandits import compare

comparison = compare.BaseOptimizerComparison(
    optimizers=["HyperOpt", "OpenTuner-BanditA", "OpenTuner-GA",
                     "OpenTuner-GA-DE", "PySOT", "RandomSearch",
                     "Scikit-GBRT-Hedge", "Scikit-GP-Hedge", "Scikit-GP-LCB"],
    classifiers=["MLP-adam","SVM","ada","lasso"],
    datasets = ["breast", "digits", "iris", "wine", "boston", "diabetes"],
    metrics = ["mse", "nll"],
    num_calls = 15,
    num_repetitions = 2,
    db_root = "./data",
    parallel = True
)

comparison.run()
dbid = comparison.get_dbid()

with open("key_dbids.txt", "a") as f:
  f.write("\n"+dbid)
