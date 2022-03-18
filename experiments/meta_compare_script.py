from blackboxbandits import compare, bandits, meta, utils
import pandas as pd
import numpy as np

DBID = "bo_20220228_124924_b7rkpeqg"

S = 1
GAMMA = lambda T: 1/T

fpml_fixedexp = {
    f"fpml_fixedexp_{T}": meta.BanditMetaOptimizer(
        bandit_type=bandits.FPMLFixed,
        T=T, S=S)
    for T in range(2,7)}
fpml_probexp = {
    f"fpml_probexp_{T}": meta.BanditMetaOptimizer(
        bandit_type=bandits.FPMLProb,
        T=T, gamma=GAMMA(T))
    for T in range(2,7)}
fpml_gr_noexp = {
    f"fpml_gr_noexp_{T}": meta.BanditMetaOptimizer(
        bandit_type=bandits.FPMLWithGR,
        T=T, gamma=0)
    for T in range(2,7)}
fpml_gr_probexp = {
    f"fpml_gr_probexp_{T}": meta.BanditMetaOptimizer(
        bandit_type=bandits.FPMLWithGR,
        T=T, gamma=GAMMA(T))
    for T in range(2,7)}

meta_optimizers = {**fpml_fixedexp, **fpml_probexp, **fpml_gr_noexp, **fpml_gr_probexp}

# meta_optimizers = {
#     f"fpml_gr_probexp_{gamma:.1f}_{T}": meta.BanditMetaOptimizer(
#         bandit_type=bandits.FPMLWithGR,
#         T=T, gamma=gamma)
#     for gamma in np.arange(0.0,1.1,0.1)
#     for T in range(1,7)
# }

meta_comparison = compare.MetaOptimizerComparison \
                         .from_precomputed_base_comparison(
    dbid=DBID,
    meta_optimizers=meta_optimizers,
    db_root = "./base_results",
    parallel_meta = True,
    num_meta_repetitions = 100
)

meta_comparison.run_meta_comparison()
results_mean, results_std = meta_comparison.full_results()
summary_mean, summary_std = meta_comparison.summary()

results_mean.to_csv("meta_results/fpml_results_mean.csv")
results_std.to_csv("meta_results/fpml_results_std.csv")
summary_mean.to_csv("meta_results/fpml_summary_mean.csv")
summary_std.to_csv("meta_results/fpml_summary_std.csv")
