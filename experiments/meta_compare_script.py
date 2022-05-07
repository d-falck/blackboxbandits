from blackboxbandits import compare, bandits, meta, utils
import pandas as pd
import numpy as np

DBID = "bo_20220228_124924_b7rkpeqg"

#-------FPML VARIANT--------------

PREFIX = "fpml_updated"
S = 1
GAMMA = lambda T: 1/T
REPS = 100

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

#-------PERTURBATION RATE--------------

# PREFIX = "pert_updated"
# REPS = 50
# meta_optimizers = {
#     f"fpml_gr_probexp_{epsilon:.3f}_{T}": meta.BanditMetaOptimizer(
#         bandit_type=bandits.FPMLWithGR,
#         T=T, gamma=0, epsilon=epsilon)
#     for epsilon in np.logspace(-3,3,num=13)
#     for T in range(1,7)
# }

#-------EXPLORATION RATE--------------

# PREFIX = "exp_updated"
# REPS = 50
# meta_optimizers = {
#     f"fpml_gr_probexp_{gamma:.1f}_{T}": meta.BanditMetaOptimizer(
#         bandit_type=bandits.FPMLWithGR,
#         T=T, gamma=gamma)
#     for gamma in np.arange(0.0,1.1,0.1)
#     for T in range(1,7)
# }

#---------MAIN COMPARISON---------------

# factors = [(a,b) for a in range(1,7) for b in range(1,7) if a*b < 7 and b > 1]

# PREFIX = "streeter_updated"
# REPS = 100

# fpml = {f"fpml_{T}": meta.BanditMetaOptimizer(
#             bandit_type=bandits.FPMLWithGR,
#             T=T, gamma=0)
#         for T in range(1,7)}
# streeter_fpml = {
#     f"streeter_fpml_{a}x{b}": meta.BanditMetaOptimizer(
#         bandit_type=bandits.StreeterFPML,
#         T=a*b, T_1=a, T_2=b, gamma=0, gr=True)
#     for a,b in factors}
# streeter_exp3 = {
#     f"streeter_exp3_{T}": meta.BanditMetaOptimizer(
#         bandit_type=bandits.Streeter,
#         T=T)
#     for T in range(1,7)}
# meta_optimizers = {**fpml, **streeter_fpml, **streeter_exp3}

#-----------BEST------------

# PREFIX = "best"
# REPS = 1
# meta_optimizers = {
#     f"best_{T}": meta.BestFixedTAlgos(T=T)
#     for T in range(1,7)}

#-----------BEST VS LEADERBOARD------------

# PREFIX = "bestvtop"
# REPS = 1
# best = {
#     f"best_{T}": meta.BestFixedTAlgos(T=T)
#     for T in range(1,10)}
# top = {
#     f"top_{T}": meta.TopTBestAlgos(T=T)
#     for T in range(1,10)}
# meta_optimizers = {**best, **top}

#===============================================================

meta_comparison = compare.MetaOptimizerComparison \
                         .from_precomputed_base_comparison(
    dbid=DBID,
    meta_optimizers=meta_optimizers,
    db_root="experiments/base_results",
    parallel_meta=True,
    num_meta_repetitions=REPS,
    alternative_order=False
)

meta_comparison.run_meta_comparison()
results_mean, results_std = meta_comparison.full_results()
summary_mean, summary_std = meta_comparison.summary()

results_mean.to_csv(f"experiments/meta_results/{PREFIX}_results_mean.csv")
results_std.to_csv(f"experiments/meta_results/{PREFIX}_results_std.csv")
summary_mean.to_csv(f"experiments/meta_results/{PREFIX}_summary_mean.csv")
summary_std.to_csv(f"experiments/meta_results/{PREFIX}_summary_std.csv")
