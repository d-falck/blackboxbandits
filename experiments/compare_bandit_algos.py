from blackboxbandits import compare, bandits, meta

DBID = "bo_20220228_124924_b7rkpeqg" # Or change to the DBID (folder name) used for the base comparison

factors = [(a,b) for a in range(1,7) for b in range(1,7) if a*b < 7 and b > 1]

PREFIX = "algo"
REPS = 100

fpml = {f"fpml_{T}": meta.BanditMetaOptimizer(
            bandit_type=bandits.FPMLWithGR,
            T=T, gamma=0)
        for T in range(1,7)}
streeter_fpml = {
    f"streeter_fpml_{a}x{b}": meta.BanditMetaOptimizer(
        bandit_type=bandits.StreeterFPML,
        T=a*b, T_1=a, T_2=b, gamma=0, gr=True)
    for a,b in factors}
streeter_exp3 = {
    f"streeter_exp3_{T}": meta.BanditMetaOptimizer(
        bandit_type=bandits.Streeter,
        T=T)
    for T in range(1,7)}
meta_optimizers = {**fpml, **streeter_fpml, **streeter_exp3}

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
