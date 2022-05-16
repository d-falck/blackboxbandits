from blackboxbandits import compare, meta

DBID = "bo_20220228_124924_b7rkpeqg" # Or change to the DBID (folder name) used for the base comparison

PREFIX = "bih"
REPS = 1
meta_optimizers = {
    f"best_{T}": meta.BestFixedTAlgos(T=T)
    for T in range(1,7)}

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
