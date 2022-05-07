from blackboxbandits import compare, bandits, synthetic

PREFIX = "synth4"

single = {f"single_arm{i}": synthetic.SingleArm(i=i)
          for i in range(4)}
best = {"best_3": synthetic.BestFixedTArms(T=3)}
top = {"top_3": synthetic.TopTBestArms(T=3)}
fpml = {"fpml_3": synthetic.BanditAlgorithm(
        bandit_type=bandits.FPMLWithGR,
        T=3, gamma=0)}
streeter_fpml = {"streeter_fpml_1x3": synthetic.BanditAlgorithm(
        bandit_type=bandits.StreeterFPML,
        T=3, T_1=1, T_2=3, gamma=0, gr=True)}
streeter_fpl = {"streeter_fpl_3": synthetic.BanditAlgorithm(
        bandit_type=bandits.StreeterFPL,
        T=3)}
streeter_exp3 = {"streeter_exp3_3": synthetic.BanditAlgorithm(
        bandit_type=bandits.Streeter,
        T=3)}
algos = {**single, **best, **top,
         **fpml, **streeter_fpml, **streeter_fpl, **streeter_exp3}


environment = synthetic.Synth4Environment(n=300)


comparison = compare.SyntheticComparison(
    environment=environment,
    algos=algos,
    parallel=False,
    num_repetitions=50
)


comparison.run()
results = comparison.full_results()
summary = comparison.summary()

results.to_csv(f"experiments/synth_results/{PREFIX}_results.csv")
summary.to_csv(f"experiments/synth_results/{PREFIX}_summary.csv")