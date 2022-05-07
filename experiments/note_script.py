from blackboxbandits import compare, bandits, synthetic


PREFIX = "note"


explore = synthetic.BanditAlgorithm(
    bandit_type=bandits.FPMLProb,
    T=1, gamma=0.2, epsilon=0.01
)
resample = synthetic.BanditAlgorithm(
    bandit_type=bandits.FPMLWithGR,
    T=1, gamma=0, epsilon=0.01
)
algos = {"explore": explore, "resample": resample}


environment = synthetic.TomCounterexample(n=300)


comparison = compare.SyntheticComparison(
    environment=environment,
    algos=algos,
    parallel=True,
    num_repetitions=200
)


comparison.run()
results = comparison.full_results()
summary = comparison.summary()

results.to_csv(f"experiments/synth_results/{PREFIX}_results.csv")
summary.to_csv(f"experiments/synth_results/{PREFIX}_summary.csv")