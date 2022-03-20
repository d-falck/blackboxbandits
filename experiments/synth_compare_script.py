from blackboxbandits import compare, bandits, synthetic


PREFIX = "synth2"


factors = [(a,b) for a in range(1,9) for b in range(1,9) if a*b < 9 and b > 1]

best = {
    f"best_{T}": synthetic.BestFixedTArms(T=T)
    for T in range(1,9)}
top = {
    f"top_{T}": synthetic.TopTBestArms(T=T)
    for T in range(1,9)}
fpml = {
    f"fpml_{T}": synthetic.BanditAlgorithm(
        bandit_type=bandits.FPMLWithGR,
        T=T, gamma=0.2, epsilon=0.3)
    for T in range(1,9)}
streeter_fpml = {
    f"streeter_fpml_{a}x{b}": synthetic.BanditAlgorithm(
        bandit_type=bandits.StreeterFPML,
        T=a*b, T_1=a, T_2=b, gamma=0.2, epsilon=0.3, gr=True)
    for a,b in factors}
streeter_exp3 = {
    f"streeter_exp3_{T}": synthetic.BanditAlgorithm(
        bandit_type=bandits.Streeter,
        T=T)
    for T in range(1,9)}
algos = {**best, **top, **fpml, **streeter_fpml, **streeter_exp3}


environment = synthetic.Synth2Environment(n=500)


comparison = compare.SyntheticComparison(
    environment=environment,
    algos=algos,
    parallel=True,
    num_repetitions=100
)


comparison.run()
results = comparison.full_results()
summary = comparison.summary()

results.to_csv(f"experiments/synth_results/{PREFIX}_results.csv")
summary.to_csv(f"experiments/synth_results/{PREFIX}_summary.csv")