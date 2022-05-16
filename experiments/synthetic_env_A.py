from blackboxbandits import compare, bandits, synthetic

PREFIX = "synthA"

factors = [(a,b) for a in range(1,6) for b in range(1,6) if a*b < 6 and b > 1]

single = {
    f"single_arm{i}": synthetic.SingleArm(i=i)
    for i in range(10)}
best = {
    f"best_{T}": synthetic.BestFixedTArms(T=T)
    for T in range(1,6)}
top = {
    f"top_{T}": synthetic.TopTBestArms(T=T)
    for T in range(1,6)}
fpml = {
    f"fpml_{T}": synthetic.BanditAlgorithm(
        bandit_type=bandits.FPMLWithGR,
        T=T, gamma=0)
    for T in range(1,6)}
streeter_fpml = {
    f"streeter_fpml_{a}x{b}": synthetic.BanditAlgorithm(
        bandit_type=bandits.StreeterFPML,
        T=a*b, T_1=a, T_2=b, gamma=0, gr=True)
    for a,b in factors}
streeter_exp3 = {
    f"streeter_exp3_{T}": synthetic.BanditAlgorithm(
        bandit_type=bandits.Streeter,
        T=T)
    for T in range(1,6)}
algos = {**single, **best, **top,
         **fpml, **streeter_fpml, **streeter_exp3}


environment = synthetic.Synth1Environment(n=300)


comparison = compare.SyntheticComparison(
    environment=environment,
    algos=algos,
    parallel=True,
    num_repetitions=50
)


comparison.run()
results = comparison.full_results()
summary = comparison.summary()

results.to_csv(f"experiments/synth_results/{PREFIX}_results.csv")
summary.to_csv(f"experiments/synth_results/{PREFIX}_summary.csv")