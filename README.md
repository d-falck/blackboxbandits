# Bandits for Black Box Optimization

A package for comparing black-box optimization algorithms and bandit combinations of them on online streams of ML hyperparameter-selection tasks. Written for the paper _"Trading Off Resource Budgets for Improved Regret Bounds"_.

## Package installation

Run `pip install -e .` from the repository root.

_Note it may be necessary to use our forked version of Bayesmark to avoid one assertion error; this can be installed by cloning the repo `https://github.com/d-falck/bayesmark` and running `pip install -e .`._

## Package documentation

Package documentation can be found in the `docs` directory and are auto-generated at https://d-falck.github.io/blackboxbandits/.

## Experimental results

The experimental results used for the paper are included in the `experiments` directory:

- `experiments/base_results` contains the base results for each of the 9 black-box optimizers on each PMLB dataset (in a format readable by the Bayesmark package).
- `experiments/meta_results` contains the results from our meta-comparison of bandit algorithms over these optimizers are in .
- `experiments/synth_results` contains the results from our evaluations of bandit algorithms in synthetic environments are in `.
- The PMLB datasets we used are included in `experiments/penn_datasets` for convenience.

## Reproduction

The experimental results can be reproduced using the scripts in the `experiments` directory:

- `experiments/evaluate_optimizers.py` will reproduce the base optimizer scores on the PMLB datasets (this will take a lot of compute)
- `experiments/compare_bandit_algos.py` will reproduce the bandit algorithm scores given the completed base comparison
- `experiments/best_in_hindsight.py` will do the same for the best-in-hindsight optimizer sets
- `experiments/synthetic_env_X.py` for `X=A,B,C` will reproduce the bandit algorithm scores on the three synthetic environments in the paper

## License

