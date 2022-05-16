# Bandits for Black Box Optimization

A package for comparing black-box optimization algorithms and bandit combinations of them on online streams of ML hyperparameter-selection tasks. Written for the paper _"Trading Off Resource Budgets for Improved Regret Bounds"_.

## Package installation

Run `pip install git+https://github.com/d-falck/blackboxbandits.git`.

_Note it may be necessary to use my forked version of Bayesmark to avoid one assertion error; this can be installed by cloning the repo `https://github.com/d-falck/bayesmark` and running `pip install -e .`._

## Package documentation

Package documentation can be found in the `docs` directory and are auto-generated at https://d-falck.github.io/blackboxbandits/.

## Experimental results

The experimental results used for the paper are included in the `experiments` directory:

- `experiments/base_results` contains the base results for each of the 9 black-box optimizers on each PMLB dataset (in a format readable by the Bayesmark package).
- The results from our meta-comparison of bandit algorithms over these optimizers are in `experiments/meta_results`.
- The results from our evaluations of bandit algorithms in synthetic environments are in `experiments/synth_results`.
- The PMLB datasets we used are included in `experiments/penn_datasets` for convenience.

## Reproduction

The four scripts `base_compare_script.py`, `meta_compare_script.py`, `synth_compare_script.py`, `synth4_compare_script.py` were used to generate these results and can be executed to reproduce them. The middle two each contain several sections of which only one should be uncommented before running.

## License

