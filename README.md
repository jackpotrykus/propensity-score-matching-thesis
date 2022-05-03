# propensity-score-matching-thesis

Source code and resources for my MSc Statistics thesis on matching methods for use with observational studies.

Matching is a field within causal inference which attempts to reduce the effects of selection bias on parameter estimates;
for example, estimates of the Average Treatment Effect (ATE).

# Repository Overview

* `paper` contains the LaTeX source (and `.bib` files) for my thesis;
* `matching` is a python package I have developed for matching observational data;
* `experiments` contains experiments conducted using `matching`; run the experiments via
```zsh
python -m experiments
```
* `docs` contains extensive documentation for the package `matching`


# Development Information

* Install rqeuirements via `python -m pip install -r requirements.txt`
* Python package written in `python 3.8.6`
* Docstrings follow [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style, 
and makes exhaustive use of Python's gradual [`typing`](https://docs.python.org/3/library/typing.html) library
