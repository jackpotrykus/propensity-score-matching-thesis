# propensity-score-matching-thesis

Source code and resources for my MSc Statistics thesis on Propensity Score Matching (PSM)

# What is propensity score matching?

<!-- TODO: Put this into my own words... -->

Per Wikipedia:
> In the statistical analysis of observational data, propensity score matching (PSM) is a statistical matching technique that attempts to estimate the effect of a treatment, policy, or other intervention by accounting for the covariates that predict receiving the treatment. PSM attempts to reduce the bias due to confounding variables that could be found in an estimate of the treatment effect obtained from simply comparing outcomes among units that received the treatment versus those that did not. Paul R. Rosenbaum and Donald Rubin introduced the technique in 1983.

# Overview of Experiments

## Datasets
<!-- TODO: describe data generation -->

## Experiments

* Investigate the "PSM Paradox": examine performance of PSM as treatment imbalance $\to 0$
* Investigate the utility of the "Prognostic Score": propensity score matching vs. prognostic score matching vs two-score matching
* Comparison of score-based methods to feature vector norm based methods
* Comparison of greedy vs optimal matching in $1:k$ matching, for various values of $k$ (say $k \in \{1, \ldots, 10\}$)
* IDEA: PSM (+ Prognostic?) with "auto-coarsened" data

# Development Information

* Python package written in `python 3.8.6`
* Docstrings follow [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style, 
and makes exhaustive use of Python's gradual [`typing`](https://docs.python.org/3/library/typing.html) library
