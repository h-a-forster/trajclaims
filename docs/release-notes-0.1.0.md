# trajclaims 0.1.0

First version. Study results are in `results/v1`, with findings in `README.md`.

- Pinned extraction of the Terminal-Bench 2 trajectory dataset, with closing-statement handling per scaffold.
- Frozen rule labeller (`rules-1`) validated on 573 blind-labelled messages.
- False-completion, honest-failure and verification estimands with task-clustered intervals, misclassification correction and a bridge sample for lost messages.
- Model against scaffold decomposition and a task-disjoint TF-IDF detector.
