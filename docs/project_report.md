# Customer Shopping Behavior Simulator Using Markov Chains

## Abstract

This project estimates a first-order Markov Chain from e-commerce event sessions and simulates shopping journeys. A real-data run on a random sample of 10,000 visitor IDs produced 12,240 sessions; the observed and simulated holdout purchase proportions were 0.6127% and 0.6000%, respectively. Findings are specific to this sample and are not claims of general predictive accuracy.

## Introduction

Online stores record sequences of views, cart additions, and transactions. Modeling those sequences can offer a compact description of how sessions progress.

## Problem statement

Convert clickstream event records into customer sessions, estimate state transitions, simulate paths, and compare simulated outcomes with held-out sessions without presenting synthetic output as real evidence.

## Objectives

- Validate and clean the event file.
- Construct sessions using a configurable inactivity threshold.
- Estimate transition probabilities from training sessions.
- Simulate reproducible journeys and validate on held-out sessions.

## Existing approach and proposed approach

Descriptive event counts do not represent event order. The proposed approach counts adjacent state transitions in complete sessions and simulates from their empirical probabilities.

## Dataset description

Retailrocket E-commerce Dataset, downloaded manually from Kaggle. The local `events.csv` contains 2,756,101 event rows and is 94,237,913 bytes. Its columns are `timestamp`, `visitorid`, `event`, `itemid`, and `transactionid`. This experiment sampled 10,000 visitor IDs and retained all 18,408 event rows belonging to them.

## Methodology

See [methodology.md](methodology.md). Sessions use a 30-minute default inactivity boundary, first-purchase truncation, and inferred exits.

## Markov Chain theory

A transition matrix `P` has entries `P(i,j)` equal to the conditional probability of next state `j` given current state `i`. Each supported row sums to one. Purchase and exit are absorbing states.

## System architecture

CSV loader → cleaner → session builder → mapper → chronological split → transition estimator → simulator → validator → CSV and chart outputs.

## Data preprocessing

Missing required values, invalid timestamps, unsupported event types, and exact duplicate rows are excluded. The first identical record is retained. For this sample, 18,408 rows were retained and 0 excluded. It contained 17,866 views, 436 add-to-cart events, and 106 transactions.

## Transition matrix estimation

Frequency-based row normalization is used without smoothing. For the sampled run, the fitted matrix (row and column order BROWSING, CART, PURCHASED, EXITED) was:

| From → To | BROWSING | CART | PURCHASED | EXITED |
|---|---:|---:|---:|---:|
| BROWSING | 0.3093 | 0.0155 | 0.0015 | 0.6737 |
| CART | 0.3651 | 0.1556 | 0.1302 | 0.3492 |
| PURCHASED | 0 | 0 | 1 | 0 |
| EXITED | 0 | 0 | 0 | 1 |

## Simulation algorithm

Start each journey at BROWSING by default. Sample a next state using the current matrix row and seeded NumPy generator. Stop at PURCHASED, EXITED, or configured step limit.

## Experimental setup

Python 3.12.3; input CSV size 94,237,913 bytes; 10,000 sampled visitor IDs; 30-minute inactivity threshold; chronological 80/20 session split; 1,000 simulated journeys; random seed 42. The sample formed 12,240 sessions (9,792 training and 2,448 holdout).

## Results and observations

This real-data sample had 83 purchased sessions and 12,157 inferred exits across all 12,240 sessions. In the 2,448-session holdout, the observed purchase proportion was 0.6127% (15 sessions) and the observed exit proportion was 99.3873%. Of 1,000 simulated journeys, 6 ended in PURCHASED (0.6000%) and 994 in EXITED (99.4000%); none reached the step limit. These are sample-specific findings, not estimates proven to generalize to all visitors.

## Validation and evaluation

Compare terminal-outcome proportions and transition frequencies between training-estimated simulations and held-out real sessions. The absolute error in purchase proportion was 0.000127 (about 0.0127 percentage points); exit had the same absolute error. This is a close terminal-outcome match for this sampled holdout and seed, not a predictive accuracy score. Larger-sample validation would help assess stability because purchases are rare.

## Limitations

See [limitations.md](limitations.md): state abstraction, inferred sessions/exits, first-order assumption, historical sample, and first-purchase truncation.

## Future scope

Evaluate alternate session gaps, add context features, assess temporal drift, quantify uncertainty, and use additional datasets for external validation.

## Conclusion

The implementation provides an explainable probability-estimation and simulation pipeline. In this 10,000-visitor sample, simulated and observed holdout terminal-outcome proportions were close. A larger visitor sample or full-data run would help assess how stable that result is.

## References

1. Retailrocket E-commerce Dataset, Kaggle: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
2. Norris, J. R. *Markov Chains*. Cambridge University Press, 1997.
