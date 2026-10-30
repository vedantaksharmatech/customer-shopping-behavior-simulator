# Methodology

1. **Read and inspect:** check file readability and required headers, then load only timestamp, visitor ID, and event. Retailrocket timestamps are interpreted as Unix milliseconds in UTC.
2. **Clean:** exclude rows with missing visitor IDs, missing event types, unparseable timestamps, unrecognized events, and exact duplicates. Duplicate handling keeps the first identical row. Counts are reported; a retention rate below 50% stops modeling for review.
3. **Sessionize:** within each visitor, order by timestamp. A gap strictly greater than the configurable inactivity threshold begins a new session. The default 30 minutes is an analytical rule, not a ground-truth label.
4. **Map states:** view→BROWSING; addtocart→CART; transaction→PURCHASED. Truncate the sequence at its first transaction. Append inferred EXITED to sessions that have no transaction. A session beginning in transaction starts at PURCHASED, with no transition from an unknown state.
5. **Split:** order sessions by start time; place the first configured fraction in training and the remainder in testing. Whole sessions remain intact, avoiding transition leakage between sets.
6. **Estimate:** count adjacent state pairs in training and divide each transient row by its observed outgoing count. PURCHASED and EXITED are modeled as absorbing. No smoothing is used. An unobserved transient row stays zero and the simulator refuses to sample from such a model.
7. **Simulate:** sample next states from the learned row with NumPy's generator and a user seed. Stop at purchase, exit, or the maximum step count. Store sequences and terminal outcome.
8. **Validate:** compare simulated terminal proportions with the test-session outcomes. The validation is descriptive and depends on the sessionization and state abstraction.
9. **Export:** write summaries, matrices, validation metrics, sample simulated paths, and labelled plots under `outputs/`.

## Eventual absorbing outcomes

For transient states, write the matrix in blocks as `P = [[Q, R], [0, I]]`. When `(I-Q)` is invertible, absorption probabilities are `B = (I-Q)^-1 R`. The implementation solves this small linear system and reports purchase/exit probabilities from the selected initial state. This is a property of the estimated chain, not a measured conversion forecast.
