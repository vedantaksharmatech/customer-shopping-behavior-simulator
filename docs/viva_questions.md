# Viva questions and concise answers

1. **What is a Markov Chain?** A model of movement among states using transition probabilities.
2. **What is the Markov property?** Given the current state, the next-state distribution does not depend on earlier states.
3. **What is a transition matrix?** A table whose entry in row i, column j is the probability of moving from i to j.
4. **Why must a probability row sum to one?** The next state must be one of the modeled states.
5. **What is a stochastic matrix?** A non-negative square matrix whose rows sum to one.
6. **What is an absorbing state?** A state that, once reached, remains the state with probability one.
7. **Why are Purchased and Exited absorbing?** They mark the modeled session's end in this first version.
8. **How is the transition matrix estimated?** Count each adjacent state pair in training sessions and divide by the outgoing count from its source state.
9. **Why use real data?** It grounds the estimated transition frequencies in recorded interactions rather than invented values.
10. **Why simulate when real data exists?** Simulation helps explore paths and expected outcomes implied by the fitted model; it does not replace validation.
11. **What is sessionization?** Grouping a visitor's events into visits using an inactivity rule.
12. **Why use a 30-minute threshold?** It is a common, simple analytical convention and is configurable; it is not known ground truth.
13. **Why split training and testing sessions?** To assess how the model behaves on sessions it did not use to estimate probabilities.
14. **What is data leakage?** Information from evaluation data influencing model fitting or training.
15. **What does conversion rate measure here?** The fraction of modeled sessions whose observed terminal outcome is PURCHASED.
16. **How is eventual purchase probability calculated?** Solve the absorbing-chain equations for the probability of eventually reaching PURCHASED from a starting state.
17. **What is a stationary distribution?** A state distribution unchanged by another transition step, when such a distribution exists.
18. **Why might simulation differ from actual behavior?** Sampling variability, sparse data, temporal drift, and model assumptions can all cause differences.
19. **What are key project limitations?** Four coarse states, inferred exits and sessions, historical data, first-order dependence, and first-purchase stopping.
20. **How could the project improve?** Test alternate session rules, add context, quantify uncertainty, and validate on more recent independent data.
21. **What does absorbing mean in the matrix?** The terminal state's row has probability one on its own diagonal entry.
22. **Why are events grouped by visitor?** Different visitors must not form transitions with each other.
23. **What happens to unknown event types?** They are counted and excluded, and the quality report records them.
24. **Why remove exact duplicate rows?** Repeated identical records can falsely inflate transition counts; the first copy is kept.
25. **What is the random seed for?** It makes the same simulation reproducible with the same parameters and software behavior.
26. **What does a zero transition row mean?** The training data contains no outgoing examples for that state; this implementation does not invent probabilities.
27. **Why stop at the first transaction?** The modeled chain treats purchase as terminal, so post-purchase events are outside the modeled journey.
28. **What is an inferred exit?** A terminal label appended when a session ends without a transaction; it is not a source event.
29. **What is Monte Carlo simulation?** Repeated random sampling from a probability model to approximate its outcome distribution.
30. **Does this project prove behavior or causation?** No. It describes associations in the supplied data under explicit assumptions.
