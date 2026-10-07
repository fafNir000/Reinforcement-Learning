# Assignment: Multi-Armed Bandits for Headline Recommendation

Sep 30, 2026 · @Ruben

## Overview

You will implement UCB1 and Thompson Sampling, compare them against ε-Greedy and a uniform A/B test, and explain why they behave differently.

The scenario is the one from the lab. A news platform has 6 candidate headlines for one home-screen slot. Each visitor sees one headline and either clicks (reward 1) or not (reward 0). The goal is to collect as many clicks as possible, which is the same as minimising **cumulative regret**: the expected clicks lost by not always showing the best headline.

The starter code already contains the environment, the A/B baseline, a working ε-Greedy agent, the simulation loop and the plotting code. ε-Greedy is your worked example: your agents must follow the same `select_arm()` / `update(arm, reward)` interface.

By the end you should be able to:

- turn the UCB1 and Thompson Sampling decision rules into working code
- read a regret curve and a traffic-share plot
- explain each algorithm's exploration behaviour in terms of its formula

## Setup

You receive two files. Put them in the same folder and install the requirements with `pip install numpy matplotlib`.

| File | What it is |
| --- | --- |
| `mab_assignment.py` | Starter code. Search for `TODO` to find every place you write code. |
| `check_my_work.py` | Self-check. Run `python check_my_work.py` until all 12 checks pass. |

1. Open `mab_assignment.py` and set `STUDENT_ID` at the top to your student ID. Your ID generates your own hidden click-through rates, so every student gets different numbers. Do not edit `make_true_ctr()`.
2. Implement Part A, then run `python check_my_work.py`. Each failed check prints a hint.
3. When all checks pass, run `python mab_assignment.py` for Parts B and C. A full run takes 1–2 minutes and saves the plots as PNG files.

The self-check tests your code against fixed CTRs, not your own. Passing it means your algorithms are correct; it does not grade your experiments or answers.

## Part A: Implement UCB1 and Thompson Sampling (40 points)

Fill in the two classes marked `TODO (Part A)`. `update()` for UCB1 is inherited from `BaseAgent`; both agents must return a plain Python `int` from `select_arm()`.

### UCB1 (20 points)

Show every headline once first. After that, show the headline with the highest upper confidence bound, where μ̂ᵢ is its observed CTR, Nᵢ is how often it was shown, t is the number of users so far and c is the exploration constant (`self.c`, default 2):

```latex
A_t = \arg\max_i \left[ \hat{\mu}_i + \sqrt{\frac{c \ln t}{N_i}} \right]
```

### Thompson Sampling (20 points)

Keep a Beta(αᵢ, βᵢ) belief for each headline's CTR, starting at Beta(1, 1). To choose, draw one sample from every belief with `self.rng.beta` and show the headline with the largest sample. After each user, add 1 to αᵢ on a click or to βᵢ on a non-click:

```latex
\theta_i \sim \mathrm{Beta}(\alpha_i, \beta_i), \qquad A_t = \arg\max_i \theta_i
```

In `update()`, call `super().update(arm, reward)` first so the shared counts and averages stay correct.

## Part B: Compare the four strategies (30 points)

Run `python mab_assignment.py`. Part B serves 20,000 users, repeats 20 times and prints each strategy's regret and the share of users shown the best headline. It saves `part_b.png` with the regret curves and the traffic split.

Include the table and the plot in your report, then answer in 3–5 sentences each. Refer to your own numbers and to the formulas.

1. Rank the four strategies by regret. Why does the winner win? (10 points)
2. Why does ε-Greedy's regret curve never become flat, even after it has found the best headline? (10 points)
3. UCB1 is a well-known algorithm, yet here it loses clearly to ε-Greedy. Explain why, using the size of its bonus term compared with the gaps between your CTRs. (10 points)

## Part C: One extension of your choice (20 points)

Pick **one** extension. The bottom of `mab_assignment.py` has a ready-made experiment for each; uncomment the one you choose. Report the table, the plot and a short explanation (half a page). The harder options earn the same points but give more to discuss.

| Option | Difficulty | New code | Question to answer |
| --- | --- | --- | --- |
| C1 Decaying ε | Easy | `DecayingEpsilonGreedy` with ε = min(1, 5/√t) | Does it beat fixed ε = 0.1? Does it catch up with Thompson Sampling? |
| C2 UCB1 constant | Easy | None | Compare c = 0.5, 2 and 5. How does c change exploration, and what is the risk of making c very small? |
| C3 Near-tie | Medium | None | Set the runner-up to 0.002 below the best. Why does the share on the best headline fall so much while regret barely rises? |
| C4 Breaking news | Hard | `DiscountedThompsonSampling` | At user 10,000 the best and worst headlines swap CTRs. Which agents adapt? Try γ = 0.99 and γ = 0.9995 and explain the difference. |

For C4, `BreakingNewsEnvironment` is already written. Your agent shrinks old evidence before every update so it can notice the change: α ← 1 + γ(α − 1) and β ← 1 + γ(β − 1). A sliding window over the last W users is also accepted.

## Part D: Recommendation (10 points)

The news platform's product manager asks you which algorithm to put in production for choosing the daily lead headline. Write at most one page for someone who is not a data scientist.

State your choice, back it with numbers from your experiments, and name at least one situation where your choice could go wrong (for example, a change in audience during the day).

## Submission and grading

Submit two files: your completed `mab_assignment.py` and a PDF report with your plots, tables and answers to Parts B–D. Deadline: \[date\].

| Part | Points | Full marks | Partial marks |
| --- | --- | --- | --- |
| A: UCB1 | 20 | All UCB1 self-checks pass; code is readable | Formula right but initial round or `self.c` missing |
| A: Thompson Sampling | 20 | All TS self-checks pass; code is readable | Beliefs update correctly but sampling or argmax is wrong |
| B: comparison | 30 | Table and plot included; each answer links the result to the formula, with own numbers | Correct but generic answers, no numbers |
| C: extension | 20 | Correct code, plot, and an explanation of *why* | Results shown without explanation |
| D: recommendation | 10 | Clear choice, backed by numbers, one realistic risk | Choice without evidence |
| **Total** | **100** |  |  |

The results must match your own `STUDENT_ID`. Discussing ideas with classmates is fine; code and written answers must be your own.
