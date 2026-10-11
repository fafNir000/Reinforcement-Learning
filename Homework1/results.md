# Results

## Overview

A news platform has 6 candidate headlines for one home-screen slot. Each visitor sees one headline and either clicks (reward 1) or not (reward 0). The goal is to collect as many clicks as possible, which is the same as minimising **cumulative regret**: the expected clicks lost by not always showing the best headline.

## Part A: Implement UCB1 and Thompson Sampling
### UCB1

Show every headline once first. After that, show the headline with the highest upper confidence bound, where μ̂ᵢ is its observed CTR, Nᵢ is how often it was shown, t is the number of users so far and c is the exploration constant (`self.c`, default 2):

$$A_t = \arg\max_i \left[ \hat{\mu}_i + \sqrt{\frac{c \ln t}{N_i}} \right]$$

```python
class UCB1(BaseAgent):
    def __init__(self, k, c=2.0):
        super().__init__(k)
        self.c = c
        self.name = "UCB1" if c == 2.0 else f"UCB1 (c={c})"
        # TODO (Part A): add any attributes you need (e.g. a counter self.t)
        self.t = 0

    def select_arm(self):
        # TODO (Part A): implement UCB1
        #  1. count this user (t)
        #  2. if some arm has never been shown, return it
        #  3. otherwise return the arm with the largest  mu_i + bonus_i
        ucb1 = np.zeros(self.k)
        self.t += 1
        for i in range(self.k):
            if self.counts[i] == 0:
                return i
            ucb1[i] = self.values[i] + np.sqrt(self.c * np.log(self.t) / self.counts[i])
        return np.argmax(ucb1)
```

### Thompson Sampling

Keep a Beta(αᵢ, βᵢ) belief for each headline's CTR, starting at Beta(1, 1). To choose, draw one sample from every belief with `self.rng.beta` and show the headline with the largest sample. After each user, add 1 to αᵢ on a click or to βᵢ on a non-click:

$$\theta_i \sim \mathrm{Beta}(\alpha_i, \beta_i), \qquad A_t = \arg\max_i \theta_i$$

```python
class ThompsonSampling(BaseAgent):
    name = "Thompson Sampling"
    def __init__(self, k, seed=None):
        super().__init__(k)
        self.rng = np.random.default_rng(seed)
        # TODO (Part A): create self.alpha and self.beta (all ones, length k)
        self.alpha = np.ones(k)
        self.beta = np.ones(k)

    def select_arm(self):
        # TODO (Part A): sample from each arm's Beta belief, return the argmax
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        # TODO (Part A): call super().update(arm, reward),
        #                then update alpha and beta for this arm
        super().update(arm, reward)
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
```

**`check_my_work.py` output**
```
Checking mab_assignment.py
============================================================
[PASS] Student ID is set
[PASS] UCB1: returns a valid int
[PASS] UCB1: shows every arm once first
[PASS] UCB1: bonus favours the least-shown arm
[PASS] UCB1: uses the constant c
[PASS] UCB1: beats the A/B test
[PASS] TS: starts with Beta(1, 1) beliefs
[PASS] TS: update() counts clicks and non-clicks
[PASS] TS: choices are random, not argmax of the mean
[PASS] TS: returns a valid int
[PASS] TS: finds the best headline
[PASS] TS: lower regret than UCB1 and the A/B test
============================================================
12/12 checks passed.
All checks passed. Now run mab_assignment.py for Parts B and C.
```
## Part B: Compare the four strategies

**`mab_assignment.py` part B output**

```
Student ID 36112 -> TRUE_CTR = [0.065, 0.088, 0.06, 0.044, 0.048, 0.025]
Best headline: H1

Part B: all strategies
Strategy                            Regret   % on best
------------------------------------------------------
A/B Test (uniform)                   660.0       16.7%
Epsilon-Greedy (eps=0.1)             121.9       81.4%
UCB1                                 453.6       36.5%
Thompson Sampling                     65.0       90.1%
Saved plot -> part_b.png
```

| Strategy | Regret | % on best |
|---|---:|---:|
| **Thompson Sampling** | **65.0** | **90.1%** |
| ε-Greedy | 121.9 | 81.4% |
| UCB1 | 453.6 | 36.5% |
| Uniform A/B | 660.0 | 16.7% |

![BResults](part_b.png)

The provided plots show the simulation results of different headline selection strategies (A/B Test, Epsilon-Greedy, UCB1, and Thompson Sampling) evaluated over 20,000 users.
### Left Plot: Cumulative regret (lower is better)
This plot tracks the number of missed clicks (Expected clicks lost) as the number of users increases. The lower the line, the better the algorithm performs.

* A/B Test (blue line): As expected, this is a perfectly straight line (linear growth). This happens because, throughout all 20,000 steps, the test continues to direct a fixed 16.6% of traffic to poorly performing headlines, consistently losing clicks with every new visitor.
* UCB1 (green line): Regret grows slower than the A/B test, but in this specific simulation, it underperformed compared to the other bandit algorithms. Due to its conservative and deterministic upper bound (especially when headline CTRs are close), it spends too much time repeatedly double-checking non-optimal variants.
* Epsilon-Greedy (orange line): Shows a strong performance. However, due to the fixed eps=0.1, it will always allocate 10% of traffic to random exploration until the end of time. As a result, its regret will continue to grow linearly forever, albeit at a very flat angle.
* Thompson Sampling (red line): The absolute winner. The curve quickly flattens out into a horizontal plateau. This indicates that the algorithm identified the best headline almost instantly and stopped making mistakes, keeping cumulative losses to an absolute minimum.

### Right Plot: Share of traffic per headline (initial CTRs)
This bar chart explains why the regret curves on the left look the way they do. It shows the percentage of total users assigned to each of the 6 headlines (H0 to H5). Their true CTRs are listed at the bottom. The best headline is H1 (CTR = 0.088).

* A/B Test (blue bars): Every single headline (from H0 to H5) receives exactly 16.6% of the traffic. Complete equality, completely ignoring the fact that H1 is the best and H5 is the worst. This explains its massive loss on the left plot.
* Thompson Sampling (red bars): Allocated around 90% of all traffic to the leading headline, H1. It spent only a tiny fraction of impressions on the other variants at the very beginning just to confirm their poor performance. This is why its regret is the lowest.
* Epsilon-Greedy (orange bars): Directed about 80% of traffic to the leader, H1. The remaining traffic is distributed as baseline impressions (exploitation of the leader + 10% random exploration spread evenly across all 6 headlines).
* UCB1 (green bars): Here we can clearly see why it lagged behind Thompson Sampling. It allocated only about 37% of traffic to H1. It spent way too many impressions (12–18% each) on the remaining headlines—especially H0 and H2, which have CTRs close to the leader—because its mathematical formulas took too long to decisively rule them out.

#### 1. Rank the four strategies by regret. Why does the winner win?

> The strategies ranked by cumulative regret from best to worst are Thompson Sampling (65.0), ε-Greedy (121.9), UCB1 (453.6), and the uniform A/B test (660.0). Thompson Sampling performs best because it quickly concentrates traffic on the headline with the highest CTR while still exploring uncertain headlines. In our experiment, it sent 90.1% of users to the best headline, compared with 81.4% for ε-Greedy. This efficient balance between exploration and exploitation resulted in the lowest regret.

#### 2. Why does ε-Greedy's regret curve never become flat?

> ε-Greedy continues to explore with probability ε = 0.1 even after it has identified the best headline. This means that approximately 10% of users are still sent to headlines chosen randomly instead of always using the best one. Those unnecessary displays create additional expected click loss, so cumulative regret continues to increase. In our experiment, ε-Greedy reached 81.4% traffic on the best headline but still accumulated 121.9 regret.

#### 3. Why does UCB1 lose clearly to ε-Greedy?

> UCB1 uses the exploration bonus $\sqrt{c\ln(t)/N_i}$, with $c=2$, and this bonus remains large compared with the gaps between our headline CTRs. The best headline has CTR 0.088, while the second-best has 0.065, so the gap is only 0.023. For example, when a headline has been shown 1,000 times at around \(t=20,000\), the exploration bonus is about 0.141, which is much larger than the 0.023 CTR gap. Therefore UCB1 continues exploring weaker headlines heavily, giving only 36.5% of traffic to the best headline and producing 453.6 regret.

## Part C

### Part C1 Decaying ε

**`mab_assignment.py` part C1 output**
```
C1: decaying epsilon
Strategy                            Regret   % on best
------------------------------------------------------
Epsilon-Greedy (eps=0.1)             121.9       81.4%
Decaying eps (5.0/sqrt t)             96.9       85.1%
Thompson Sampling                     65.0       90.1%
Saved plot -> part_c1.png
```
![C1Results](part_c1.png)

### Part C2 UCB1 constant

**`mab_assignment.py` part C2 output**
```
C2: UCB1 constant
Strategy                            Regret   % on best
------------------------------------------------------
UCB1 (c=0.5)                         275.2       59.1%
UCB1                                 453.6       36.5%
UCB1 (c=5.0)                         531.0       28.2%
Saved plot -> part_c2.png
```
![C2Results](part_c2.png)

### Part C3 Near-tie

**`mab_assignment.py` part C3 output**
```
C3: near-tie
Strategy                            Regret   % on best
------------------------------------------------------
Epsilon-Greedy (eps=0.1)              91.7       52.1%
UCB1                                 338.3       31.5%
Thompson Sampling                     68.3       50.5%
Saved plot -> part_c3.png
```
![C3Results](part_c3.png)

### Part C4 Breaking news

**`mab_assignment.py` part C4 output**
```
C4: breaking news at user 10,000
Strategy                            Regret   % on best
------------------------------------------------------
Epsilon-Greedy (eps=0.1)             511.1       44.0%
UCB1                                 485.0       34.5%
Thompson Sampling                    397.8       58.5%
Discounted TS (gamma=0.99)           594.4       21.8%
Discounted TS (gamma=0.9995)         315.5       55.8%
Saved plot -> part_c4.png
```
![C4Results](part_c4.png)

## Part D: Recommendation

