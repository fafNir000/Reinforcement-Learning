# Task 1

## Contents
- [Part A: Implement UCB1 and Thompson Sampling](#part-a-implement-ucb1-and-thompson-sampling)
- [Part B: Compare the four strategies](#part-b-compare-the-four-strategies)
- [Part C4 Breaking news](#part-c4-breaking-news)
- [Part D: Recommendation](#part-d-recommendation)

## Part A: Implement UCB1 and Thompson Sampling
### UCB1

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

### `mab_assignment.py` part B output

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
This bar chart explains why the regret curves on the left look the way they do. It shows the percentage of total users assigned to each of the 6 headlines (H0 to H5). Their true CTRs are listed at the bottom. The best headline is H1.

* A/B Test (blue bars): Every single headline (from H0 to H5) receives exactly 16.7% of the traffic. Complete equality, completely ignoring the fact that H1 is the best and H5 is the worst. This explains its massive loss on the left plot.
* Thompson Sampling (red bars): Allocated around 90% of all traffic to the leading headline, H1. It spent only a tiny fraction of impressions on the other variants at the very beginning just to confirm their poor performance. This is why its regret is the lowest.
* Epsilon-Greedy (orange bars): Directed about 80% of traffic to the leader, H1. The remaining traffic is distributed as baseline impressions (exploitation of the leader + 10% random exploration spread evenly across all 6 headlines).
* UCB1 (green bars): Here we can clearly see why it lagged behind Thompson Sampling. It allocated only about 37% of traffic to H1. It spent way too many impressions (12–18% each) on the remaining headlines—especially H0 and H2, which have CTRs close to the leader—because its mathematical formulas took too long to decisively rule them out.

### 1. Rank the four strategies by regret. Why does the winner win?

> The strategies ranked by cumulative regret from best to worst are Thompson Sampling (65.0), ε-Greedy (121.9), UCB1 (453.6), and the uniform A/B test (660.0). Thompson Sampling performs best because it quickly concentrates traffic on the headline with the highest CTR while still exploring uncertain headlines. In our experiment, it sent 90.1% of users to the best headline, compared with 81.4% for ε-Greedy. This efficient balance between exploration and exploitation resulted in the lowest regret.

### 2. Why does ε-Greedy's regret curve never become flat?

> ε-Greedy continues to explore with probability ε = 0.1 even after it has identified the best headline. This means that approximately 10% of users are still sent to headlines chosen randomly instead of always using the best one. Those unnecessary displays create additional expected click loss, so cumulative regret continues to increase. In our experiment, ε-Greedy reached 81.4% traffic on the best headline but still accumulated 121.9 regret.

### 3. Why does UCB1 lose clearly to ε-Greedy?

> UCB1 uses the exploration bonus $\sqrt{c\ln(t)/N_i}$, with $c=2$, and this bonus remains large compared with the gaps between our headline CTRs. The best headline has CTR 0.088, while the second-best has 0.065, so the gap is only 0.023. For example, when a headline has been shown 1,000 times at around \(t=20,000\), the exploration bonus is about 0.141, which is much larger than the 0.023 CTR gap. Therefore UCB1 continues exploring weaker headlines heavily, giving only 36.5% of traffic to the best headline and producing 453.6 regret.

## Part C4 Breaking news

### `mab_assignment.py` part C4 output
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

This chart illustrates one of the most realistic and challenging scenarios for news media—a **non-stationary environment**, labeled here as *"breaking news at user 10,000"*.

In this simulation, exactly halfway through the test (at the 10,000th user), the true CTRs of the headlines changed drastically (for example, an old news story lost relevance, while another headline became a "hot breaking story").

Here is how the algorithms handled this abrupt shift:

### Left Chart: Cumulative Regret

* **Classic algorithms (Epsilon-Greedy, UCB1, Thompson Sampling):** Up to 10,000 users, they performed exceptionally well (especially Thompson Sampling and Epsilon-Greedy, whose regret barely grew). However, after the turning point, their curves sharply shot upward diagonally. Why? They accumulated too much "confidence" in the old leader and became prisoners of their historical data. It takes them a colossal amount of time to unlearn, during which they relentlessly waste clicks.
* **Discounted Thompson Sampling:** These modifications use a decay factor ($\gamma$) that forces the algorithm to "forget" older clicks and prioritize fresh data.
* **Discounted TS ($\gamma=0.9995$) (purple line):** The clear winner. Before the 10,000-step mark, its losses grew slightly faster than standard TS (due to artificially forgetting good historical data). But after the environment changed, it adapted faster than any other method; its curve leveled off, ultimately yielding the lowest cumulative regret (~310).
* **Discounted TS ($\gamma=0.99$) (red line):** An example of over-discounting. The algorithm forgets history too rapidly, suffering from a form of amnesia—it constantly re-evaluates options from scratch, accumulating high linear regret right from the beginning.

### Right Chart: Share of Traffic per Headline

This chart displays traffic allocation, and here an important nuance should be noted: the CTR values listed at the bottom (H1 = 0.088, H5 = 0.025) represent the initial probabilities that applied before user 10,000. Judging by algorithmic behavior, after the break, the worst headline (H5) became the new absolute leader!

* **Epsilon-Greedy (blue) and Standard TS (green):** Allocated over 60% of traffic to the former leader, H1. They simply didn't realize in time that the environment had changed, continuing to serve a headline no longer of interest to readers. They barely began giving attention to H5 (the new leader).
* **Discounted TS ($\gamma=0.9995$) (purple bars):** Demonstrated an ideal balance. It promptly reduced the share of the fading H1 (down to just 32%) and redirected the freed-up traffic to the new hit, H5 (nearly 30%). This allowed the platform to retain reader engagement right when the trend shifted.

### Key Takeaway from the "Breaking News" Scenario

In a real, rapidly changing environment (news feeds, stock markets, social media trends), classic bandit algorithms get "stuck" on historical data. For such tasks, it is vital to employ dynamic modifications (**Sliding Window** or **Discounted** approaches) that deliberately decay past memory, keeping the platform flexible and ready for shifting trends.

Here is the English translation, formatted clearly for precision and readability:

---

In the "Breaking News" scenario—where the best (H1) and worst (H5) headlines swap CTRs at user 10,000—bandit algorithms undergo a strict test of adaptability.

Here is a detailed breakdown of which agents can adapt and the critical difference between the two discount factors ($\gamma$):

### 1. Which Agents Adapt?

* **Classic Agents (Standard TS, UCB1, Epsilon-Greedy):** They practically **do not adapt** (or do so far too slowly). Over the first 10,000 steps, they accumulated massive "mathematical confidence" that H1 is the undisputed leader and H5 is a hopeless underperformer. When the swap occurs, standard Thompson Sampling has thousands of accumulated clicks in its $\alpha$ and $\beta$ counters for H1. For new unsuccessful impressions of H1 to outweigh that history, the algorithm requires tens of thousands of steps—during which the platform suffers colossal losses (reflected in the sharp upward surge in regret past the 10,000 mark).
* **Discounted Agents (Discounted Thompson Sampling):** They **adapt successfully**. By introducing a decay multiplier $\gamma$ that scales down the weight of past data at each step, these agents retain the ability to unlearn and re-evaluate. However, the speed and success of this adaptation critically depend on the choice of $\gamma$.

---

### 2. Comparing $\gamma = 0.99$ vs. $\gamma = 0.9995$

In Discounted Thompson Sampling, the update rule multiplies the accumulated success counters ($\alpha$) and failure counters ($\beta$) by the discount factor $\gamma$ (where $\gamma < 1$) at every step. This means the value of historical information decays exponentially.

#### $\gamma = 0.99$ (Excessively Rapid "Amnesia")

* **Behavior:** The agent has an extremely short memory. Data collected just 100–200 steps ago carries almost no weight.
* **Chart Result:** This agent adapts to role swaps instantaneously, but pays a heavy price. Throughout the entire simulation (both before and after 10,000 steps), its cumulative regret (red line) climbs steadily and linearly.
* **Why this happens:** Due to this "amnesia," the algorithm is never confident in its choices. If the new leader fails to get a click a couple of times in a row (a normal random occurrence at a CTR of ~8%), the agent immediately forgets that the headline is good and resumes testing everything indiscriminately. On the right-hand chart, the red bars are spread across all headlines—a clear hallmark of perpetual, chaotic exploration.

#### $\gamma = 0.9995$ (Ideal Balance / Optimal Adaptation)

* **Behavior:** The agent's memory fades much more slowly. It accumulates enough statistical evidence to confidently separate good headlines from bad ones, without getting permanently stuck on past data.
* **Chart Result:** This is the only agent whose curve flattens out **after** the shift at user 10,000 (purple line). Although its regret spiked briefly right after the swap (taking roughly 2,500 steps to adjust), the curve turned into a horizontal plateau once again. By step 20,000, it emerged as the clear winner with minimal total losses (~310 clicks).
* **Why this happens:** Up to 10,000 steps, it performed nearly as efficiently as standard TS. When the swap occurred, its historical "baggage" was light enough for it to adapt within a few thousand impressions. On the right chart, the purple bars show that it reacted in time: reducing impressions for the former leader H1 and allocating a large portion of traffic to the new favorite, H5.

---

### Key Takeaway

In a non-stationary environment, the parameter $\gamma$ controls the trade-off between **stability** and **responsiveness**:

* If $\gamma$ is **too small ($0.99$)**, the algorithm becomes a memoryless, overly reactive agent that constantly shifts choices and incurs heavy losses from chaotic exploration.
* If $\gamma$ equals **1 (classic bandit)**, the algorithm becomes a rigid conservative, incapable of responding to changing trends.
* A value of **$\gamma = 0.9995$** serves as the **"Goldilocks zone"** (optimal setting) for this task—allowing the algorithm to build reliable confidence while remaining flexible when environmental dynamics shift.

## Part D: Recommendation

> I would recommend Thompson Sampling for production because it achieved the best overall performance in our experiments. It had the lowest cumulative regret at 65.0, meaning that it lost the fewest expected clicks compared with always choosing the best headline. It also sent 90.1% of users to the best headline, compared with 81.4% for ε-Greedy, 36.5% for UCB1, and 16.7% for the uniform A/B test. This shows that Thompson Sampling was able to balance exploration and exploitation effectively: it continued to explore uncertain headlines while quickly concentrating traffic on the strongest one.
However, this choice could fail if the audience or headline performance changes over time. For example, if a different headline suddenly becomes more attractive because of breaking news or a change in user interests, an algorithm based on historical observations may continue favouring the previously successful headline. In such a situation, the system would need a mechanism for adapting to changing click-through rates.

### Performance Proof (Simulation Results)

In our simulation across 6 candidate headlines over 20,000 visitors, Thompson Sampling significantly outperformed all alternative strategies:

* **Lowest Regret:** Thompson Sampling incurred a cumulative regret (missed clicks) of only **65.0**, vastly outperforming Epsilon-Greedy (121.9), UCB1 (453.6), and a standard uniform A/B test (660.0).

* **Maximum Efficiency:** It directed **90.1%** of total user traffic to the best-performing headline, compared with 81.4% for Epsilon-Greedy and just 16.7% for a traditional A/B test.

---

### Where This Choice Could Go Wrong: Breaking News & Audience Shifts

While Thompson Sampling excels under normal, stable conditions, **it can fail when a sudden breaking news event or rapid audience shift occurs mid-day**.

* If a major story breaks and an old, poorly performing headline suddenly becomes the most engaging, standard Thompson Sampling will struggle to adapt quickly.

* Because it accumulates heavy historical confidence in the previous winning headline, it requires thousands of new impressions to unlearn its past data and recognize the trend change.

### Production Summary

* Use standard **Thompson Sampling** for standard daily operations where reader interests are relatively steady.

* If our platform frequently encounters sudden breaking news stories or rapid audience shifts throughout the day, deploy **Discounted Thompson Sampling ($\gamma = 0.9995$)**, which systematically discounts past data to maintain flexibility and adapt quickly to shifting trends.
