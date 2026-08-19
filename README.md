
# Reward Prediction Error Prioritised Experience Replay (RPE-PER)

> **We learn more when reality differs from what we expected.**  
> RPE-PER brings this biological learning principle into experience replay.

**Reward Prediction Error Prioritised Experience Replay (RPE-PER)** is an experience replay strategy for off-policy reinforcement learning that prioritises transitions according to discrepancies between **predicted and observed rewards**.

Instead of relying solely on value-based errors to determine which experiences should be replayed, RPE-PER asks a more direct question:

> **How different was the received reward from what the agent expected?**

The method is implemented in **PyTorch**, integrated with **TD3** and **SAC**, and evaluated on continuous-control tasks from the [MuJoCo](https://www.gymlibrary.dev/environments/mujoco/index.html) benchmark suite.

**Paper:** [Reward Prediction Error Prioritisation in Experience Replay: The RPE-PER Method](https://arxiv.org/abs/2501.18093)  
**Presented at:** Australasian Conference on Robotics and Automation (**ACRA 2024**)

---

## Biological Motivation: Learning from Violated Expectations

Learning is strongly influenced by the difference between **what we expect** and **what actually happens**.

Consider visiting a familiar restaurant. Based on previous experiences, you already have an expectation of how rewarding the meal will be.

If the meal is exactly as expected, there may be little reason to substantially revise that expectation. In contrast, if the experience is **far better or far worse than expected**, the discrepancy provides a stronger signal for updating what has been learned and can have a greater influence on future decisions.

This discrepancy between **expected** and **received** reward is known as **Reward Prediction Error (RPE)**.

<p align="center">
  <img src="readme_media/RPE.png" width="850" alt="Reward Prediction Error in biological learning">
</p>

Conceptually, signed reward prediction error can be written as

$$
\delta_t^{(r)} = r_t - \hat{r}_t,
$$

where \(r_t\) is the reward actually received and \(\hat{r}_t\) is the expected reward.

The three principal cases are:

- **Positive RPE:** the outcome is better than expected.
- **Near-zero RPE:** the outcome closely matches expectation.
- **Negative RPE:** the outcome is worse than expected.

The central intuition is simple:

> **The greater the violation of expectation, the stronger the potential learning signal.**

RPE-PER transfers this idea to reinforcement learning by using reward-prediction discrepancies to determine **which stored experiences should be replayed more frequently**.

---

## From Reward Surprise to Experience Replay

Off-policy reinforcement-learning agents store past interactions in a replay buffer as transitions

$$
x_i = (s_i, a_i, r_i, s'_i),
$$

where:

- \(s_i\) is the current state,
- \(a_i\) is the selected action,
- \(r_i\) is the observed reward,
- \(s'_i\) is the resulting next state.

During training, a replay buffer can accumulate a large number of transitions. However, **not every stored experience is equally informative**.

Standard **Experience Replay** samples transitions uniformly. **Prioritised Experience Replay (PER)** instead increases the probability of replaying transitions with large temporal-difference (TD) errors.

TD error measures a discrepancy between a value estimate and its bootstrapped target. Although useful, it depends on value-function estimation and can be influenced by function approximation, bootstrapping, and changes in the learned policy.

RPE-PER takes a different perspective.

Instead of asking:

> **How wrong was the estimated value of this transition?**

RPE-PER asks:

> **How different was the observed reward from the reward predicted for this experience?**

This provides a direct and interpretable signal of how poorly the reward outcome is currently predicted.

---

## RPE-PER

For each stored transition

$$
x_i = (s_i, a_i, r_i, s'_i),
$$

the critic predicts the immediate reward associated with the state-action pair:

$$
\hat{r}_i = R_{\theta}(s_i,a_i),
$$

where \(R_{\theta}(s_i,a_i)\) is the predicted immediate reward.

RPE-PER then defines the reward prediction error used for prioritisation as

$$
\mathrm{RPE}_i
=
\left(
R_{\theta}(s_i,a_i)-r_i
\right)^2.
$$

This is the **squared discrepancy between predicted and observed reward**.

Unlike the signed biological RPE introduced above, RPE-PER uses the **magnitude of the reward-prediction mismatch** for replay prioritisation. Squaring the error makes the prioritisation signal non-negative and gives greater emphasis to larger discrepancies.

Therefore:

- an outcome that is **much better than expected** can produce a large RPE;
- an outcome that is **much worse than expected** can also produce a large RPE;
- an outcome that closely matches expectation produces a small RPE.

In short:

```text
Observed reward ≈ Predicted reward
              │
              ▼
          Small RPE
              │
              ▼
      Lower replay priority


Observed reward ≠ Predicted reward
              │
              ▼
          Large RPE
              │
              ▼
      Higher replay priority
              │
              ▼
       Replay more often
```

The principle behind RPE-PER is therefore:

> **Experiences whose rewards violate the agent's expectations more strongly receive greater replay priority.**

---

## Prioritised Sampling with RPE

For each transition, replay priority is defined as

$$
\sigma_i = \mathrm{RPE}_i + \epsilon,
$$

where \(\epsilon > 0\) ensures that all transitions retain a non-zero probability of being sampled.

The probability of sampling transition \(i\) is

$$
p_i
=
\frac{\sigma_i^\alpha}
{\sum_j \sigma_j^\alpha},
$$

where \(\alpha \geq 0\) controls the degree of prioritisation.

When \(\alpha = 0\), all transitions have equal sampling probability. Increasing \(\alpha\) places progressively greater emphasis on transitions with larger RPE values.

Because non-uniform sampling changes the training distribution, importance-sampling weights are applied:

$$
w_i
=
\left(
\frac{1}{N p_i}
\right)^\beta,
\qquad
\beta \in [0,1],
$$

where \(N\) is the replay-buffer size.

The overall idea can be summarised as

$$
\text{Reward prediction}
\;\longrightarrow\;
\text{Reward prediction error}
\;\longrightarrow\;
\text{Replay priority}
\;\longrightarrow\;
\text{Learning update}.
$$

---

## Enhanced Model Critic Network

To enable reward-based prioritisation, RPE-PER introduces an **Enhanced Model Critic Network (EMCN)**.

A conventional critic primarily estimates the action value associated with a state-action pair. EMCN extends this architecture by jointly predicting the action value, immediate reward, and next state.

For a state-action pair \((s,a)\),

$$
C_{\theta}(s,a)
=
\left(
Q_{\theta}(s,a),
R_{\theta}(s,a),
T_{\theta}(s,a)
\right),
$$

where:

- \(Q_{\theta}(s,a)\) estimates the **action value**;
- \(R_{\theta}(s,a)\) predicts the **immediate reward**;
- \(T_{\theta}(s,a)\) predicts the **next state** or its representation.

The reward-prediction component provides the quantity required to compute RPE:

$$
R_{\theta}(s_i,a_i)
\quad \text{vs.} \quad
r_i.
$$

The resulting discrepancy

$$
\left(
R_{\theta}(s_i,a_i)-r_i
\right)^2
$$

is then used to determine the replay priority.

Importantly, **predicted rewards do not replace observed rewards in value learning**.

The observed environment reward \(r_i\) is still used to construct the value target for TD3 or SAC. Reward prediction serves a separate purpose: identifying experiences that should receive greater attention during replay.

> **The RL objective determines how the agent learns; RPE determines which experiences it learns from more often.**

---

## Network Architecture

<p align="center">
  <img src="readme_media/RPE-PER.png" width="850" alt="RPE-PER architecture">
</p>

At a high level, the RPE-PER learning process is:

```text
             Environment Interaction
                      │
                      ▼
            Transition (s, a, r, s')
                      │
                      ▼
                 Replay Buffer
                      │
                      ▼
           Enhanced Model Critic
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
     Q-value        Reward       Next-state
     estimate      prediction    prediction
                      │
                      ▼
              Predicted reward
                      │
                      ▼
          Compare with observed reward
                      │
                      ▼
            Reward Prediction Error
                      │
                      ▼
                Replay Priority
                      │
                      ▼
            Prioritised Sampling
                      │
                      ▼
              TD3 / SAC Update
```

---

## Why Reward Prediction Error?

Traditional PER uses TD error as a proxy for identifying informative experiences. TD error combines the immediate reward with bootstrapped estimates of future value and therefore depends heavily on the quality and stability of value-function estimation.

RPE-PER isolates a simpler relationship:

$$
\text{Predicted reward}
\quad \longleftrightarrow \quad
\text{Observed reward}.
$$

This gives RPE-based prioritisation several useful properties:

- **Reward-grounded** — priority is directly related to the difference between predicted and observed task reward.
- **Interpretable** — a large RPE means that the observed reward was poorly predicted.
- **Independent of TD error for prioritisation** — replay priority does not rely directly on bootstrapped value discrepancies.
- **Biologically motivated** — the approach is inspired by the role of reward prediction error in biological learning and memory.
- **Simple to integrate** — the underlying TD3 and SAC learning objectives remain unchanged.
- **Applicable across actor-critic settings** — evaluated with both deterministic and stochastic off-policy algorithms.

---

## Supported Algorithms

RPE-PER is integrated with two off-policy continuous-control algorithms:

### TD3

**Twin Delayed Deep Deterministic Policy Gradient**

RPE-PER changes the replay prioritisation mechanism while retaining the underlying TD3 learning procedure.

### SAC

**Soft Actor-Critic**

RPE-PER is also integrated with SAC to evaluate reward-based prioritisation under a stochastic, entropy-regularised policy.

---

## Experiments

RPE-PER is evaluated on six continuous-control environments from the [MuJoCo](https://www.gymlibrary.dev/environments/mujoco/index.html) benchmark suite:

- `Ant-v4`
- `HalfCheetah-v4`
- `Hopper-v4`
- `Humanoid-v4`
- `Swimmer-v4`
- `Walker2d-v4`

The method is compared against several replay strategies:

- **Uniform Replay** — uniform random sampling
- **PER** — Prioritised Experience Replay
- **LAP** — Loss-Adjusted Prioritisation
- **LA3P** — Loss-Adjusted Approximate Actor Prioritised Experience Replay
- **MaPER** — Model-Augmented Prioritised Experience Replay

Experiments are conducted with both **TD3** and **SAC**, allowing RPE-based prioritisation to be evaluated under deterministic and stochastic policy-learning settings.

Across the evaluated tasks, RPE-PER demonstrates strong overall performance with both algorithms, with particularly consistent improvements in the TD3 experiments.

---

## Getting Started

### Prerequisites

| Library | Version |
|---|---:|
| `pydantic` | `1.10.10` |
| `MuJoCo` | `2.3.3` |

Clone the repository and install the required dependencies before running the experiments.

---

## Training

### TD3 + RPE-PER

```bash
python3 training_loop_TD3.py
```

### SAC + RPE-PER

```bash
python3 training_loop_SAC.py
```

---

## Core Idea

> ### **Replay the experiences that violate the agent's reward expectations.**

Rather than treating every stored transition equally, RPE-PER gives greater replay priority to experiences whose observed rewards differ substantially from what the agent predicted.

When the outcome is already well predicted, RPE is small.

When the outcome violates expectation, RPE is large.

**Those unexpected experiences are revisited more often during learning.**

---

## Paper

This repository accompanies our work:

**Reward Prediction Error Prioritisation in Experience Replay: The RPE-PER Method**  
Hoda Yamani, Yuning Xing, Lee Violet C. Ong, Bruce A. MacDonald, and Henry Williams

Presented at the **Australasian Conference on Robotics and Automation (ACRA 2024)**.

**Paper:** [arXiv:2501.18093](https://arxiv.org/abs/2501.18093)

The paper investigates **Reward Prediction Error (RPE)** as a biologically motivated signal for prioritising informative experiences in continuous-control reinforcement learning.

---

## Citation

If you use **RPE-PER** or this repository in your research, please cite:

```bibtex
@article{yamani2025reward,
  title={Reward prediction error prioritisation in experience replay: The RPE-PER method},
  author={Yamani, Hoda and Xing, Yuning and Ong, Lee Violet C and MacDonald, Bruce A and Williams, Henry},
  journal={arXiv preprint arXiv:2501.18093},
  year={2025}
}

