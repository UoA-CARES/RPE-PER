# Reward Prediction Error Prioritised Experience Replay (RPE-PER)

[![arXiv](https://img.shields.io/badge/arXiv-2501.18093-b31b1b.svg)](https://arxiv.org/abs/2501.18093)
[![ACRA 2024](https://img.shields.io/badge/ACRA-2024-6A5ACD)](https://www.araa.asn.au/conference/acra-2024/)
[![Proceedings](https://img.shields.io/badge/ACRA-Proceedings-4B0082)](https://ssl.linklings.net/conferences/acra/acra2024_proceedings/views/includes/files/pap119s2.pdf)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TD3](https://img.shields.io/badge/Algorithm-TD3-007ACC)](https://arxiv.org/abs/1802.09477)
[![SAC](https://img.shields.io/badge/Algorithm-SAC-007ACC)](https://arxiv.org/abs/1801.01290)
[![MuJoCo](https://img.shields.io/badge/Benchmark-MuJoCo-00599C)](https://mujoco.org/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)

> **Unexpected outcomes drive learning. RPE-PER uses this principle to decide which experiences should be replayed more often.**

**Reward Prediction Error Prioritised Experience Replay (RPE-PER)** is a biologically motivated experience replay strategy for off-policy reinforcement learning. It prioritises transitions according to the discrepancy between **predicted and observed rewards**, allowing the agent to revisit experiences whose outcomes are not yet well predicted.

RPE-PER is implemented in **PyTorch**, integrated with **TD3** and **SAC**, and evaluated on continuous-control tasks from the **MuJoCo** benchmark suite.

**Paper:** [Reward Prediction Error Prioritisation in Experience Replay: The RPE-PER Method](https://arxiv.org/abs/2501.18093)  
**Proceedings:** [ACRA 2024 Proceedings Paper](https://ssl.linklings.net/conferences/acra/acra2024_proceedings/views/includes/files/pap119s2.pdf)  
**Presented at:** Australasian Conference on Robotics and Automation (**ACRA 2024**)

---

## Biological Motivation

Learning is strongly influenced by the difference between **what we expect** and **what actually happens**.

Consider visiting a familiar restaurant. Based on previous experiences, you already have an expectation of how rewarding the meal will be. If the experience is exactly as expected, there may be little reason to substantially revise that expectation. But if the outcome is **far better or far worse than expected**, the discrepancy can produce a stronger learning signal and have a greater influence on future decisions.

This difference between **expected reward** and **received reward** is known as **Reward Prediction Error (RPE)**.

<p align="center">
  <img src="readme_media/RPE.png" width="850" alt="Reward Prediction Error in biological learning">
</p>

Conceptually, the signed reward prediction error can be written as

```math
\delta_t^{(r)} = r_t - \hat{r}_t
```

where $r_t$ is the reward actually received and $\hat{r}_t$ is the expected reward.

- **Positive RPE:** the outcome is better than expected.
- **Near-zero RPE:** the outcome matches expectation.
- **Negative RPE:** the outcome is worse than expected.

The key intuition is:

> **The greater the violation of expectation, the stronger the potential learning signal.**

RPE-PER transfers this principle to experience replay by prioritising transitions whose rewards differ most from what the agent predicted.

---

## From Reward Surprise to Experience Replay

Off-policy reinforcement-learning agents store interactions in a replay buffer as transitions

```math
x_i = (s_i, a_i, r_i, s'_i)
```

where $s_i$ is the current state, $a_i$ is the selected action, $r_i$ is the observed reward, and $s'_i$ is the resulting next state.

Standard **Experience Replay** samples stored transitions uniformly. However, not every experience is equally informative.

**Prioritised Experience Replay (PER)** addresses this by increasing the probability of replaying transitions with large temporal-difference (TD) errors.

TD error measures a discrepancy between a value estimate and its bootstrapped target. Although useful, it depends on value-function estimation and can be influenced by bootstrapping, function approximation, and changes in the learned policy.

RPE-PER takes a different perspective and focuses directly on the observed reward outcome:

> **How different was the received reward from the reward the agent predicted?**

This provides a direct and interpretable signal for identifying experiences whose outcomes are still poorly predicted.

---

## RPE-PER

For each stored transition, the critic predicts the immediate reward associated with the state-action pair:

```math
\hat{r}_i = R_{\theta}(s_i,a_i)
```

where $R_{\theta}(s_i,a_i)$ is the predicted immediate reward.

RPE-PER defines the reward prediction error used for prioritisation as

```math
\mathrm{RPE}_i =
\left(
R_{\theta}(s_i,a_i) - r_i
\right)^2
```

This is the **squared discrepancy between predicted and observed reward**.

Unlike the signed biological RPE introduced above, RPE-PER uses the **magnitude of the reward-prediction mismatch** for replay prioritisation. Squaring the discrepancy makes the prioritisation signal non-negative and gives greater emphasis to larger prediction errors.

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

> **Experiences whose rewards violate the agent's expectations more strongly receive greater replay priority.**

---

## Prioritised Sampling

For each transition, replay priority is defined as

```math
\sigma_i = \mathrm{RPE}_i + \epsilon
```

where $\epsilon > 0$ ensures that every transition remains sampleable.

The probability of sampling transition $i$ is

```math
p_i =
\frac{\sigma_i^{\alpha}}
{\sum_j \sigma_j^{\alpha}}
```

where $\alpha \geq 0$ controls the strength of prioritisation.

- $\alpha = 0$ corresponds to uniform sampling.
- Larger $\alpha$ places greater emphasis on transitions with high RPE.

To compensate for the bias introduced by non-uniform sampling, importance-sampling weights are applied:

```math
w_i =
\left(
\frac{1}{N p_i}
\right)^{\beta},
\qquad
\beta \in [0,1]
```

where $N$ is the replay-buffer size.

The replay process can therefore be summarised as:

**Reward prediction → Reward prediction error → Replay priority → Prioritised sampling → Learning update**

---

## Enhanced Model Critic Network

To enable reward-based prioritisation, RPE-PER introduces an **Enhanced Model Critic Network (EMCN)**.

A conventional critic primarily estimates the action value associated with a state-action pair. EMCN extends this architecture by jointly predicting the action value, immediate reward, and next state.

For a state-action pair $(s,a)$, the EMCN outputs

```math
C_{\theta}(s,a)
=
\left(
Q_{\theta}(s,a),
R_{\theta}(s,a),
T_{\theta}(s,a)
\right)
```

where:

- $Q_{\theta}(s,a)$ estimates the **action value**;
- $R_{\theta}(s,a)$ predicts the **immediate reward**;
- $T_{\theta}(s,a)$ predicts the **next state** or its representation.

The reward-prediction component provides the expected reward required to compute RPE. For transition $i$, the predicted reward $R_{\theta}(s_i,a_i)$ is compared with the observed reward $r_i$:

```math
\mathrm{RPE}_i =
\left(
R_{\theta}(s_i,a_i) - r_i
\right)^2
```

This RPE value is then used to determine replay priority.

Importantly, **predicted rewards do not replace observed rewards in value learning**. TD3 and SAC continue to use the actual environment reward $r_i$ when constructing their value targets.

Reward prediction serves a separate purpose: identifying which stored experiences should receive greater attention during replay.

> **The underlying RL algorithm determines how the agent learns; RPE-PER determines which experiences are replayed more often.**

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

## Why RPE-Based Prioritisation?

RPE-PER is built around a simple idea: **not all experiences deserve equal replay attention**.

Compared with conventional TD-error-based prioritisation, RPE provides several useful properties:

- **Reward-grounded** — priority is directly related to the discrepancy between predicted and observed reward.
- **Interpretable** — a large RPE means that the observed outcome was poorly predicted.
- **Independent of TD error for prioritisation** — replay priority does not directly rely on bootstrapped value discrepancies.
- **Biologically motivated** — inspired by the role of reward prediction error in learning and memory.
- **Simple to integrate** — the underlying TD3 and SAC objectives remain unchanged.
- **Applicable across actor-critic settings** — evaluated with both deterministic and stochastic off-policy algorithms.

---

## Supported Algorithms

RPE-PER is integrated with two off-policy continuous-control algorithms:

- **TD3** — Twin Delayed Deep Deterministic Policy Gradient
- **SAC** — Soft Actor-Critic

This allows RPE-based replay to be evaluated under both deterministic and stochastic policy-learning settings.

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

- **Uniform Replay**
- **PER** — Prioritised Experience Replay
- **LAP** — Loss-Adjusted Prioritisation
- **LA3P** — Loss-Adjusted Approximate Actor Prioritised Experience Replay
- **MaPER** — Model-Augmented Prioritised Experience Replay

Experiments are conducted with both **TD3** and **SAC** to evaluate RPE-based prioritisation across deterministic and stochastic policy-learning settings.

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

> **Replay the experiences that violate the agent's reward expectations.**

If an outcome is already well predicted, its RPE is small and it receives lower replay priority.

If the observed reward differs substantially from what the agent expected, its RPE is large and the experience is replayed more frequently.

---


## Paper

**Reward Prediction Error Prioritisation in Experience Replay: The RPE-PER Method**
Hoda Yamani, Yuning Xing, Lee Violet C. Ong, Bruce A. MacDonald, and Henry Williams

Published in the **The Australasian Conference on Robotics and Automation (ACRA 2024)**, Auckland, New Zealand.

**Paper:** [ACRA 2024 Proceedings](https://ssl.linklings.net/conferences/acra/acra2024_proceedings/views/includes/files/pap119s2.pdf)
**arXiv:** [arXiv:2501.18093](https://arxiv.org/abs/2501.18093)

---

## Citation

If you use **RPE-PER** or this repository in your research, please cite:

```bibtex
@inproceedings{yamani2024reward,
  title     = {Reward Prediction Error Prioritisation in Experience Replay: The RPE-PER Method},
  author    = {Yamani, Hoda and Xing, Yuning and Ong, Lee Violet C. and MacDonald, Bruce A. and Williams, Henry},
  booktitle = {Proceedings of the Australasian Conference on Robotics and Automation (ACRA 2024)},
  pages     = {154--163},
  year      = {2024},
  address   = {Auckland, New Zealand},
  publisher = {Australian Robotics and Automation Association}
}
```

```
