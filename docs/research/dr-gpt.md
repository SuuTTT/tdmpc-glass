Below is the English version, with references appended and claims slightly tightened.

# Survey: Abstraction Mechanisms for Improving TD-MPC2

## 0. Baseline: what TD-MPC2 already does

TD-MPC2 is already a strong latent world-model algorithm. It performs local trajectory optimization in the latent space of a learned decoder-free world model, and reports strong results across 104 online RL tasks with a single hyperparameter setting. It also scales to large multi-task agents. Therefore, any new “abstraction” module must add something that TD-MPC2 does **not** already get from its encoder, latent dynamics, reward/value heads, latent consistency loss, SimNorm-style normalization, and short-horizon MPPI planning. ([arXiv][1])

The key question is not “can abstraction help RL in general?” but rather: **what abstraction remains useful on top of TD-MPC2’s already task-oriented latent model?**

---

# B. Bisimulation / Behavioral State Abstraction

## DeepMDP

**Mechanism.** DeepMDP learns a continuous latent MDP by jointly predicting rewards and next latent-state distributions. Architecturally, it contains an encoder, latent transition model, and reward prediction head. The representation is trained so that latent dynamics and rewards preserve the original MDP’s behaviorally relevant structure. The paper explicitly connects this objective to bisimulation and argues that the learned latent space can be a valid abstraction of high-dimensional observations. ([arXiv][2])

**Clean win?** DeepMDP showed improvements as an auxiliary representation-learning objective on Atari and recovered latent structure in synthetic environments, but it did **not** provide a clean compute-matched win over modern strong model-based baselines such as Dreamer, TD-MPC, or TD-MPC2. Its main evidence predates those baselines. ([arXiv][2])

**Why it helps.** It forces the latent representation to preserve reward and transition information, rather than reconstructing every visual detail. This is useful when observations contain nuisance factors.

**Fit to TD-MPC2.** Largely redundant. TD-MPC2 already learns latent dynamics, reward/value prediction, and consistency objectives. DeepMDP’s main missing ingredient relative to TD-MPC2 is the more explicit theoretical link to bisimulation, not a fundamentally new architecture.

**Failure modes.** The objective can create tension between informative reward prediction and easy-to-predict transition compression. In practice, DeepMDP-style losses may collapse to locally convenient representations if the loss balance is poor.

---

## DBC: Deep Bisimulation for Control

**Mechanism.** DBC learns an encoder whose latent distances approximate bisimulation distances. Two states should be close in latent space if they have similar rewards and transition to similarly close future states. Unlike reconstruction-based world models, DBC intentionally discards task-irrelevant visual details. ([arXiv][3])

**Clean win?** DBC achieved strong results on visual MuJoCo / DMC tasks with distractor backgrounds and on a CARLA driving task. In the driving experiment, DBC’s final performance was reported as **46.8% better than the next best baseline**, compared against SAC, DeepMDP, reconstruction, and contrastive alternatives. However, this is **not** a clean win over Dreamer/TD-MPC/TD-MPC2. ([ar5iv][4])

**Why it helps.** DBC directly targets a common representation failure: the encoder may waste capacity modeling irrelevant predictable details such as moving backgrounds, lighting, clouds, or texture. Bisimulation tells the model to preserve only what affects rewards and future control.

**Fit to TD-MPC2.** Moderately useful, especially for pixel observations with distractors or domain shift. TD-MPC2 already has a task-oriented latent model, but it does not explicitly enforce behavioral equivalence between latent states. A DBC-style loss could make TD-MPC2’s encoder more robust to nuisance variation.

**Failure modes.** It requires careful loss balancing and assumptions about reward/transition distances. If the task has sparse or misleading rewards, bisimulation distances may be noisy or uninformative early in training.

---

## MICo

**Mechanism.** MICo introduces “Matching under Independent Couplings,” a sampling-based behavioral distance designed to be more scalable than classical bisimulation metrics. It shapes RL representations by encouraging states with similar long-term behavior to have similar representations. ([NeurIPS 会议录][5])

**Clean win?** MICo improved several deep RL agents on ALE and DM-Control and was evaluated with IQM curves over five independent seeds. However, the main comparisons were against value-based agents, SAC, and DBC variants, not against TD-MPC2 or Dreamer-style world models. Therefore, it is evidence that behavioral distances can help representation learning, but not decisive evidence that MICo beats modern MBRL. 

**Why it helps.** MICo reduces the computational and statistical difficulty of estimating bisimulation-like distances. It offers a more practical representation-shaping signal than exact Wasserstein-based bisimulation.

**Fit to TD-MPC2.** Potentially additive but uncertain. MICo could be used as an auxiliary latent regularizer, but TD-MPC2’s existing consistency loss already shapes latent dynamics. The open question is whether MICo gives a cleaner control-aware geometry than TD-MPC2’s reward/value/dynamics losses alone.

**Failure modes.** Behavioral similarity losses may underperform if their weighting is wrong, if rewards are sparse, or if the metric over-compresses states that the planner still needs to distinguish.

---

## BS-MPC: Bisimulation Metric for Model Predictive Control

**Mechanism.** BS-MPC is the closest existing result to your desired direction. It keeps the TD-MPC-style architecture—encoder, latent dynamics, reward model, value model, policy, and MPPI planning—but adds an explicit bisimulation metric loss to directly train the encoder. The paper states that BS-MPC shares the same five core components as TD-MPC and differs mainly by adding the bisimulation loss at every time step. ([ar5iv][6])

**Clean win?** This is the strongest evidence among the surveyed methods. BS-MPC compares directly with TD-MPC under the same architecture, hyperparameters, parameter count, and random seeds, with the only intended difference being the explicit encoder bisimulation loss. On 26 state-based DMControl tasks, the paper reports that BS-MPC consistently outperforms SAC, Dreamer-v3, and TD-MPC, especially on high-dimensional Dog and Humanoid tasks. On image-based DMC tasks, it runs 3M environment steps and is competitive with TD-MPC, DrQ-v2, and Dreamer-v3; under Kinetics-background distractions, it reports that BS-MPC outperforms TD-MPC in every tested environment. ([ar5iv][6])

**Important caveat.** The direct clean comparison is strongest against TD-MPC, not TD-MPC2. The paper includes an appendix comparison with TD-MPC2 and says BS-MPC is comparable or slightly better overall in some settings, but this is less decisive because TD-MPC2 changes model scale and architecture. ([ar5iv][6])

**Why it helps.** It addresses a specific TD-MPC weakness: the encoder is mostly trained indirectly through dynamics, reward, and value losses. BS-MPC adds a direct control-aware encoder objective, making the representation more robust and less sensitive to irrelevant noise.

**Fit to TD-MPC2.** Very high. This is the most natural mechanism to combine with TD-MPC2 because it modifies the representation objective without changing the planner or rollout structure. The main research question is whether the same idea remains beneficial after TD-MPC2’s stronger distributional value learning, SimNorm, and improved scaling.

**Failure modes.** BS-MPC requires tuning the bisimulation loss coefficient, and the paper explicitly notes that this parameter required grid search. It may also help most in noisy/high-dimensional settings and less on clean dense-control tasks. ([ar5iv][6])

---

# C. Temporal Abstraction / Hierarchy / Skills

## Options and Skills

**Mechanism.** Options convert sequences of primitive actions into temporally extended macro-actions or skills. A high-level policy chooses options; a low-level policy executes primitive actions for several steps.

**Clean win?** Classical options are conceptually important, but there is no standard, clean, compute-matched result showing vanilla options beating TD-MPC2/Dreamer/MuZero across modern benchmarks. Most strong evidence is task-specific, often with hand-designed skills, offline data, or specialized exploration settings.

**Why it helps.** Temporal abstraction shortens the effective planning horizon and improves long-horizon credit assignment.

**Fit to TD-MPC2.** Conceptually attractive but implementation-heavy. TD-MPC2 plans over a short latent horizon and uses a learned value function for long-term returns. Options could help when a 3–5 step MPPI horizon is too local, but they require skill discovery, skill-conditioned dynamics, and high-level planning.

**Failure modes.** Learned options often collapse into trivial behaviors, fail to align with task structure, or introduce non-stationarity between high- and low-level policies.

---

## Director

**Mechanism.** Director is a hierarchical world-model agent. It learns a world model from pixels, then trains a manager to select latent goals and a worker to reach those goals. The manager acts at a slower temporal scale, while the worker solves local goal-reaching. ([arXiv][7])

**Clean win?** Director gives strong evidence for temporal abstraction in sparse-reward, long-horizon visual tasks. It outperforms Dreamer and Plan2Explore on sparse 3D maze traversal with a quadruped robot, and the paper reports that Director is the only method to reliably find and reach goals in larger mazes. It also reports broad success on visual control, Atari, and DMLab. However, this is not a direct comparison against TD-MPC2. ([arXiv][7])

**Why it helps.** It fixes a failure of flat world-model agents: they struggle to discover and execute long behavior chains under sparse rewards. A latent goal hierarchy turns long-horizon exploration into a sequence of shorter goal-reaching problems.

**Fit to TD-MPC2.** Additive but expensive. It would require adding a manager, latent-goal representation, goal-conditioned worker, and likely a goal autoencoder. This is a true architecture-level abstraction, but much larger than adding an auxiliary abstraction loss.

**Failure modes.** It is most compelling in sparse long-horizon environments. On dense-control tasks, the hierarchy may add instability and overhead without improving performance.

---

## Jumpy / Compositional World Models

**Mechanism.** Jumpy world models predict multi-step outcomes of temporally extended policies, allowing planning over pre-trained policies rather than primitive actions. Compositional Planning with Jumpy World Models learns predictive models for policy-induced state occupancies across multiple time scales. ([arXiv][8])

**Clean win?** The 2026 CompPlan paper reports an average **200% relative improvement** over primitive-action planning on long-horizon manipulation and navigation tasks. However, the setting is offline / zero-shot compositional planning over pre-trained policies, not online TD-MPC2-style learning. ([arXiv][8])

**Why it helps.** Long-horizon prediction with primitive actions compounds model error. Planning over temporally extended policies reduces horizon length and allows the agent to compose reusable behavior chunks.

**Fit to TD-MPC2.** Promising but not minimal. It would require skill libraries or pre-trained policies, plus skill-conditioned latent transition models. This is more like a new hierarchical TD-MPC2 variant than a small abstraction module.

**Failure modes.** Strong dependence on the quality and coverage of pre-trained skills. If the skill library is poor, high-level planning cannot recover. It is also less suitable for pure online fair-compute comparisons.

---

## Hierarchical World Models: Negative Evidence

A 2024 study on hierarchical world models found that multi-level world models can support two-level decision-making, but did **not** outperform traditional methods in final episode returns. The authors identify model exploitation at the abstract level as a central failure mode. ([arXiv][9])

This is important for your project: hierarchy is intuitively aligned with abstraction, but reviewers will not accept “hierarchy should help” unless the benchmark specifically requires long-horizon abstraction and the implementation avoids abstract-model exploitation.

---

# D. Self-Predictive / Contrastive Representations

## SPR: Self-Predictive Representations

**Mechanism.** SPR trains an agent to predict its own future latent representations multiple steps ahead. It uses an exponential-moving-average target encoder and a learned transition model; data augmentation enforces consistency across different views of the same observation. ([arXiv][10])

**Clean win?** SPR achieved a median human-normalized score of **0.415** on Atari-100K, a **55% relative improvement** over the previous state of the art, and exceeded expert human scores on 7 of 26 games. However, this is not a clean comparison against Dreamer/TD-MPC/TD-MPC2. ([arXiv][10])

**Why it helps.** It improves sample efficiency from pixels by forcing representations to encode temporally predictive structure rather than only value-relevant information.

**Fit to TD-MPC2.** Partly redundant. TD-MPC2 already has latent dynamics and consistency losses. SPR’s specific contribution—EMA target prediction plus augmentation-consistent future latent prediction—may still help from pixels, but it is not as clearly orthogonal as bisimulation.

**Failure modes.** It can learn predictive but not necessarily control-relevant features. In environments with many predictable distractors, self-prediction may preserve nuisance information unless combined with task-aware filtering.

---

## BYOL-Explore

**Mechanism.** BYOL-Explore uses bootstrapped latent prediction as an intrinsic reward for exploration. It jointly learns world representation, dynamics, and exploration policy through a single latent prediction loss. ([arXiv][11])

**Clean win?** BYOL-Explore solved the majority of DM-HARD-8 tasks using its intrinsic reward, whereas prior methods reportedly needed human demonstrations. It also achieved superhuman performance on ten hard-exploration Atari games. But this is primarily an exploration method, not a direct architecture-level abstraction win over TD-MPC2. ([arXiv][11])

**Why it helps.** It addresses hard exploration by rewarding states where the model’s latent prediction is still difficult.

**Fit to TD-MPC2.** Useful only if the target benchmark is exploration-limited. For your HopperHop experience, this is relevant, but it would be an exploration bonus rather than a pure abstraction module. Since your stated goal rejects procedure/exploration tricks, BYOL-Explore is probably not the best match.

**Failure modes.** Intrinsic rewards can chase unpredictable noise or distractors. It may improve exploration while obscuring whether the abstraction module itself improves planning or sample efficiency.

---

# Decision Questions

## 1. Best single abstraction mechanism for TD-MPC2

The most defensible mechanism is **behavioral / bisimulation-style abstraction**, specifically an explicit encoder loss that makes latent distances reflect reward-and-transition equivalence.

Why: it is directly compatible with TD-MPC2, requires minimal architectural disruption, and targets a real gap in TD-MPC-style agents: the encoder is not explicitly trained to preserve behavioral equivalence. BS-MPC is the closest prior evidence because it modifies TD-MPC with an explicit bisimulation encoder objective while keeping the rest of the architecture nearly unchanged. ([ar5iv][6])

For your “Glass” idea, the strongest direction is therefore not generic clustering alone, but **behavior-aware clustering**: clusters should group latents that are equivalent under reward, value, and transition behavior. Pure prototype clustering or structural entropy may look elegant, but unless the clusters are control-aligned, reviewers will ask why TD-MPC2’s latent consistency does not already do the same job.

## 2. Standard fair evaluation protocol

The modern standard is to avoid single-seed or best-reward reporting. Use multiple seeds, full learning curves, aggregate metrics such as median and IQM, 95% confidence intervals, and performance profiles. The RLiable paper argues that point estimates are unreliable in few-run deep RL and recommends interval estimates, IQM, and performance profiles across benchmark suites. ([arXiv][12])

Recommended benchmarks:

| Goal                                             | Better benchmark                                      |
| ------------------------------------------------ | ----------------------------------------------------- |
| Representation abstraction / nuisance invariance | DMC pixels with distractor backgrounds                |
| Sample efficiency                                | DMC pixels, Atari-100K                                |
| Long-horizon abstraction                         | Crafter, sparse DMLab, AntMaze-style tasks            |
| Multi-task / transfer                            | Meta-World, multi-task DMC, task families             |
| TD-MPC2-specific comparison                      | DMC proprio + DMC pixels, using official TD-MPC2 code |

The best benchmark to isolate **abstraction value** is probably **DMC pixels with controlled distractors**, because it tests whether the abstraction discards behavior-irrelevant information while preserving control. HopperHop “best reward reached” is a poor abstraction benchmark because the signal is dominated by basin-entry luck rather than representation quality.

## 3. Cleanest minimal experiment

The cleanest experiment is:

**TD-MPC2 vs TD-MPC2 + Glass**, same codebase, same model size, same replay ratio, same environment steps, same seeds, same planner budget, same training schedule. The only difference should be the abstraction module/loss.

Use two benchmark groups:

1. **DMC pixels with distractors**: walker-walk, cheetah-run, quadruped-run, finger-spin, hopper-hop if desired, with fixed Kinetics or randomized video backgrounds.
2. **DMC proprio/state tasks**: include clean state-based tasks to show the module does not harm when nuisance abstraction is unnecessary.

Report:

* learning curves;
* IQM and median final return;
* 95% bootstrap CIs via RLiable;
* area under the learning curve for sample efficiency;
* ablations: TD-MPC2, TD-MPC2 + clustering only, TD-MPC2 + structural entropy only, TD-MPC2 + behavior-aware Glass.

A reviewer-believable win would be: **Glass improves IQM/AUC on distractor-pixel tasks with non-overlapping or clearly shifted 95% CIs, while not degrading clean proprioceptive DMC.** The most convincing diagnostic would show that clusters/prototypes align with reward/value/transition equivalence, not just visual similarity.

---

# References

1. Hansen, N., Su, H., & Wang, X. **TD-MPC2: Scalable, Robust World Models for Continuous Control.** ICLR 2024. arXiv:2310.16828. ([arXiv][1])
2. Gelada, C., Kumar, S., Buckman, J., Nachum, O., & Bellemare, M. G. **DeepMDP: Learning Continuous Latent Space Models for Representation Learning.** ICML 2019. arXiv:1906.02736. ([arXiv][2])
3. Zhang, A., McAllister, R., Calandra, R., Gal, Y., & Levine, S. **Learning Invariant Representations for Reinforcement Learning without Reconstruction.** ICLR 2021 oral. arXiv:2006.10742. ([arXiv][3])
4. Castro, P. S., Kastner, T., Panangaden, P., & Rowland, M. **MICo: Improved Representations via Sampling-Based State Similarity for Markov Decision Processes.** NeurIPS 2021. ([NeurIPS 会议录][5])
5. Shimizu, Y., & Tomizuka, M. **Bisimulation Metric for Model Predictive Control.** arXiv:2410.04553. ([arXiv][13])
6. Hafner, D., Lee, K.-H., Fischer, I., & Abbeel, P. **Deep Hierarchical Planning from Pixels.** arXiv:2206.04114. ([arXiv][7])
7. Farebrother, J., Pirotta, M., Tirinzoni, A., Bellemare, M. G., Lazaric, A., & Touati, A. **Compositional Planning with Jumpy World Models.** arXiv:2602.19634. ([arXiv][8])
8. Schiewer, R., Subramoney, A., & Wiskott, L. **Exploring the Limits of Hierarchical World Models in Reinforcement Learning.** arXiv:2406.00483. ([arXiv][9])
9. Schwarzer, M., Anand, A., Goel, R., Hjelm, R. D., Courville, A., & Bachman, P. **Data-Efficient Reinforcement Learning with Self-Predictive Representations.** arXiv:2007.05929. ([arXiv][10])
10. Guo, Z. D., Thakoor, S., Pîslar, M., et al. **BYOL-Explore: Exploration by Bootstrapped Prediction.** arXiv:2206.08332. ([arXiv][11])
11. Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. **Deep Reinforcement Learning at the Edge of the Statistical Precipice.** NeurIPS 2021 oral. arXiv:2108.13264. ([arXiv][12])

[1]: https://arxiv.org/abs/2310.16828 "[2310.16828] TD-MPC2: Scalable, Robust World Models for Continuous Control"
[2]: https://arxiv.org/abs/1906.02736 "[1906.02736] DeepMDP: Learning Continuous Latent Space Models for Representation Learning"
[3]: https://arxiv.org/abs/2006.10742 "[2006.10742] Learning Invariant Representations for Reinforcement Learning without Reconstruction"
[4]: https://ar5iv.org/pdf/2006.10742 "[2006.10742] Learning Invariant Representations for Reinforcement Learning without Reconstruction"
[5]: https://proceedings.neurips.cc/paper/2021/hash/fd06b8ea02fe5b1c2496fe1700e9d16c-Abstract.html "MICo: Improved representations via sampling-based state similarity for Markov decision processes"
[6]: https://ar5iv.org/html/2410.04553v1 "[2410.04553] Bisimulation metric for Model Predictive Control"
[7]: https://arxiv.org/abs/2206.04114 "[2206.04114] Deep Hierarchical Planning from Pixels"
[8]: https://arxiv.org/abs/2602.19634 "[2602.19634] Compositional Planning with Jumpy World Models"
[9]: https://arxiv.org/abs/2406.00483 "[2406.00483] Exploring the limits of Hierarchical World Models in Reinforcement Learning"
[10]: https://arxiv.org/abs/2007.05929 "[2007.05929] Data-Efficient Reinforcement Learning with Self-Predictive Representations"
[11]: https://arxiv.org/abs/2206.08332 "[2206.08332] BYOL-Explore: Exploration by Bootstrapped Prediction"
[12]: https://arxiv.org/abs/2108.13264 "[2108.13264] Deep Reinforcement Learning at the Edge of the Statistical Precipice"
[13]: https://arxiv.org/abs/2410.04553 "[2410.04553] Bisimulation metric for Model Predictive Control"
