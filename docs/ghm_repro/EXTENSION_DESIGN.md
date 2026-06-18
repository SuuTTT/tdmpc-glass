# GHM Extension Design — turning InFOM's flow-occupancy scaffold into a CompPlan-style jumpy world model

Status: 2026-06-18. Scaffolding draft. **No paper numbers claimed.** Targets a faithful
reproduction of *Compositional Planning with Jumpy World Models* (CompPlan, arXiv 2602.19634),
built on *Temporal-Difference Flows* (TD-Flow, arXiv 2503.09817). No official code exists for either;
we adapt InFOM (`ghm_repro/infom`), the closest open JAX scaffold.

---

## 0. What InFOM already gives us (the verified map)

Files: `agents/infom.py`, `utils/networks.py`, `main.py`.

### 0.1 The velocity field `v_theta(x, t, conditioning)`

`utils/networks.py :: VectorField` (lines 261-317). It is a plain MLP. Its `__call__` signature is

```
__call__(self, noisy_goals, times, observations=None, actions=None, latents=None)
```

and it simply **concatenates** all provided tensors along the last axis and feeds them to
`vector_field_net` (an `MLP` whose final layer has width `vector_dim = obs_dim`):

```
times = times[..., None]
inputs = concat([noisy_goals, times, observations, actions, latents])   # any None dropped
vf = vector_field_net(inputs)                                            # shape (..., obs_dim)
```

So the conditioning is: **current state** `observations` (`s`), **action** `actions` (`a`), and the
**intention latent** `latents` (`z`). Flow time `t` is fed as a single extra scalar channel
(`times[..., None]`). There is **no horizon/discount input and no policy input** — those are the two
things CompPlan needs that InFOM lacks (plus the planner).

### 0.2 The flow-matching + TD-bootstrap occupancy loss (as implemented)

`agents/infom.py :: flow_occupancy_loss` (lines 111-186). InFOM learns a flow whose marginal at `t=1`
is the **discounted occupancy (successor) measure** `m^pi_gamma(. | s, a)` rather than the one-step
next state. It uses the **SARSA^2 / discounted-TD flow-matching** objective. Linear (rectified-flow)
interpolant with `x_0 = noise`, `x_1 = data goal`:

```
x_t = t * x_1 + (1 - t) * x_0          # infom.py:142, :160-161, :165-166
```

so the ground-truth conditional velocity is `dx_t/dt = x_1 - x_0`.

Two terms, mixed by the discount, exactly as in the code:

**(A) Current term** — match the velocity toward the *immediate* state `s` (the `(1-gamma)` mass that
the geometric horizon places on "stay here"):

```
current_noises  x0 ~ N(0, I)                                            # :139
current_vf_pred = v_theta( t*s + (1-t)*x0,  t,  sg(s), a, z )           # :141-146
current_fm_loss = || sg(s - x0) - current_vf_pred ||^2  (mean over dim) # :148-149
```
(`sg` = `jax.lax.stop_gradient`; the encoder/state target is stopped.)

**(B) Future / bootstrap term** — match the velocity toward a **target-network sample** of the
occupancy at `(s', a')`, i.e. the TD bootstrap (this is the TD-Flow piece):

```
future_noises x0' ~ N(0, I)                                            # :151
g'  = compute_fwd_flow_goals( x0', s', a', sg(z), use_target_network=True )   # :153-158
                                          # integrate the TARGET velocity field 0->1 (Euler, lax.scan)
future_vf_target = v_theta_TARGET( t*g' + (1-t)*x0', t, s', a', sg(z) )       # :159-163
future_vf_pred   = v_theta(        t*g' + (1-t)*x0', t, sg(s), a, sg(z) )     # :164-169
future_fm_loss   = || future_vf_target - future_vf_pred ||^2                  # :170
```

**Mixture (the exact loss):**

```
flow_matching_loss = mean[ (1 - gamma) * current_fm_loss  +  gamma * future_fm_loss ]    # :172-173
neg_elbo_loss      = flow_matching_loss + kl_weight * kl_loss                             # :176
```

where `kl_loss` (`:132-134`) is the standard VAE KL on the intention encoder
`q(z | s', a')` toward `N(0, I)`:
`kl = -0.5 * mean(1 + 2 log sigma - mu^2 - sigma^2)`.

This is the **geometric-horizon identity** for a *single fixed* `gamma`:
`m_gamma(.|s,a) = (1-gamma) delta_s + gamma * E_{s',a'}[ m_gamma(.|s',a') ]`.
The bootstrap target is the **target velocity field** integrated from noise to a sample, then
flow-matched in the online field — exactly TD-Flow's "flow-matching satisfies a Bellman equation".

### 0.3 Sampling, target network, downstream use

- `compute_fwd_flow_goals` (`:243-279`): Euler-integrate the (online or target) field from `x_0=noise`,
  `t: 0->1` in `num_flow_steps` (default 10) steps via `jax.lax.scan`; optional clip to obs bounds.
  Supports `init_times`/`end_times` (currently always 0->1) — **handy hook for horizon-truncated rollouts.**
- Target net: Polyak `target_update` (`tau=0.005`), `target_reset` copies online->target at finetune start.
- Conditioning supported today: `(s, a, z)` with `z` from `IntentionEncoder` (`critic_latent_type`
  `'encoding'`) or from the prior `N(0,I)` (`'prior'`).
- Downstream (`critic_loss`, `:45-109`): sample `num_flow_goals` future states, score with a learned
  reward head, form `Q = 1/(1-gamma) * mean_g reward(g)`; IQL-expectile-regress the critic.

---

## 1. The three things CompPlan needs that InFOM lacks

CompPlan's GHM is a **horizon- and policy-conditioned** flow model of the discounted future-state
measure, used to **plan over sequences of pre-trained goal-conditioned base policies** on OGBench.
InFOM has the generative core for ONE fixed `gamma` and ONE implicit behavior policy. We add:

### (a) HORIZON-conditioning

**Goal.** Instead of one fixed `gamma`, the GHM should sample the discounted-future-state measure for a
**chosen geometric horizon** `h` (equivalently a discount `gamma_h`). At plan time we want to ask the
model "where do I end up after roughly `h` steps under policy `pi`?" for several `h`.

**Parameterization.** Add a horizon embedding as an extra conditioning channel of the velocity field,
alongside `(noisy_goals, t, s, a, z)`:

- Represent the horizon by its discount `gamma_h in (gamma_min, gamma_max)` (geometric horizon
  `h ~ 1/(1-gamma_h)`). Feed a small **sinusoidal/Fourier embedding** of `gamma_h` (or of `log(1/(1-gamma_h))`)
  through a 1-layer MLP, then concatenate to the VectorField inputs. (Concretely: extend
  `VectorField.__call__` to accept `horizons` and append `horizon_embed(horizons)` to `inputs`.)
  A cheap first version simply concatenates the raw scalar `gamma_h[..., None]` (mirrors how `t` is fed);
  the Fourier embed is an upgrade if the scalar underfits.

**Training (per-batch horizon sampling + horizon-consistent bootstrap).** Sample one `gamma_h` per
example each minibatch (e.g. `gamma_h = 1 - exp(u)`, `u ~ U[log(1-gamma_max), log(1-gamma_min)]`, so
horizons are log-uniform). Then make the **TD recursion horizon-consistent**: the geometric-horizon
identity holds *for the same* `gamma_h` on both sides, so both the `(1-gamma)`/`gamma` mixture weights
and the bootstrap target field use the **same sampled `gamma_h`**:

```
x_t = t*x_1 + (1-t)*x_0
current_fm = || (s - x0)        - v(x_t^cur, t, s,  a,  z, gamma_h) ||^2
g'  = flow( x0', s', a', z, gamma_h, target=True )          # target field conditioned on SAME gamma_h
future_fm = || v_target(x_t^fut, t, s', a', z, gamma_h)
              - v(       x_t^fut, t, s,  a,  z, gamma_h) ||^2
loss = mean[ (1 - gamma_h) * current_fm + gamma_h * future_fm ]   # gamma_h now per-example, shape (B,)
```

This is the only change needed for the math to stay valid: `gamma` becomes a per-example vector
`gamma_h` and is also a network input. (TD-Flow / CompPlan call this the "geometric horizon model" —
one model that answers all horizons.) **Discount-consistency subset:** the paper applies the bootstrap
term to a fraction of the minibatch (25% antmaze / 12.5% cube); we expose `discount_consistency_frac`
and apply the future term as a Bernoulli mask over the batch (mass `(1-gamma_h)*cur + mask*gamma_h*fut`),
defaulting to 1.0 to recover plain InFOM.

### (b) POLICY-conditioning

**Goal.** Model the future-state measure **under a given base policy** `pi_k`, so the **same** GHM can
predict outcomes of different pre-trained policies (the planner mixes policies). InFOM instead conditions
on the *action* `a` and an intention latent inferred from `(s',a')` of the behavior data — it has no
explicit policy handle.

**Parameterization (two compatible hooks, choose at config):**
1. **Action-from-policy conditioning (default, simplest, faithful).** Keep the VectorField conditioned on
   `a`, but during training **relabel** the conditioning action with the policy's own action
   `a_pi = pi_k(s)` (and bootstrap with `a'_pi = pi_k(s')`) instead of the dataset action. With
   goal-conditioned base policies `pi_k = pi(.|s, g_k)`, the policy identity enters through the action
   it proposes. This re-uses the entire InFOM machinery unchanged except *which action is fed*.
2. **Explicit policy embedding (upgrade).** Replace/augment the intention latent `z` with a
   **policy embedding** `e_k` (a learned per-policy vector, or an embedding of the policy's goal `g_k`
   for goal-conditioned policies) and condition the field on `e_k`. Add `policy_embed` as an extra
   conditioning channel (same mechanism as the horizon embed). For a finite library of `K` pre-trained
   policies, `e_k` can be a `nn.Embed(K, d)`.

We implement hook (1) via a **policy-conditioning hook**: `flow_occupancy_loss` takes optional
`cond_actions` / `cond_next_actions`; if absent it falls back to dataset `actions`/`next_actions`
(= plain InFOM). At train time those are filled by a frozen base policy (or a mixture-of-policies
sampler) supplied by `main.py`. Hook (2) is left as a documented config switch (`policy_cond='embed'`)
with the embedding channel wired into the field but not exercised in the first smoke version.

**Training note.** To learn outcomes for *several* policies in one model, sample a policy `k ~ Unif(K)`
per example (or per minibatch), relabel `(a, a') <- (pi_k(s), pi_k(s'))`, and optionally append `e_k`.
The discount/horizon machinery from (a) is orthogonal and stacks on top.

### (c) PLAN-OVER-POLICIES (the planner)

**Goal.** Given start state `s_0` and goal `g`, search over **sequences of base policies**
`(pi_{k_1}, ..., pi_{k_L})`, using the GHM to *jump* to predicted future states and scoring proximity
to `g`. This is the CompPlan contribution: long-horizon composition by chaining jumpy predictions
instead of acting one primitive step at a time.

**Algorithm (beam / sampling over policy sequences):**

```
inputs: s0, goal g, policy library {pi_1..pi_K}, horizon discount gamma_h (jump length),
        plan depth L, beam width B, subgoal samples M, scorer d(s, g)
beam <- [ (s0, [], 0.0) ]                       # (current predicted state, policy-seq, score)
for step in 1..L:
    candidates <- []
    for (s, seq, sc) in beam:
        for k in 1..K:                          # try each base policy as next segment
            a_cond  <- pi_k(s)   (and z or e_k as configured)
            # GHM jump: sample M future states ~ m^{pi_k}_{gamma_h}( . | s, a_cond )
            x0   ~ N(0, I)  [M, obs_dim]
            s_next_samples <- compute_fwd_flow_goals(x0, repeat(s,M), repeat(a_cond,M),
                                                     z/e_k, horizons=gamma_h)   # M jumped states
            for s_next in s_next_samples:
                candidates.append( (s_next, seq+[k], sc + step_cost) )
    # score each candidate by predicted proximity to goal and keep top-B
    score(c) = d(c.s, g)                         # e.g. -||s_next - g|| in (encoded) obs space,
                                                 #     or the learned GHM reward / a goal classifier
    beam <- top_B candidates by score
return best policy-sequence in beam; execute pi_{k_1} until its sub-goal/horizon, then re-plan (MPC-style)
```

**Faithful settings (from the paper recipe in PLAN.md):**
- Subgoal samples `M`: **256 for antmaze, 1024 for cube** sequences sampled at plan time.
- Jump length: pick `gamma_h` so one jump ~ one policy-segment horizon (the horizon-conditioning from (a)
  lets us choose this without retraining).
- Scoring: distance-to-goal in (encoded) observation space, or the learned reward head `reward(s_next)`
  used as a goal-proximity proxy, or a goal-reaching classifier. Keep the **best** sequence; execute the
  first policy, then **re-plan** (receding horizon) — this is robust to GHM sampling error.
- Depth `L`: small (2-4); long horizons come from each jump covering many primitive steps.

**Where it lives.** A separate, eval-only module `planner.py` (no training) that calls the trained GHM's
`compute_fwd_flow_goals` and the frozen base policies. Not part of the train step, so it does not affect
the smoke test. OGBench eval (antmaze-medium-navigate, cube-single-play) drives it.

---

## 2. Mapping table (InFOM -> GHM)

| CompPlan need              | InFOM today                              | GHM change                                                                 |
|----------------------------|------------------------------------------|----------------------------------------------------------------------------|
| Horizon conditioning       | fixed scalar `config['discount']`        | per-example `gamma_h` sampled per batch + fed to VectorField; TD weights use `gamma_h` |
| Policy conditioning        | conditions on dataset `a`, latent `z` from `q(z\|s',a')` | relabel cond action with `pi_k(s)` (hook); optional policy embed `e_k` channel |
| Plan over policies         | single DDPG+BC actor, 1-step `sample_actions` | eval-only beam/sampling planner over `{pi_k}` sequences using GHM jumps     |
| Bootstrap / TD-Flow target | target-net occupancy sample, fixed gamma | same, but horizon-consistent (same `gamma_h` both sides) + consistency subset mask |

## 3. Implementation phasing

1. **(this commit)** `agents/ghm.py` = `infom.py` + horizon-conditioning (sampled per batch, fed to a
   horizon-aware VectorField) + a policy-conditioning hook (optional `cond_actions`). Runnable via
   `--agent=agents/ghm.py`; registered as `agent_name='ghm'`. Defaults reduce to InFOM behavior.
2. Wire a frozen OGBench base policy into `main.py` to fill `cond_actions` (policy conditioning live).
3. `planner.py` + OGBench eval hook; reproduce cube-1 lift, then antmaze-medium.
