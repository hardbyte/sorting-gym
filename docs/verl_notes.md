# veRL onboarding notes (for sorting-gym)

Distilled from the veRL docs, 2026-09. Marked **[docs]** for what the documentation
actually states and **[inferred]** for my own reading. "not documented" means exactly that.

## 1. Minimal GRPO job

Install **[docs, [quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html))]**:

```bash
pip install verl          # Docker images are the recommended path
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

Dataset format: **parquet** files, one row per prompt, holding "the necessary fields
for computing RL rewards". The preprocessing script writes the chat-style `prompt`
plus `data_source`, `reward_model` (containing `ground_truth`), and `extra_info`
columns — these are the names the reward hook receives back (§2). Point the trainer
at them with `data.train_files` / `data.val_files`.

Run:

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=512 data.max_response_length=512 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.total_epochs=15
```

Fields that matter, per [algo/grpo](https://verl.readthedocs.io/en/latest/algo/grpo.html):
`adv_estimator=grpo` (no critic is trained — drop every `critic.*` key), `rollout.n > 1`
(the group size; this is what GRPO's baseline is computed over), `use_kl_loss=True`,
`kl_loss_coef` (default 0.001), `kl_loss_type` ∈ `kl(k1) | abs | mse(k2) | low_var_kl(k3)`,
`norm_adv_by_std_in_grpo=False` for Dr.GRPO. Three model roles share one `model.path`:
`actor` (trained), `rollout` (vLLM/SGLang inference), `ref` (frozen KL reference).
Effective rollout cost is `train_batch_size × rollout.n` sequences per step.
Example script: `examples/grpo_trainer/run_qwen3_8b_fsdp.sh`.

## 2. Custom (programmatic) reward

Exactly our case — verifiable reward, no reward model.
[preparation/reward_function](https://verl.readthedocs.io/en/latest/preparation/reward_function.html):

```python
# my_reward.py
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return float(...)
```

```bash
custom_reward_function.path=/abs/path/my_reward.py \
custom_reward_function.name=compute_score   # omittable if named compute_score
```

`solution_str` is the **detokenized response text**; `data_source` and `ground_truth`
come from `non_tensor_batch`. **[inferred]** For sorting-gym this means: stash the env
seed / initial array in `extra_info`, re-instantiate the env inside `compute_score`,
replay the parsed instructions with `sorting_gym/text/`'s parser, and return the summed
negative cost (plus a shaping/format penalty for unparseable output). This is a
single-shot scoring hook — it sees the whole completion, not per-step. Per-step
interaction needs §3.

## 3. Multi-turn / agent loop

Two overlapping mechanisms:

**AgentLoop** ([advance/agent_loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)) —
the general one. `AgentLoopManager` samples batches, wakes inference servers and hands
work to `AgentLoopWorker`s. You subclass `AgentLoopBase`; **"the `run` method is the only
interface that user need to implement"**:

```python
async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput
```

Inside `run` you call `LLMServerClient.generate()` and may "call tools: web search,
database query, code sandbox" — i.e. anything, including stepping a Gymnasium env.
Returns `AgentLoopOutput(prompt_ids, response_ids, response_mask)`. Enabled with
`actor_rollout_ref.rollout.mode=async` plus an agent-loop config path.

**Multi-turn tool rollout** ([sglang_multiturn](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)):
`actor_rollout_ref.rollout.multi_turn.enable=True`, `.format=hermes`,
`rollout.name=sglang`, and either `multi_turn.tool_config_path` (classes extending
`BaseTool` with `create`/`release` lifecycle) or `function_tool_path` (`@function_tool`
decorator, stateless, schema inferred).

Stability: **[docs]** warns "future models' chat template could break compatibility" and
recommends `tokenization_sanity_check_mode`. **[inferred]** AgentLoop is the newer,
less-settled surface; the SGLang tool path is the better-trodden one. Treat both as
moving targets and pin a veRL commit.

## 4. Token masking

`response_ids` is "Response token ids including LLM generated token, tool response
token"; `response_mask` is "1 for LLM generated token, 0 for tool response token"
**[docs]**. So env-injected observations sit in the sequence (the model conditions on
them) but contribute no loss.

The tool path builds this mask by **delta tokenization**: tokenize messages up to turn
*i* with `add_generation_prompt=True`, then through turn *i+1* with
`add_generation_prompt=False`, and mask only the delta — "only tokens generated by the
assistant are included in the loss mask." What you must get right: the same chat
template and special tokens at rollout and train time (hence the sanity check), and, if
you write your own `AgentLoop`, constructing `response_mask` yourself — an off-by-one
there trains the model on its own observations.

## 5. Feasibility on one 12GB RTX 3080 Ti

Blunt: **the quickstart states "at least 24GB" GPU memory** for Qwen2.5-0.5B PPO.
Single-GPU is supported (`trainer.n_gpus_per_node=1`), but 12GB is under the documented
floor. The GPU must simultaneously hold the actor (+Adam states, ~16 bytes/param fp32
moments), the frozen ref model, and the vLLM KV cache, which takes
`gpu_memory_utilization` (0.4) of the card up front.

- **0.5B full-finetune**: borderline-to-OOM. Only plausible with GRPO (no critic —
  that alone removes a whole model), `ppo_micro_batch_size_per_gpu=1`,
  `gpu_memory_utilization≈0.3`, short sequences (256/256), FSDP CPU offload for
  optimizer and params, `rollout.n=4`. **[inferred]** — not documented as a supported config.
- **1.5B+ full-finetune**: will OOM.
- **LoRA** ([advance/ppo_lora](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)):
  `model.lora_rank=32`, `lora_alpha=32`, `target_modules=all-linear`, `use_shm=True`,
  `rollout.load_format=safetensors` (required), and `rollout.layered_summon=True`
  ("recommended if … GPU memory is limited (<48GB)"). Docs frame LoRA as 70B on 8×80G,
  not as a 12GB enabler. LoRA removes optimizer state but not the base weights, KV
  cache, or the ref model — **[inferred]** 1.5B LoRA is the realistic ceiling, maybe.
- Smallest sensible experiment: Qwen2.5-0.5B-Instruct, GRPO, `rollout.n=4`,
  `train_batch_size=32`, 256/256 tokens, arrays of length 4-6, single-turn (§2 reward
  replaying a whole emitted program) — *not* the agent loop. Get that to run before
  touching multi-turn.

Cheapest cloud that removes the constraint: one A100-40G or H100-80G spot instance
(RunPod / Lambda / Vast), roughly $1-2/hr; a 24GB 4090/L4 is enough for the 0.5B
single-turn experiment and is ~$0.35-0.60/hr. **[inferred]** — pricing is not from the docs.

## 6. What we would have to build

1. `examples/verl/make_dataset.py` — emit parquet rows: `prompt` (chat messages from
   the existing text renderer), `data_source="sorting_gym"`, `reward_model.ground_truth`
   (optimal/reference cost), `extra_info` (env id, seed, array length).
2. `examples/verl/reward.py::compute_score` — reconstruct the env from `extra_info`,
   parse `solution_str` with `sorting_gym/text/`'s action parser, step to termination,
   return summed reward; penalise parse failures and non-termination distinctly.
3. Determinism: the env must be exactly reproducible from `(id, seed)` — the reward is
   computed in a separate worker process from the rollout.
4. Shared prompt/format contract between the renderer and `examples/llm_synthesis.py`
   so the frozen-LLM baseline and the RL run are measured on the same interface.
5. Only then: a `SortingAgentLoop(AgentLoopBase)` for true per-step interaction, owning
   `response_mask` (observations → 0) and a turn cap.
6. Pin the veRL commit; run `tokenization_sanity_check_mode` on the first job.
