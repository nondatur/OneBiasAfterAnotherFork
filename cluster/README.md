# Running on the hessian.AI 42 cluster

Workspace **`IL_rm_bias`** (id 142) · PFSS root **`/pfss/mlde/workspaces/mlde_wsp_IL_rm_bias`**

Our workload is **inference only**: forward passes, last-layer activation extraction, and
linear algebra. No training, no checkpointing, no hyperparameter search — so none of
Determined's trial machinery is needed. A task with a plain `entrypoint` just runs our CLI.

---

## 0. Connect

```bash
source .venv-mlx/bin/activate                 # `det` 0.35.0 lives here
export DET_MASTER=https://login01.ai.tu-darmstadt.de:8080
det user login <TU_ID>                        # TU VPN must be up first
det workspace ls
```

Fill the two placeholders in `cluster/config.yaml` (`<TU_ID>`, `<REGISTRY>`) before launching
anything.

## 1. Build and push the image

A custom image is **mandatory**. The stock Determined images are `py-3.8-pytorch-1.12`; our
reward models need torch 2.x and a modern transformers, and the compute nodes forbid
installing anything at run time.

The cluster is amd64 and this Mac is arm64, so build on an amd64 machine or a CI runner.
Cross-building locally works but is slow:

```bash
docker buildx build --platform linux/amd64 \
  -f cluster/Dockerfile -t <REGISTRY>/rm-bias:latest --push .
```

The registry must be **public**, or the cluster cannot pull without credentials.

## 2. Stage code, data and checkpoints

Everything lives on PFSS, never on the instance — instance storage is wiped when a task ends,
and overrunning it crashes the whole compute node.

```bash
det shell start -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=0
det shell show-ssh-command <shell-id>
./cluster/stage.sh <ssh-target>               # ~730 MB of code + corpora
```

Then, still on **slots=0** (downloading while holding an A100 wastes the allocation):

```bash
export HF_HOME=/pfss/mlde/workspaces/mlde_wsp_IL_rm_bias/hf_cache
python cluster/prefetch_models.py --check     # connectivity + free space
python cluster/prefetch_models.py --tier small
```

`--check` answers the one thing we could not determine from outside: **whether compute nodes
have outbound internet.** If they do not, fetch the checkpoints on a machine that does and
rsync the cache across.

## 3. Parity smoke test — do this before spending the allocation

Everything so far ran on the MLX (Apple Silicon) backend. The cluster uses the
transformers/CUDA path, which the demographic arms have never exercised. We have reference
numbers, so this is a real regression check rather than a vibe check:

```bash
det shell start -w IL_rm_bias --config-file cluster/config.yaml     # slots: 1
# then, inside:
python experiments/run_experiment.py --config configs/demographic_credit_sex_qwen06.yaml
```

**Expected: auto-influence 1.00 baseline → 0.06 nulled.** A mismatch means the CUDA path
diverges, and every scaled number would inherit the fault. This single run catches padding
side, dtype, and chat-template differences at once.

## 4. Scale the ladder

| models | `resources.slots` |
|---|---|
| 0.6B, DeBERTa, 3× 8B | 1 |
| 2× 27B, 2× 32B (54–64 GB bf16) | 1 |
| 2× 70B (~140 GB bf16) | **2** |

`device_map="auto"` is already in the loader, so multi-GPU sharding needs no code change.

Interactive:
```bash
det shell start -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=2
```

Unattended (preferred once the smoke test passes — it releases the GPU when the script exits):
```bash
det experiment create --project_id <ID> cluster/config.yaml    # add an `entrypoint:`
```

Create a project first: `det project create IL_rm_bias scaling`.

---

## Things that will bite

- **No default compute pool** on this workspace, so `resource_pool: 42_Compute` must be set on
  every launch. It is in `config.yaml`; do not drop it. `42_Priority` does not exist for us.
- **Idle shells keep burning GPU quota.** `det shell kill <id>` when done, or launch with
  `slots=0`. With a two-month allocation, a forgotten JupyterLab is expensive.
- **Write results to PFSS.** Anything under the instance is deleted on exit, and overfilling
  instance storage crashes the node and any job sharing it.
- **Inodes:** aim under 2M per workspace. We are naturally fine — a few large corpora and
  single-file `pairs.jsonl` — but the HF cache of 11 checkpoints is the one real consumer.
  `tar` up caches you are done with rather than leaving them expanded.
- **Determined.AI is end-of-life.** The only docs are at
  `https://login01.ai.tu-darmstadt.de:8080/docs/index.html`, and its search is broken; navigate
  by the left-hand tree.
- **Two Nemotron model ids were never verified** (`working_notes.tex` flags this). They are in
  `prefetch_models.py` but skipped by default — confirm them against current NVIDIA releases
  before the 32B/70B runs, rather than discovering it mid-ladder.
