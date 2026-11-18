# PicoChat

From-scratch GPT: custom BPE → pretrain → SFT → RL (optional) → chat demo.

Inspired by Karpathy’s [nanochat](https://github.com/karpathy/nanochat) and the GPT speedrun. Same idea — plain PyTorch, no trainer frameworks — not a fork.

**Demo:** [seif-222--picochat-chat-serve.modal.run](https://seif-222--picochat-chat-serve.modal.run) (~20 seconds cold start)

**Write-up:** [Training a 253M chat model from scratch](WRITEUP.md)

**W&B:** [seif-222-student/PicoChat](https://wandb.ai/seif-222-student/PicoChat)

253.2M params · ~1.97B pretrain tokens · ~13 hours on 1× A100-80GB · ~$40 in useful GPU compute on Modal.

Distributed training is supported in the training scripts for larger-scale runs.

| Stage | Status |
| --- | --- |
| Tokenizer | Done. Vocab 65,536 (rustbpe to train, tiktoken to serve). |
| Pretrain | Done. Step 3750, val 3.121, ~7.8 tok/param. |
| SFT | Done. 1 epoch, ~164k conversations, best val 1.5527. **This is the chat model.** |
| RL (GRPO / GSM8K) | Wired. Skipped at this version. See Reinforcement learning. |
| Demo | Live on Modal (CPU, scale-to-zero). |

---

## Run it locally

The clone includes the model code and trained 65,536-token tokenizer. You can test both on CPU without downloading training data or model weights. To chat with the trained model, use the demo above or supply the SFT checkpoint described below.

### Install and smoke-test

From a Linux shell with Git and Python 3.12:

```bash
git clone https://github.com/seif-222/PicoChat.git
cd PicoChat
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.6.0' 'tiktoken==0.12.0'
cd PicoChat
```

The second `cd PicoChat` enters the source directory containing `Config.py`. Run the remaining commands there. For GPU training, install a PyTorch build compatible with your CUDA driver. The CPU checks below were run with PyTorch 2.6.0 and tiktoken 0.12.0.

```bash
python - <<'PY'
import torch
from Config import GPT_config
from Model import GPT
from Tokenizer import RustTokenizer

tokenizer = RustTokenizer.from_directory("output/tokenizer")
text = "PicoChat from scratch."
assert tokenizer.decode(tokenizer.encode(text)) == text
torch.manual_seed(33)
config = GPT_config(n_embd=32, n_head=2, n_kv_head=2, n_layer=4,
                    block_size=12, vocab_size=64, window_size=4, device="cpu")
model = GPT(config)
tokens = torch.randint(0, config.vocab_size, (2, 10))
logits, loss = model(tokens[:, :-1], tokens[:, 1:].contiguous())
loss.backward()
print(tokenizer.get_vocab_size(), tuple(logits.shape), bool(torch.isfinite(loss)))
PY
```

Expected output: `65536 (2, 9, 64) True`. The tiny model uses random weights and synthetic token IDs to exercise a forward and backward pass.

### Chat with the trained checkpoint

The reported result uses SFT step 9,633. The checkpoint is separate from Git. If you have a trusted copy, set `PICOCHAT_CHECKPOINT` to its path. The original local layout was `output/Models/SFT/model_checkpoint_best.pt` inside the source directory. You can also use the best checkpoint from your own SFT run.

```bash
export PICOCHAT_CHECKPOINT=output/Models/SFT/model_checkpoint_best.pt
python - <<'PY'
import os
from itertools import islice

import torch
from Config import GPT_config
from Engine import InferenceEngine
from Model import GPT
from Tokenizer import RustTokenizer

torch.set_num_threads(min(4, torch.get_num_threads()))
with torch.serialization.safe_globals([GPT_config]):
    ckpt = torch.load(os.environ["PICOCHAT_CHECKPOINT"], map_location="cpu",
                      weights_only=True, mmap=True)
cfg = ckpt["config"]
cfg.device = "cpu"
with torch.device("meta"):
    model = GPT(cfg)
model.load_state_dict(ckpt["model"], assign=True)
model.eval()

tokenizer = RustTokenizer.from_directory("output/tokenizer")
messages = [{"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": ""}]
prompt = torch.tensor([tokenizer.render_for_completion({"messages": messages})])
eos = tokenizer.encode_special("<|assistant_end|>")
stream = InferenceEngine(model).stream_generation(prompt, temperature=0, eos_token=eos)
answer = []
for token in islice(stream, 48):
    if token.item() == eos:
        break
    answer.append(token.item())
stream.close()
print(tokenizer.decode(answer))
PY
```

On the original SFT checkpoint this printed `The capital of France is Paris.`

### Train and evaluate

The full configuration was run on an A100 with 80 GB of GPU memory. For local training, use a CUDA-enabled environment and start with a fresh output directory. The trainer automatically resumes from checkpoints in its log directory.

1. **Prepare the data.** Download the pretokenized FineWeb-Edu shards and build the SFT conversation file. This downloads several gigabytes. The tokenizer in Git matches the published shards.

   ```bash
   python -m pip install numpy datasets huggingface_hub tqdm requests
   python - <<'PY'
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id="seif-222/Tokenized_FineWebEdu_Shards",
                     repo_type="dataset", local_dir="output/data/fineweb-edu_tokenized")
   PY
   python prepare_sft_data.py
   ```

2. **Set local paths in `Config.py`.** Set `tokenizer_dir` to `"output/tokenizer"`, `data_root` to `"output/data/fineweb-edu_tokenized"`, and `use_wandb` to `False` for a local run. The original Modal mount expects `/data/tokenizer/rustbpe_tokenizer.pkl`. The clone already supplies `output/tokenizer/rustbpe_tokenizer.pkl` relative to the source directory. Set the directory, not the `.pkl` filename. The defaults target the original Modal volume and W&B account. To use W&B, install `wandb` and set your own entity and project. Pretraining also downloads the HellaSwag validation data on its first evaluation.

3. **Run pretraining, then supervised fine-tuning.** Run the commands separately so the base checkpoint exists before SFT starts.

   ```bash
   LOG_DIR=output/checkpoints/pretrain python Train.py
   ```

   ```bash
   SFT_PRETRAIN_CKPT=output/checkpoints/pretrain/model_checkpoint_step_03750.pt \
   SFT_DATA_PATH=output/data/sft_conversations.jsonl \
   SFT_LOG_DIR=output/checkpoints/sft python train_sft.py
   ```

   For the chat example above, point `PICOCHAT_CHECKPOINT` at `output/checkpoints/sft/model_checkpoint_best.pt`. The committed config runs through step 3,750 with a cosine horizon of 2,200. The published run started with a horizon of 3,100, resumed from checkpoint 1,500 with the shorter horizon, then continued through step 3,750 at the learning-rate floor. To follow that schedule, first set `max_steps = 3100` and `training_steps = 3750`, launch the pretraining command, and stop after the step-1,500 checkpoint is written and before the next checkpoint. Set `max_steps = 2200`, leave `training_steps = 3750`, and run the same command with the same `LOG_DIR` to resume. A single uninterrupted run with the default configuration uses a different learning-rate history. The [write-up](WRITEUP.md) records the transitions and validation results.

4. **Evaluate the checkpoint.** This invokes the repository’s HellaSwag evaluator for all 10,042 validation examples. On PyTorch 2.6 the checkpoint’s serialized configuration class requires explicit allowlisting. Use a checkpoint you trust.

   ```bash
   export PICOCHAT_CHECKPOINT=output/checkpoints/sft/model_checkpoint_best.pt
   python - "$PICOCHAT_CHECKPOINT" <<'PY'
   import runpy
   import sys
   import torch
   from Config import GPT_config

   sys.argv = ["HellaSwag.py", "-c", sys.argv[1], "-t", "output/tokenizer"]
   with torch.serialization.safe_globals([GPT_config]):
       runpy.run_path("HellaSwag.py", run_name="__main__")
   PY
   ```

**Optional paths.** To rebuild the tokenizer and shards for a new run, install `rustbpe`, then run `python Train_tokenizer.py` followed by `python Shard_tokenization.py` in a separate checkout. Training a tokenizer writes to the same path as the bundled one. To experiment with GRPO after SFT, run `python gsm8k_data.py`, set `rl_log_dir` to a writable local directory in `Config.py`, and use:

```bash
RL_SFT_CKPT=output/checkpoints/sft/model_checkpoint_best.pt \
RL_GSM8K_TRAIN=output/data/gsm8k/train.jsonl \
RL_GSM8K_TEST=output/data/gsm8k/test.jsonl python train_rl.py
```

The published chat checkpoint ends at SFT. The optional RL stage and full training/benchmark runs were not repeated for the CPU smoke check.

---

## Model

253.2M trainable. `n_embd=896`, `n_layer=14`, `n_head=7`, context 1024, vocab 65,536.

* Pre-norm, weightless RMSNorm
* RoPE (base 100k)
* QK-Norm (RMSNorm on Q/K, then ×1.2)
* ReLU² MLP (4×)
* Untied language-model head, zero-init output projections
* Logit softcap
* Sliding-window attention (`SSSL`, window 256; every 4th layer and the last layer are full) (cheaper attention, periodic full layers keep long range)
* SmearGate (mix in the previous token)
* Value embeddings (token identity on attention values; one small shared table, every other layer)
* Residual-stream skip (mix the first hidden state back in each layer)
* Mid-stack backout
* Muon + AdamW (Muon on matrices, AdamW on embeddings / head / scalars)

---

## Inference engine

Custom `Engine` — this architecture does not go through vLLM or Hugging Face pipelines.

* KV cache (sliding-window + SmearGate; checked against a full forward pass)
* Streaming generation
* Sampling: temperature, top-p, repetition penalty, top-k (current setup: 0.8, 0.95, 1.05, no top-k respectively)
* Generation clipped to the remaining room in the 1024 context

---

## Pretraining

* FineWeb-Edu `sample-10BT`, 40 shards × 100M tokens = 4B tokens (≈ 1.97B tokens used for this version)
* Custom BPE on 2B characters, GPT-4-style split
* Chat special tokens put in the vocab up front

Planned warmup + cosine over 3100 steps. The run died mid-way (Modal server cutoff). Last clean checkpoint was step 1500. Horizon was cut to 2200 to finish on budget — cosine is locked to `max_steps`, so that cut dropped the learning rate at the resume point (~61% of peak → ~34%), an intentional trade-off to finish annealing on budget. 1500→2199 then fully annealed (val 3.167, ~1.15B tokens). (WSD prevents this: you can train and cut without pre-planning the step count, but in this case training was already 1500 steps into cosine, so that schedule was kept.)

Extra budget (a fresh $30). Val was still falling at 2199. Two options: rewind to step 1500 and finish the original 3100-step cosine, or continue from 2200 at the floor learning rate through step 3750. I chose the second — 700 steps were already trained, so more tokens seen over a cleaner schedule. `max_steps` stayed at 2200 (a new cosine aimed at 3750 would have jumped the learning rate ~4.5×). Val **3.167 → 3.121**, ~1.97B tokens ≈ **7.8 tok/param**.

The nanochat speedrun with Muon puts compute-optimal around **8–12 tok/param**, not Chinchilla-20. This run sits on that line.

---

## SFT

~164k conversations:

* `everyday-conversations-llama3.1-2k`
* `no_robots` (CC-BY-NC-4.0, non-commercial)
* `databricks-dolly-15k`
* `smol-smoltalk` — the smaller, simpler mix (no function-calling / heavy rewrite / hard math), capped at 300k, then longer conversations dropped (~3,200 characters ≈ 1,024 tokens, this model's block size)

Loss only on assistant tokens. One epoch. Best checkpoint is step 9633, val **1.5527** (this is the served model in the demo).

---

## Reinforcement learning

GRPO on GSM8K is in the repo. **Not launched** (mainly because of the scale). SFT best scored 0/20 on a last-number probe, so RL would not be effective at this scale. Karpathy’s d20 only went 4.55% → 7.58% after mid-training that already included GSM8K. RL can make it better at math, not necessarily a better chatbot.

* GSM8K preprocessing
* Grouped sampling
* Group-relative advantages
* KL regularization
* Bounded generation
* Greedy validation
* Modal execution

---

## Infra

* Shards on Hugging Face: `seif-222/Tokenized_FineWebEdu_Shards`
* Train on Modal (`picochat-training`, volume `nanochat-data`): data, pretrain, SFT, RL
* Serve on a separate Modal app (`picochat-chat`)

Load `output/Models/SFT/model_checkpoint_best.pt`.

[seif-222/PicoChat](https://github.com/seif-222/PicoChat)

---

## More information

### Samples

```
User: Hello
Assistant: Hello! How can I help you today?

User: What is the capital of United States?
Assistant: The capital of the United States is Washington, D.C.

User: What is the Photosynthesis process.
Assistant: The Photosynthesis process is a process used by plants, algae, and some bacteria to convert light energy into chemical energy.

User: Is this process important?
Assistant: Yes, the process of photosynthesis is important because it provides the energy needed for all living organisms to function properly.
```

* For information outside the training distribution, the model can make things up or claim false information. Expected at this scale with this training.

### HellaSwag

Full validation split (10,042 examples):

| Checkpoint           | Correct | Accuracy |
| -------------------- | ------- | -------- |
| Pretrain step 2199   | 3068    | 0.3055   |
| Pretrain step 3750   | 3137    | 0.3124   |
| SFT best (step 9633) | 2914    | 0.2902   |

* SFT focuses on chat format, so the raw language-modelling score can drop as the model is conditioned into conversation.

## Files

```
PicoChat/
├── Config.py              # model + training configuration
├── Model.py               # GPT architecture + KV cache
├── optimizer.py           # MuonAdamW
├── Engine.py              # inference / sampling / streaming
├── Tokenizer.py           # BPE tokenizer + chat formatting
├── DataLoader.py          # pretraining + SFT loaders
├── train_utils.py         # shared training utilities
├── Train.py               # pretraining
├── train_sft.py           # supervised fine-tuning
├── train_rl.py            # GRPO / GSM8K
├── prepare_sft_data.py    # SFT dataset preparation
├── gsm8k_data.py          # GSM8K preparation
├── Train_tokenizer.py     # tokenizer training
├── Shard_tokenization.py  # FineWeb-Edu tokenization
├── HellaSwag.py           # evaluation
├── Modal_app.py           # cloud training stages
└── output/tokenizer/rustbpe_tokenizer.pkl  # tokenizer file
```

## Acknowledgements

Andrej Karpathy (nanoGPT, nanochat), Hugging Face (FineWeb-Edu, SmolTalk), Modal.
