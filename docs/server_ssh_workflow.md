# Stable SSH Workflow For The A100 Server

This document captures the stable workflow for using the Linux A100 GPU server
from Windows and from Codex. It was adapted from the earlier GenFEC workflow for
this LLaDA Diffusion FEC project.

## Server Identity

- Host: `10.96.50.180`
- User: `rmuller7`
- Main working area: `/mnt/bst/a100/yxie2/rmuller7`
- Intended repo copy: `/mnt/bst/a100/yxie2/rmuller7/LLaDA_Diffusion_FEC`
- GitHub repo: `https://github.com/ReuvenMuller/LLaDA_Diffusion_FEC.git`

## What Is Unstable

The unstable parts are usually transport and terminal behavior, not experiment code:

- repeated password-based SSH prompts,
- interactive SSH sessions inside the Codex terminal,
- streaming long output with `cat`, `scp`, or large terminal dumps.

## Stable Practices

### 1. Use SSH keys, not passwords

Password-based SSH can hang or stall. Key auth makes short SSH commands reliable.

Verification command:

```bash
ssh -T rmuller7@10.96.50.180 "echo connected"
```

If the Windows OpenSSH client connects but hangs during key exchange, use the
explicit key-exchange fallback that worked during sparse-fountain validation:

```bash
ssh -o KexAlgorithms=diffie-hellman-group14-sha256 -T rmuller7@10.96.50.180 "echo connected"
```

Use the same `-o KexAlgorithms=diffie-hellman-group14-sha256` option for `git
pull`, `pytest`, and short status commands when the default SSH path times out.

### 2. Prefer one-shot commands over interactive shells

Use:

```bash
ssh -T rmuller7@10.96.50.180 "hostname; pwd"
```

Avoid opening an interactive shell unless absolutely necessary.

### 3. Use `tmux` for real runs

Do not rely on a foreground SSH session for model loading or experiment runs. Start
long work in `tmux` so it continues if the SSH client disconnects.

Useful `tmux` commands:

```bash
ssh -T rmuller7@10.96.50.180 "tmux ls"
ssh -T rmuller7@10.96.50.180 "tmux capture-pane -pt llada-smoke -S -120"
ssh -t rmuller7@10.96.50.180 "tmux attach -t llada-smoke"
```

### 4. Keep outputs outside the repo copy

Use sibling writable paths for virtualenvs, Hugging Face cache, and run outputs:

```text
/mnt/bst/a100/yxie2/rmuller7/.venvs/llada-diffusion-fec
/mnt/bst/a100/yxie2/rmuller7/.hf-cache-llada-diffusion-fec
/mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs
```

### 5. Pin a GPU manually after checking load

Check current usage:

```bash
ssh -T rmuller7@10.96.50.180 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
```

Then launch with a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=2 ...
```

### 6. Separate setup from execution

First install dependencies once in a project-specific virtualenv. Future runs should
reuse that environment instead of reinstalling.

Recommended setup:

```bash
ssh -T rmuller7@10.96.50.180 "cd /mnt/bst/a100/yxie2/rmuller7 && git clone https://github.com/ReuvenMuller/LLaDA_Diffusion_FEC.git || true"
ssh -T rmuller7@10.96.50.180 "python3 -m venv /mnt/bst/a100/yxie2/rmuller7/.venvs/llada-diffusion-fec"
ssh -T rmuller7@10.96.50.180 "cd /mnt/bst/a100/yxie2/rmuller7/LLaDA_Diffusion_FEC && /mnt/bst/a100/yxie2/rmuller7/.venvs/llada-diffusion-fec/bin/python -m pip install -e '.[hf,dev]'"
```

### 7. Do not stream long logs to the Codex terminal

Avoid:

```bash
ssh ... "cat large_file.log"
```

Prefer:

```bash
ssh -T rmuller7@10.96.50.180 "tail -n 80 /path/to/log"
ssh -T rmuller7@10.96.50.180 "head -n 40 /path/to/log"
ssh -T rmuller7@10.96.50.180 "wc -l /path/to/file"
ssh -T rmuller7@10.96.50.180 "ls -lah /path/to/output_dir"
```

### 8. Prefer small status checks over large transfers

For monitoring:

```bash
ssh -T rmuller7@10.96.50.180 "tmux capture-pane -pt llada-smoke -S -120"
ssh -T rmuller7@10.96.50.180 "ls -lah /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs"
```

### 9. Use a normal local terminal for remote operations when possible

Codex is useful for quick shared visibility, but repeated SSH traffic can be more
fragile there. For long sessions, use a normal Windows terminal and have Codex read
small local status snippets or artifacts afterward.

## Recommended OpenSSH Config

Add this to `C:\Users\reuve\.ssh\config` if desired:

```sshconfig
Host genfec-a100
    HostName 10.96.50.180
    User rmuller7
    IdentityFile C:\Users\reuve\.ssh\id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ConnectTimeout 10
    LogLevel ERROR
```

Then commands become:

```bash
ssh -T genfec-a100 "echo connected"
ssh -T genfec-a100 "tmux ls"
```

## Known Good Commands

### Verify Connectivity

```bash
ssh -T rmuller7@10.96.50.180 "echo connected; hostname; pwd"
```

Expected:

```text
connected
a100
/mnt/bst/a100/yxie2/rmuller7
```

### Check GPU Load

```bash
ssh -T rmuller7@10.96.50.180 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
```

### Clone Or Update This Repo

```bash
ssh -T rmuller7@10.96.50.180 "cd /mnt/bst/a100/yxie2/rmuller7 && if [ -d LLaDA_Diffusion_FEC/.git ]; then cd LLaDA_Diffusion_FEC && git pull --ff-only; else git clone https://github.com/ReuvenMuller/LLaDA_Diffusion_FEC.git; fi"
```

### Run The Fake Smoke On Server

```bash
ssh -T rmuller7@10.96.50.180 "cd /mnt/bst/a100/yxie2/rmuller7/LLaDA_Diffusion_FEC && /mnt/bst/a100/yxie2/rmuller7/.venvs/llada-diffusion-fec/bin/python -m diffusion_fec.experiments.runner --output-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/fake_smoke --sample-count 2 --seed 0"
```

### Run The Real LLaDA Smoke In `tmux`

Pick a GPU after checking `nvidia-smi`, then launch:

```bash
ssh -T rmuller7@10.96.50.180 "tmux new-session -d -s llada-smoke 'cd /mnt/bst/a100/yxie2/rmuller7/LLaDA_Diffusion_FEC && HF_HOME=/mnt/bst/a100/yxie2/rmuller7/.hf-cache-llada-diffusion-fec CUDA_VISIBLE_DEVICES=0 /mnt/bst/a100/yxie2/rmuller7/.venvs/llada-diffusion-fec/bin/python -m diffusion_fec.experiments.runner --output-dir /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/real_llada_smoke --real-llada-smoke --sample-count 1 --loss-rate 0.5 --seed 1 --tokens-per-packet 1 --hash-bits 4 --steps 1'"
```

### Check Run Status

```bash
ssh -T rmuller7@10.96.50.180 "tmux capture-pane -pt llada-smoke -S -120"
ssh -T rmuller7@10.96.50.180 "ls -lah /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/real_llada_smoke"
ssh -T rmuller7@10.96.50.180 "head -n 5 /mnt/bst/a100/yxie2/rmuller7/llada-diffusion-fec-runs/real_llada_smoke/results.csv"
```

## Large File Transfer Workaround

When `scp` or `sftp` stalls, use cloud staging or another resumable transfer path
instead of direct server-to-local transfer. Prefer small status checks while jobs run.
