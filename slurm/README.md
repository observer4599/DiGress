# Running DiGress on Idun

## 1. Login

```bash
ssh <your_username>@idun-login1.hpc.ntnu.no
```

Do not run training directly on the login node — use SLURM to submit a job instead.

### Passwordless login

Run this once on your **local machine** to copy your public key to Idun:

```bash
ssh-copy-id <your_username>@idun-login1.hpc.ntnu.no
```

You can also add an alias to `~/.ssh/config` on your local machine to avoid typing the full address:

```bash
nano ~/.ssh/config
```

Add the following:

```
Host idun
    HostName idun-login1.hpc.ntnu.no
    User <your_username>
```

After that, logging in is just:

```bash
ssh idun
```

## 2. First-time setup

Do this once after logging in.

### Add SSH key for GitHub

Generate an SSH key on Idun and add it to your GitHub account so you can clone private repositories.

```bash
# Generate a key (press Enter to accept defaults)
ssh-keygen -t ed25519 -C "<your_github_email>"

# Print the public key
cat ~/.ssh/id_ed25519.pub
```

Copy the output, then add it to GitHub:
**GitHub → Settings → SSH and GPG keys → New SSH key**

Verify it works:

```bash
ssh -T git@github.com
```

You should see: `Hi <username>! You've successfully authenticated...`

### Clone the repository

```bash
cd /cluster/home/$USER
mkdir -p Development && cd Development
git clone git@github.com:observer4599/DiGress.git
cd DiGress
```

### Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
source ~/.bashrc
```

### Install dependencies

```bash
pixi install
```

### Compile the orca binary (required for graph metrics)

```bash
cd src/analysis/orca
g++ -O2 -std=c++11 -o orca orca.cpp
cd ../../..
```

## 3. Submit a job

From the repository root:

```bash
sbatch slurm/sbm_job.slurm
```

To use a specific account (if you have multiple):

```bash
# Find your account name and QOS level
# QOS controls job priority: normal (lower priority) vs high (higher priority, shorter queue wait)
sacctmgr show assoc format=Account%15,User,QOS | grep -e QOS -e $USER

# Submit with explicit account
sbatch --account=<your_account> slurm/sbm_job.slurm
```

## 4. Monitor a job

```bash
# List your running and pending jobs
squeue --me

# Stream live output (replace JOBID with the number printed by sbatch)
tail -f logs/sbm_<JOBID>.out
# Or for the most recent job:
tail -f logs/$(ls -t logs/ | head -1)

# Check GPU usage for a running job
srun --jobid=<JOBID> nvidia-smi
# Or for the most recent running job:
srun --jobid=$(squeue --me -h -o %i | tail -1) nvidia-smi

# View resource usage after a job completes
sacct -j <JOBID> --format="JobID,JobName,Elapsed,ReqTRES%45,State"
# Or for the most recent job:
sacct -j $(ls -t logs/ | head -1 | grep -o '[0-9]*') --format="JobID,JobName,Elapsed,ReqTRES%45,State"
```

## 5. Cancel a job

```bash
scancel <JOBID>
# Or for the most recent running job:
scancel $(squeue --me -h -o %i | tail -1)
```
