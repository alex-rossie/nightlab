# GPU runner setup

The cloud routines have no GPU. Experiments run on the lab machine through a self-hosted GitHub Actions runner, triggered when a human applies the `needs-gpu` label to a PR and then approves the `gpu` deployment. The runner:

- runs one job at a time, because there is one runner; queued jobs wait
- has a 45-minute job limit and a 30-minute limit on the training step, which is the 10-minute budget (compile included) plus the final evaluation over the whole validation slice and slack
- never writes to the ledger; it comments the result on the PR and uploads `result.json`, or, when training fails or hits the time limit, comments the failure, removes `needs-gpu`, and labels the PR `stuck`

Read the Safety section before registering anything.

## Install

1. Put `uv` and `gh` on the PATH the runner service will see (for example `/usr/local/bin`). The job installs its own `uv` but calls `gh` directly.
2. Create the runner under the repo: Settings → Actions → Runners → New self-hosted runner. Follow the Linux steps. Add the label `rocm` when prompted.
3. Set the persistent data directory and the hardware label before starting the service. The runner reads `.env` at startup and passes the variables to every job step; the job refuses to start if either is missing:

```
echo "NIGHTLAB_DATA=$HOME/nightlab-data" >> ~/actions-runner/.env
echo "NIGHTLAB_HARDWARE=amd-consumer-24gb" >> ~/actions-runner/.env
```

4. Prime the data cache once, outside Actions. The workflow checks that both token files exist and fails with a pointer here if they do not; it never downloads inside a job:

```
uv sync --extra rocm --extra data
NIGHTLAB_DATA=$HOME/nightlab-data uv run nightlab data fineweb-edu
```

That downloads and tokenizes about 300M tokens and takes a while the first time. Training reads `NIGHTLAB_DATA` too; with no variable set both commands use `./data` in the checkout.

5. Run the runner as a systemd service so it survives reboots. Restart it after any change to `.env`:

```
cd ~/actions-runner && sudo ./svc.sh install $USER && sudo ./svc.sh start
```

6. Create the `gpu` environment: Settings → Environments → New environment → `gpu`, and add yourself as a required reviewer. The job is bound to that environment and waits for the approval click. If the environment does not exist GitHub creates it without protection and the click is skipped, so do this before the first label.

The job keeps its virtualenv in `$HOME/nightlab-venv` because the checkout is wiped on every run.

## Safety

A self-hosted runner executes code from PRs. The workflow uses the `pull_request_target` trigger, so GitHub reads the workflow file, its trigger type, its job condition, and its environment binding from `main`, never from the PR. A labeled PR runs only when all of these hold:

- the `needs-gpu` label was applied by a user account that is the repository owner, not a bot or another collaborator
- the PR is open and its head is a branch in this repo, not a fork
- the `gpu` environment approved the job

What that does and does not protect against, plainly:

- **The routines act as you.** They authenticate with the repository owner's GitHub identity, so to GitHub a routine applying `needs-gpu` or approving the environment is indistinguishable from you doing it. The sender check and the environment reviewer become real boundaries only once the routines run under a separate identity with no write access to this repo. Until then the guard is on the agent side: the implement prompt forbids the label; the PreToolUse hook in `.claude/hooks/` blocks the obvious `gh` forms of adding it, of approving a deployment, and of force-pushing, from Bash in any session opened in this repo; and `.claude/settings.json` denies `gh api`. That stops a routine that follows stale instructions or a naive prompt injection. It does not stop a determined one: a label passed through a shell variable or a non-Bash tool gets past the hook.
- **Apply the label yourself, in the GitHub UI, after reading the diff, then approve the deployment when GitHub asks.** `/lab review` shows you the diff and the link and will not do either for you.
- **The routines push to `main`.** Three of them commit night reports and README renders there directly, so protecting `main` would break them as written; and since they act as you, a protection rule would not bind them anyway. The workflow gate is safe from PR edits because of `pull_request_target`, not because of branch protection. Set "Require approval for all external contributors" in Actions settings; the default only covers first-time contributors.
- **You do not need the runner.** `/lab run N` on this machine, after reading the diff, produces the same ledger-eligible result without registering anything. Leave the runner unregistered until the points above are settled.
