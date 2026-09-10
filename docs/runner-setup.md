# GPU runner setup

The cloud routines have no GPU. Experiments run on the lab machine through a self-hosted GitHub Actions runner, triggered by the `needs-gpu` label on a PR. The runner:

- runs one job at a time (the `gpu` concurrency group)
- has a 40-minute timeout, which is the 10-minute budget plus compile, data cache checks, and slack
- never writes to the ledger; it comments the result on the PR and uploads `result.json`

## Install

1. Create the runner under the repo: Settings → Actions → Runners → New self-hosted runner. Follow the Linux steps. Add the labels `rocm` when prompted.
2. Run it as a systemd user service so it survives reboots:

```
cd ~/actions-runner && sudo ./svc.sh install $USER && sudo ./svc.sh start
```

3. Set a persistent data directory so FineWeb-Edu is tokenized once:

```
echo "NIGHTLAB_DATA=$HOME/nightlab-data" >> ~/actions-runner/.env
echo "NIGHTLAB_HARDWARE=amd-consumer-24gb" >> ~/actions-runner/.env
```

4. Prime the cache once, outside Actions:

```
uv sync --extra rocm --extra data
uv run nightlab data fineweb-edu --root ~/nightlab-data
```

That downloads and tokenizes about 300M tokens and takes a while the first time.

## Safety

Self-hosted runners execute code from PRs. This repo only accepts PRs from branches in the repo itself, never from forks, and the workflow triggers on `labeled`, so a human has to apply `needs-gpu` before anything runs. Keep it that way. Set "Require approval for all outside collaborators" in Actions settings as a second lock.
