"""CPU smoke tests. These run in CI on every push and must stay under a minute."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
import torch

from nightlab import components, ledger
from nightlab.config import ExperimentConfig
from nightlab.data import FixedWindows, fingerprint, prepare
from nightlab.model import GPT
from nightlab.train import git_commit, run

TINY = {
    "name": "smoke",
    "model": {"n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 64, "vocab_size": 50304},
    "optim": {"warmup_steps": 2},
    "train": {"batch_size": 4, "seq_len": 32, "wall_seconds": 3, "eval_every_seconds": 1,
              "eval_batches": 2, "final_eval_batches": 4, "compile": False, "dtype": "float32"},
    "data": {"dataset": "synthetic"},
}


@pytest.fixture(autouse=True)
def _no_ambient_data_root(monkeypatch):
    # On the lab machine NIGHTLAB_DATA points at the real token cache; tests that
    # set data.path must not be redirected into it.
    monkeypatch.delenv("NIGHTLAB_DATA", raising=False)


def test_registry_has_baseline_components():
    assert "causal" in components.available("attention")
    assert {"swiglu", "gelu", "relu2"} <= set(components.available("mlp"))
    assert {"rmsnorm", "layernorm"} <= set(components.available("norm"))
    assert {"adamw", "muon"} <= set(components.available("optim"))


def test_config_rejects_unknown_keys():
    with pytest.raises(KeyError):
        ExperimentConfig.from_dict({"model": {"n_layers": 3}})


def test_model_forward_shapes():
    cfg = ExperimentConfig.from_dict(TINY)
    model = GPT(cfg.model)
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, cfg.model.vocab_size)
    assert loss.item() > 0
    assert model.num_params() > 0


@pytest.mark.parametrize("optim", ["adamw", "muon"])
def test_train_loop_produces_result(tmp_path, optim):
    raw = json.loads(json.dumps(TINY))
    raw["optim"]["name"] = optim
    raw["data"]["path"] = str(tmp_path / "data")
    cfg = ExperimentConfig.from_dict(raw)
    result = run(cfg, out_dir=tmp_path / "run", quiet=True)
    assert result["steps"] > 0
    assert result["val_loss"] > 0
    assert result["verdict"] == "pending"
    assert result["final_eval_batches"] == 4
    assert len(result["data_fingerprint"]) == 16
    assert (tmp_path / "run" / "result.json").exists()
    assert (tmp_path / "run" / "log.jsonl").exists()
    # A synthetic CPU result must never be ledger-eligible on hardware alone,
    # but structurally it should validate once given a clean commit.
    result["commit"] = "0" * 40
    ledger.validate(result)
    with pytest.raises(FileExistsError):
        run(cfg, out_dir=tmp_path / "run", quiet=True)


def test_ledger_rejects_dirty_commit(tmp_path):
    raw = json.loads(json.dumps(TINY))
    raw["data"]["path"] = str(tmp_path / "data")
    result = run(ExperimentConfig.from_dict(raw), out_dir=tmp_path / "run", quiet=True)
    result["commit"] = "abc123-dirty"
    with pytest.raises(jsonschema.ValidationError):
        ledger.validate(result)


def test_checked_in_ledger_is_valid():
    assert ledger.validate_file() == []


def test_experiment_configs_load():
    base = ExperimentConfig.from_yaml(ledger.ROOT / "experiments" / "baseline" / "config.yaml")
    for path in Path(ledger.ROOT / "experiments").glob("*/config.yaml"):
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.train.wall_seconds == 600, f"{path}: budget must be 600 s"
        assert cfg.name == path.parent.name
        # One change per experiment means a model or optimizer change. The
        # training budget, eval schedule, and data are not levers.
        assert cfg.train == base.train, f"{path}: train section must match baseline"
        assert cfg.data == base.data, f"{path}: data section must match baseline"


def test_compare_against_checked_in_baseline():
    from nightlab.review import compare, format_compare, pr_comment

    entries = ledger.load()
    if not entries:
        pytest.skip("no ledger entries")
    base = next(e for e in entries if e["verdict"] == "baseline")
    c = compare(base)
    assert c["baseline"]["run_id"] == base["run_id"]
    assert c["delta_val_loss"] == 0
    assert "val loss" in format_compare(c)
    assert "GPU run complete" in pr_comment(base)


def test_compare_refuses_a_baseline_scored_differently(tmp_path):
    from nightlab.review import compare, format_compare, pr_comment

    entries = ledger.load()
    if not entries:
        pytest.skip("no ledger entries")
    base = next(e for e in entries if e["verdict"] == "baseline")
    raw = json.loads(json.dumps(TINY))
    raw["data"]["path"] = str(tmp_path / "data")
    result = run(ExperimentConfig.from_dict(raw), out_dir=tmp_path / "run", quiet=True)
    result["hardware"] = base["hardware"]
    c = compare(result)
    assert c["baseline"] is None and c["baseline_problems"]
    assert "NOT COMPARABLE" in format_compare(c)
    assert "No comparable baseline" in pr_comment(result)


def _git(cwd, *args):
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
           "GIT_CONFIG_NOSYSTEM": "1"}
    ident = ["-c", "user.email=t@t", "-c", "user.name=t", "-c", "init.defaultBranch=main"]
    return subprocess.run(["git", *ident, *args], cwd=cwd, check=True, env=env,
                          capture_output=True, text=True)


def test_git_commit_marks_staged_and_untracked_dirty(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("runs/\n")
    _git(tmp_path, "add", ".gitignore")
    _git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.chdir(tmp_path)
    assert len(git_commit()) == 40
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "x.json").write_text("{}")
    assert len(git_commit()) == 40, "ignored files are not dirt"
    (tmp_path / "new_component.py").write_text("x = 1\n")
    assert git_commit().endswith("-dirty"), "an untracked file is dirt"
    _git(tmp_path, "add", "new_component.py")
    assert git_commit().endswith("-dirty"), "a staged file is dirt"


def test_train_reads_nightlab_data(tmp_path, monkeypatch):
    root = tmp_path / "elsewhere"
    cwd = tmp_path / "checkout"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("NIGHTLAB_DATA", str(root))
    raw = json.loads(json.dumps(TINY))
    result = run(ExperimentConfig.from_dict(raw), out_dir=tmp_path / "run", quiet=True)
    assert (root / "synthetic" / "train.bin").exists()
    assert not (cwd / "data").exists(), "training must not touch ./data when NIGHTLAB_DATA is set"
    assert result["data_tokens"]["train"] > 0
    assert result["device"] in ("cuda", "cpu")


def test_fixed_windows_are_fixed(tmp_path):
    d = prepare("synthetic", tmp_path / "data")
    val = FixedWindows(d / "val.bin", seq_len=32, batch_size=4)
    first_two = [(x.clone(), y.clone()) for x, y in val.batches(2)]
    again = list(val.batches(2))
    whole = list(val.batches())
    assert len(whole) == len(val) > 2
    for (x, y), (x2, y2), (x3, y3) in zip(first_two, again, whole[:2], strict=True):
        assert torch.equal(x, x2) and torch.equal(x, x3)
        assert torch.equal(y, y2) and torch.equal(y, y3)
        assert torch.equal(x[:, 1:], y[:, :-1]), "targets are the next token"
    assert whole[0][0][1, 0] == whole[0][1][0, -1], "windows are contiguous"


def test_fixed_windows_refuse_a_slice_too_short_for_one_batch(tmp_path):
    d = prepare("synthetic", tmp_path / "data", max_tokens=1000)
    with pytest.raises(ValueError, match="needs at least"):
        FixedWindows(d / "val.bin", seq_len=32, batch_size=4)


def test_fingerprint_tracks_bytes(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(bytes(range(256)) * 16)
    fp = fingerprint(a)
    assert fp == fingerprint(a) and len(fp) == 16
    a.write_bytes(bytes(range(256)) * 15 + bytes(range(255)) + b"\x00")
    assert fingerprint(a) != fp


def test_muon_orthogonalizes_qkv_slices_separately():
    from nightlab.components.optim import Muon

    torch.manual_seed(0)
    d = 16
    grad = torch.randn(3 * d, d)
    fused = torch.nn.Parameter(torch.randn(3 * d, d))
    fused.grad = grad.clone()
    parts = [torch.nn.Parameter(fused.detach()[i * d:(i + 1) * d].clone()) for i in range(3)]
    for i, p in enumerate(parts):
        p.grad = grad[i * d:(i + 1) * d].clone()
    Muon([{"params": [fused], "chunks": 3}], lr=0.1).step()
    Muon([{"params": parts, "chunks": 1}], lr=0.1).step()
    stacked = torch.cat([p.detach() for p in parts])
    assert torch.allclose(fused.detach(), stacked, atol=1e-2)
    joint = torch.nn.Parameter(stacked.clone())
    joint.grad = grad.clone()
    Muon([joint], lr=0.1).step()
    assert not torch.allclose(joint.detach(), fused.detach(), atol=1e-2), \
        "joint orthogonalization must differ, or the chunks option does nothing"


def test_muon_weight_decay_is_not_adamws():
    raw = json.loads(json.dumps(TINY))
    raw["optim"] = {"name": "muon", "weight_decay": 0.1, "warmup_steps": 2}
    cfg = ExperimentConfig.from_dict(raw)
    muon, adamw = components.get("optim", "muon")(GPT(cfg.model), cfg.optim)
    assert all(g["weight_decay"] == 0.0 for g in muon.param_groups)
    assert adamw.param_groups[0]["weight_decay"] == 0.1
    assert sorted(g["chunks"] for g in muon.param_groups) == [1, 3]


def _gpu_like(result, **over):
    entry = json.loads(json.dumps(result))
    entry.update({"commit": "0" * 40, "hardware": "amd-consumer-24gb", "device": "cuda",
                  "budget_seconds": 600, "wall_seconds": 600.2, "verdict": "baseline"})
    entry["config"]["train"]["wall_seconds"] = 600
    entry["config"]["data"]["dataset"] = "fineweb-edu"
    entry.update(over)
    entry["config"]["seed"] = entry["seed"]
    entry["config"]["name"] = entry["experiment"]
    return entry


def test_ledger_refuses_smoke_stale_baselines_and_lone_replications(tmp_path):
    raw = json.loads(json.dumps(TINY))
    raw["data"]["path"] = str(tmp_path / "data")
    result = run(ExperimentConfig.from_dict(raw), out_dir=tmp_path / "run", quiet=True)
    path = tmp_path / "ledger.jsonl"
    seed = result["seed"]

    smoke = dict(result, commit="0" * 40, verdict="replicated")
    with pytest.raises(ValueError, match="budget"):
        ledger.append(smoke, path)

    ledger.append(_gpu_like(result, run_id="baseline-seed-1"), path)

    exp = _gpu_like(result, run_id="experiment-1", experiment="exp", verdict="failed", pr=7)
    with pytest.raises(ValueError, match="baseline_run_id"):
        ledger.append(exp, path)
    exp["baseline_run_id"] = "baseline-seed-1"
    ledger.append(exp, path)

    nopr = _gpu_like(result, run_id="experiment-nopr", experiment="exp", verdict="failed",
                     baseline_run_id="baseline-seed-1")
    with pytest.raises(ValueError, match="pr is missing"):
        ledger.append(nopr, path)

    # A baseline scored under another evaluation protocol is not comparable.
    stale = _gpu_like(result, run_id="experiment-stale", experiment="exp", verdict="failed",
                      pr=7, baseline_run_id="baseline-seed-1", final_eval_batches=8)
    with pytest.raises(ValueError, match="not comparable"):
        ledger.append(stale, path)
    assert ledger.baseline_for(ledger.load(path), "amd-consumer-24gb", like=stale) is None

    # `replicated` needs a second seed of the same config judged replicated or
    # partial; a `failed` sibling or the same seed does not count.
    rep = _gpu_like(result, run_id="experiment-2", experiment="exp", verdict="replicated",
                    pr=7, baseline_run_id="baseline-seed-1", seed=seed + 1)
    with pytest.raises(ValueError, match="second seed"):
        ledger.append(rep, path)
    partial = _gpu_like(result, run_id="experiment-3", experiment="exp", verdict="partial",
                        pr=7, baseline_run_id="baseline-seed-1", seed=seed + 2)
    ledger.append_all([partial, rep], path)

    # Two seeds can enter as replicated in one call; alone, neither could.
    pair = [_gpu_like(result, run_id=f"experiment-two-{i}", experiment="exp2",
                      verdict="replicated", pr=8, baseline_run_id="baseline-seed-1",
                      seed=100 + i) for i in range(2)]
    with pytest.raises(ValueError, match="second seed"):
        ledger.append(pair[0], path)
    ledger.append_all(pair, path)
    assert ledger.validate_file(path) == []
    assert len(ledger.load(path)) == 6


def test_gpu_label_hook_blocks_only_applying_the_label():
    hook = ledger.ROOT / ".claude" / "hooks" / "guard_gpu_label.py"

    def code(command):
        payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
        return subprocess.run([sys.executable, str(hook)], input=payload,
                              capture_output=True, text=True).returncode

    blocked = [
        "gh pr edit 12 --add-label needs-gpu",
        "gh pr edit 12 --add-label=needs-gpu",
        "gh pr edit 12 --add-label NEEDS-GPU",
        "gh pr edit 12 --add-label\tneeds-gpu",
        "gh pr edit 12 \\\n  --add-label needs-gpu",
        'gh pr edit 12 --add-label "experiment, needs-gpu"',
        "gh issue edit 3 --add-label needs-gpu",
        "gh pr create --title x --label experiment,needs-gpu",
        "gh pr create --title x -lneeds-gpu",
        "gh pr list --label experiment && gh pr edit 3 --add-label needs-gpu",
        'gh api repos/a/b/issues/1/labels -f "labels[]=needs-gpu"',
        'echo \'{"labels":["experiment","needs-gpu"]}\' '
        "| gh api -X POST repos/a/b/issues/1/labels --input -",
        "gh api -X POST repos/a/b/actions/runs/1/pending_deployments",
        "git push origin main --force",
        "git push -f origin exp/0002",
        "git push origin +main",
        "git push --force-with-lease origin exp/0002",
    ]
    allowed = [
        "gh pr edit 12 --remove-label needs-gpu --add-label gpu-done",
        "gh pr edit 12 --add-label gpu-done --remove-label needs-gpu",
        "gh pr list --label needs-gpu --json number",
        "prs=$(gh pr list --label needs-gpu --state open --json number)",
        "GH_PAGER=cat gh -R alex-rossie/nightlab pr list --label needs-gpu",
        "grep -rn needs-gpu docs/",
        'git commit -m "docs: needs-gpu is human-applied"',
        "git push -u origin exp/0002",
        "git push origin exp/0002",
        "uv run pytest -q",
    ]
    for command in blocked:
        assert code(command) == 2, f"should be blocked: {command!r}"
    for command in allowed:
        assert code(command) == 0, f"should be allowed: {command!r}"
