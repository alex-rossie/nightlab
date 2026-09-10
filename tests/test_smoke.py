"""CPU smoke tests. These run in CI on every push and must stay under a minute."""

import json
from pathlib import Path

import jsonschema
import pytest
import torch

from nightlab import components, ledger
from nightlab.config import ExperimentConfig
from nightlab.model import GPT
from nightlab.train import run

TINY = {
    "name": "smoke",
    "model": {"n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 64, "vocab_size": 50304},
    "optim": {"warmup_steps": 2},
    "train": {"batch_size": 4, "seq_len": 32, "wall_seconds": 3, "eval_every_seconds": 1,
              "eval_batches": 2, "compile": False, "dtype": "float32"},
    "data": {"dataset": "synthetic"},
}


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
    assert (tmp_path / "run" / "result.json").exists()
    assert (tmp_path / "run" / "log.jsonl").exists()
    # A synthetic CPU result must never be ledger-eligible on hardware alone,
    # but structurally it should validate once given a clean commit.
    result["commit"] = "0" * 40
    ledger.validate(result)


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
    for path in Path(ledger.ROOT / "experiments").glob("*/config.yaml"):
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.train.wall_seconds == 600, f"{path}: budget must be 600 s"
        assert cfg.name == path.parent.name


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
