# Reference hardware

The lab machine is a single consumer AMD GPU with 24 GB of VRAM, running PyTorch's ROCm wheels on Linux. The exact card is not the point. The point is that it is one card, it is the same card every night, and anyone with a card of the same class can check the numbers.

| | |
|---|---|
| GPU | one consumer AMD RDNA card, 24 GB |
| Stack | PyTorch ROCm 7.2 wheels from the PyTorch index, version pinned in `uv.lock` |
| OS | Linux with the in-tree `amdgpu` driver |

The hardware string in the ledger comes from `torch.cuda.get_device_name()`, lower-cased with spaces replaced by dashes, unless `NIGHTLAB_HARDWARE` is set. The reference machine sets it to `amd-consumer-24gb`. Baselines are matched on that string, so a run on any other card gets its own baseline and is never compared against the reference card.

## Why an AMD consumer card

Because almost nobody publishes small-scale research results on one, and the ROCm software stack is now good enough that the lack of results is a gap rather than a verdict. Fused SDPA, `torch.compile`, and bf16 all work. If something in the stack turns out not to work, that is itself a result and goes in the night report.

## Setup

The PyTorch ROCm wheels bundle their own ROCm runtime, so the system needs only the `amdgpu` kernel driver that mainstream distributions ship. No `/opt/rocm` install is required.

```
uv sync --extra rocm
uv run python -c "import torch; print(torch.cuda.get_device_name())"
```

If the card is not detected, check that the user is in the `render` and `video` groups. Older or unusual cards may need `HSA_OVERRIDE_GFX_VERSION` set; current RDNA3 cards do not.
