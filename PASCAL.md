# FreeToken on Pascal (sm_60 / sm_61)

This fork makes FreeToken's JIT kernels build and run on **pre-Volta NVIDIA GPUs** —
Tesla P100 (sm_60) and the GTX 10xx line (sm_61). Upstream targets sm_75+ and
ships a CUDA 13 build, which excludes Pascal on two independent axes.

Everything here is additive. `sm_70+` behaviour is unchanged.

---

## What upstream blocks, and why

### 1. `__grid_constant__` (the only source-level blocker)

Four kernel parameters are annotated `__grid_constant__`:

| File | Count |
|---|---|
| `python/freetoken/kernel/csrc/jit/fast_index_copy.cuh` | 1 |
| `python/freetoken/kernel/csrc/jit/index.cu` | 2 |
| `python/freetoken/kernel/csrc/jit/store.cu` | 1 |

nvcc rejects it below compute_70:

```
fast_index_copy.cuh(488): error: __grid_constant__ annotation is only
allowed for architecture compute_70 or later
```

Because every PCIe-gather kernel lives behind that error, **the entire
expert-offload path — the core of FreeToken — fails to build on Pascal.**

`__grid_constant__` is a pure optimization hint: it lets the compiler keep a large
kernel parameter in constant memory instead of copying it to local storage. Removing
it changes no semantics.

**The fix** (`include/freetoken/utils.cuh`):

```c
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#  define FT_GRID_CONSTANT
#else
#  define FT_GRID_CONSTANT __grid_constant__
#endif
```

The four sites now use `FT_GRID_CONSTANT`. **sm_70+ still gets the annotation** — this
is a guard, not a removal. Verified: the sm_86 object file is larger than the sm_60
one, because the annotation is still applied there.

This follows a convention upstream already uses — `__nanosleep` in
`fast_index_copy.cuh` is guarded with exactly the same `__CUDA_ARCH__ >= 700` test.

### 2. PyTorch's CUDA 13 wheel (an install-time choice, not a code change)

FreeToken pins `torch>=2.11,<2.12`. As of 2.11, `pip install torch` gives you a
**cu130** wheel, whose arch list is:

```
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

No sm_60. But `torch==2.11.0+cu126` exists and carries:

```
['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
```

So this needs no patch — just the right wheel. PyTorch's own guidance is to use the
CUDA 12.6 builds for Maxwell/Pascal.

### 3. `TVM_FFI_CUDA_ARCH_LIST` — a footgun on mixed-generation boxes

`tvm_ffi/cpp/extension.py::_get_cuda_target()` detects the target arch by running:

```
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

and taking **`.split("\n")[0]` — the first GPU only**.

On a mixed box this silently picks the wrong card. Worse, `nvidia-smi` ignores
`CUDA_VISIBLE_DEVICES`, so masking the unwanted GPU by index *or* by UUID does not
change the result, and `TORCH_CUDA_ARCH_LIST` is not consulted at all.

Set the arch explicitly (space-separated `major.minor`):

```bash
export TVM_FFI_CUDA_ARCH_LIST="6.0 8.6"
```

---

## Installing

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[accel]"

# Pascal-capable torch (NOT the default cu130 wheel)
uv pip install --reinstall "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu126
```

Runtime environment:

```bash
export CUDA_HOME=/usr/local/cuda-12.9        # 12.x: nvcc still accepts compute_60
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TVM_FFI_CUDA_ARCH_LIST="6.0 8.6"      # list every arch in the box
```

CUDA **13** will not work for Pascal — `nvcc --list-gpu-arch` starts at `compute_75`.
Use a 12.x toolkit for the JIT.

---

## Measured on `pluto`

Tesla P100 16GB (sm_60) + RTX 3060 12GB (sm_86), Ryzen 5 5600X, 128GB DDR4,
driver 580.159.04, CUDA 12.9, `ft bench bw`:

| Configuration | CPU STREAM read | PCIe H2D | Selected policy |
|---|---|---|---|
| RTX 3060 only (sm_86) | 30.1 GB/s | **26.4 GB/s** | `hybrid`, ~49% of misses over PCIe |
| P100 as `cuda:0` (sm_60) | 30.1 GB/s | **2.9 GB/s** | `hybrid`, ~9% of misses over PCIe |

### The honest caveat

Pascal now *builds and runs*, but on this machine the P100 is the **worse** FreeToken
device despite having more VRAM. Its ~2.9 GB/s reflects a Gen3 x4 chipset slot, not the
GPU itself — the 3060 sits in the Gen4 x16 slot and gets ~9× the bandwidth.

FreeToken is PCIe-bound on expert misses, so **slot placement matters more than VRAM
capacity here.** If you are putting a Pascal card to work with this fork, give it the
x16 slot. A P100 in a full-bandwidth slot is a genuinely different measurement than the
one above, and has not been taken.

---

## Relationship to upstream

No upstream functionality is removed or altered for supported architectures —
`sm_70+` codegen is byte-for-byte what upstream produces.

This fork is maintained independently. There is no plan to file an upstream PR, and
no expectation of tracking upstream releases on any particular schedule.

Upstream: <https://github.com/FlashML-org/FreeToken> —
FreeToken is Apache 2.0, and so is this fork. Credit for the engine belongs to the
FreeToken authors (Yang et al., UC Berkeley); see their paper at
<https://arxiv.org/abs/2608.16157>.

---

*Proudly Made in Nebraska. Go Big Red! 🌽 <https://xkcd.com/2347/>*
