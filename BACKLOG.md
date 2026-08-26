# BACKLOG — FreeToken-Pascal

Second view of the GitHub **Issues** tab for
[CryptoJones/FreeToken-Pascal](https://github.com/CryptoJones/FreeToken-Pascal).
Every item here has a matching issue, and vice versa.

## Open

- [ ] Upstream the `FT_GRID_CONSTANT` guard as a PR to `FlashML-org/FreeToken` — it is
      additive, follows their existing `__nanosleep` convention, and costs sm_70+ nothing
      ([#1](https://github.com/CryptoJones/FreeToken-Pascal/issues/1))
- [ ] File an upstream issue on `apache-tvm-ffi`: `_get_cuda_target()` reads only
      `nvidia-smi`'s **first** GPU and ignores `CUDA_VISIBLE_DEVICES`, silently breaking
      mixed-generation machines
      ([#2](https://github.com/CryptoJones/FreeToken-Pascal/issues/2))
- [ ] Validate a real MoE end-to-end on sm_60 — `ft bench bw` passes, but no model has
      actually been served on the P100 yet
      ([#3](https://github.com/CryptoJones/FreeToken-Pascal/issues/3))
- [ ] Re-measure a Pascal card in a **Gen4 x16** slot. The 2.9 GB/s figure in `PASCAL.md`
      is a Gen3 x4 chipset slot and understates the architecture
      ([#4](https://github.com/CryptoJones/FreeToken-Pascal/issues/4))
- [ ] Benchmark FreeToken vs `llama.cpp` b10639 on the same MoE checkpoint, and state the
      hardware asymmetry explicitly in any published numbers
      ([#5](https://github.com/CryptoJones/FreeToken-Pascal/issues/5))
- [ ] Document the `torch==2.11.0+cu126` requirement in upstream's `docs/install.md`, or
      have `[accel]` detect a pre-Volta card and warn
      ([#6](https://github.com/CryptoJones/FreeToken-Pascal/issues/6))

## Done

- [x] Guard all four `__grid_constant__` annotations behind `FT_GRID_CONSTANT` so
      sm_60/sm_61 compile while sm_70+ keeps the hint
- [x] Verify both `sm_60` and `sm_86` compile from the patched tree
- [x] Confirm `torch==2.11.0+cu126` restores `sm_60` to the arch list
- [x] Identify `TVM_FFI_CUDA_ARCH_LIST` as the required override on mixed-GPU hosts
- [x] Capture `ft bench bw` ceilings for both the RTX 3060 and the P100

---

*Proudly Made in Nebraska. Go Big Red! 🌽 <https://xkcd.com/2347/>*
