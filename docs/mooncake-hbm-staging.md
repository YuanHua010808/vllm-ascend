# Mooncake HBM staging experiment

Patch base: vllm-ascend v0.23.0, commit
`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`.

The implementation is included in this experimental branch.
Build/install this checkout in the prepared v0.23.0 NPU environment.
Set `hbm_staging_bytes` in mooncake.json to a positive byte limit
(for example 67108864). The default 0 preserves direct scatter/gather I/O.

PUT gathers each object's address list into a contiguous HBM allocation.
GET reads into that allocation and scatters successful results back to the
original addresses. Key order and byte layout are preserved.

This is an experimental synchronous ACL device-copy implementation. It uses
one transport call per key and allocates/registers staging per operation;
there is no persistent buffer pool or fused kernel. No performance gain is
claimed. Oversized objects fail; failed GETs are not scattered. An unregister
failure deliberately retains the raw allocation to avoid freeing registered
memory, and requires operational recovery.

Local checks: syntax parsing, synthetic round-trip/failed-GET/capacity test,
git diff whitespace check, workspace scaffold checks passed.
Remaining target checks: ACL allocation and registration, stream ordering,
GLM-5.1 output correctness, concurrent transfers, failure cleanup, HBM usage,
latency and throughput. No NPU or server validation has been performed.

AI assistance: Codex.
