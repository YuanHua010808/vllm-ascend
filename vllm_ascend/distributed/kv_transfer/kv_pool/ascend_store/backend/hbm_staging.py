"""Bounded, byte-preserving HBM staging for Mooncake objects."""

def transfer_packed(store, keys, addrs, sizes, capacity, *, config=None):
    """Gather before PUT or scatter successful GETs; preserve object ordering."""
    import acl

    if not (len(keys) == len(addrs) == len(sizes)):
        raise ValueError("keys, addresses and sizes must have equal lengths")
    for pointers, lengths in zip(addrs, sizes):
        if len(pointers) != len(lengths) or any(n <= 0 for n in lengths):
            raise ValueError("Invalid scatter/gather layout")
        if sum(lengths) > capacity:
            raise ValueError("Mooncake object exceeds hbm_staging_bytes")
    if not keys:
        return []

    size_bytes = max(map(sum, sizes))
    base, rc = acl.rt.malloc(size_bytes, 0)
    if rc != 0:
        raise RuntimeError(f"Staging allocation failed: {rc}")
    rc = store.register_buffer(base, size_bytes)
    if rc != 0:
        acl.rt.free(base)
        raise RuntimeError(f"Staging registration failed: {rc}")

    def copy(pointers, lengths, *, gather):
        offset = 0
        for pointer, length in zip(pointers, lengths):
            src, dst = (pointer, base + offset) if gather else (base + offset, pointer)
            # ACL_MEMCPY_DEVICE_TO_DEVICE; synchronous before transport/reuse.
            rc = acl.rt.memcpy(dst, length, src, length, 3)
            if rc != 0:
                raise RuntimeError(f"HBM staging copy failed: {rc}")
            offset += length

    try:
        results = []
        for key, pointers, lengths in zip(keys, addrs, sizes):
            size = sum(lengths)
            if config is not None:
                copy(pointers, lengths, gather=True)
                result = store.batch_put_from_multi_buffers([key], [[base]], [[size]], config)
            else:
                result = store.batch_get_into_multi_buffers([key], [[base]], [[size]])
                if len(result) != 1:
                    raise RuntimeError("Invalid Mooncake GET result count")
                if result[0] >= 0:
                    copy(pointers, lengths, gather=False)
            if len(result) != 1:
                raise RuntimeError("Invalid Mooncake result count")
            results.extend(result)
        return results
    finally:
        rc = store.unregister_buffer(base)
        if rc != 0:
            # Keep the allocation alive if transport registration remains live.
            raise RuntimeError(f"Staging unregistration failed: {rc}")
        rc = acl.rt.free(base)
        if rc != 0:
            raise RuntimeError(f"Staging free failed: {rc}")
