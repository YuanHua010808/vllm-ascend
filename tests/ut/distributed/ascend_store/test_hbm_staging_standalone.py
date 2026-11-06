"""Synthetic pointer round-trip test; no NPU runtime is required."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class StagingTest(unittest.TestCase):
    def test_round_trip_and_failed_get(self):
        path = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/hbm_staging.py"
        spec = importlib.util.spec_from_file_location("staging", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        memory = {10: b"ab", 20: b"cde"}

        def memcpy(dst, capacity, src, size, kind):
            memory[dst] = memory[src][:size]
            return 0

        runtime = SimpleNamespace(
            malloc=lambda size, policy: (100, 0), free=lambda ptr: 0, memcpy=memcpy
        )
        pending = []

        def memcpy_async(dst, capacity, src, size, kind, stream):
            pending.append((dst, capacity, src, size, kind))
            return 0

        def synchronize(stream):
            for args in pending:
                memcpy(*args)
            pending.clear()
            return 0

        runtime.create_stream = lambda: (1, 0)
        runtime.memcpy_async = memcpy_async
        runtime.synchronize_stream = synchronize
        runtime.destroy_stream = lambda stream: 0

        class Store:
            value = b""
            fail = False

            def register_buffer(self, ptr, size):
                return 0

            def unregister_buffer(self, ptr):
                return 0

            def batch_put_from_multi_buffers(self, keys, pointers, sizes, config):
                self.value = memory[100] + memory[102]
                return [0]

            def batch_get_into_multi_buffers(self, keys, pointers, sizes):
                memory[100] = self.value
                memory[102] = self.value[2:]
                return [-1] if self.fail else [5]

        store = Store()
        with patch.dict(sys.modules, {"acl": SimpleNamespace(rt=runtime)}):
            module.transfer_packed(store, ["key"], [[10, 20]], [[2, 3]], 5, config=object())
            self.assertEqual(store.value, b"abcde")
            memory[10], memory[20] = b"xx", b"yyy"
            module.transfer_packed(store, ["key"], [[10, 20]], [[2, 3]], 5)
            self.assertEqual((memory[10], memory[20]), (b"ab", b"cde"))
            store.fail = True
            memory[10] = b"xx"
            module.transfer_packed(store, ["key"], [[10, 20]], [[2, 3]], 5)
            self.assertEqual(memory[10], b"xx")
            store.fail = False
            memory[10], memory[20] = b"ab", b"cde"
            module.transfer_packed(store, ["key"], [[10, 20]], [[2, 3]], 5,
                                   config=object(), async_copy=True)
            self.assertEqual(store.value, b"abcde")
            memory[10], memory[20] = b"xx", b"yyy"
            module.transfer_packed(store, ["key"], [[10, 20]], [[2, 3]], 5, async_copy=True)
            self.assertEqual((memory[10], memory[20]), (b"ab", b"cde"))
            self.assertFalse(pending)
            with self.assertRaises(ValueError):
                module.transfer_packed(store, ["key"], [[10, 20]], [[2, 3]], 4)


if __name__ == "__main__":
    unittest.main()
