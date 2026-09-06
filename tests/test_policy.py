import unittest

from randkv import KVPolicy, RandKVConfig, RandomEvictionPolicy


def accepts_policy(policy: KVPolicy) -> KVPolicy:
    return policy


class RandomEvictionPolicyTests(unittest.TestCase):
    def test_zero_config_defaults(self) -> None:
        policy = RandomEvictionPolicy()

        self.assertEqual(policy.config.budget, 2048)
        self.assertEqual(policy.config.buffer_size, 64)
        self.assertTrue(policy.config.protect_prompt)
        self.assertIs(accepts_policy(policy), policy)

    def test_does_not_evict_below_threshold(self) -> None:
        policy = RandomEvictionPolicy(RandKVConfig(budget=8, buffer_size=2))
        positions = tuple(range(10))

        self.assertEqual(policy.select(positions, prompt_length=3), positions)
        self.assertEqual(policy.stats().eviction_events, 0)

    def test_retains_prompt_recent_buffer_and_budget(self) -> None:
        policy = RandomEvictionPolicy(RandKVConfig(budget=8, buffer_size=2))

        retained = policy.select(range(20), prompt_length=3)

        self.assertEqual(len(retained), 10)
        self.assertTrue(set(range(3)).issubset(retained))
        self.assertTrue({18, 19}.issubset(retained))
        self.assertEqual(policy.stats().positions_evicted, 10)

    def test_selection_is_deterministic(self) -> None:
        config = RandKVConfig(budget=8, buffer_size=2, seed=42)
        first = RandomEvictionPolicy(config).select(
            range(30), prompt_length=3, layer=2, kv_head=1, request_id="req-7"
        )
        second = RandomEvictionPolicy(config).select(
            range(30), prompt_length=3, layer=2, kv_head=1, request_id="req-7"
        )

        self.assertEqual(first, second)

    def test_heads_receive_independent_draws(self) -> None:
        policy = RandomEvictionPolicy(RandKVConfig(budget=12, buffer_size=2, seed=42))

        heads = policy.select_heads(
            range(100), prompt_length=4, num_kv_heads=4, request_id="req-7"
        )

        self.assertEqual(len(set(heads)), 4)
        for retained in heads:
            self.assertTrue(set(range(4)).issubset(retained))
            self.assertEqual(len(retained), 14)

    def test_eviction_event_changes_the_draw(self) -> None:
        policy = RandomEvictionPolicy(RandKVConfig(budget=8, buffer_size=2))

        first = policy.select(range(30), prompt_length=3, eviction_index=0)
        second = policy.select(range(30), prompt_length=3, eviction_index=1)

        self.assertNotEqual(first, second)

    def test_rejects_prompt_larger_than_persistent_budget(self) -> None:
        policy = RandomEvictionPolicy(RandKVConfig(budget=4, buffer_size=2))

        with self.assertRaisesRegex(ValueError, "prompt requires 5 cache slots"):
            policy.select(range(20), prompt_length=5)

    def test_rejects_cache_with_missing_prompt_position(self) -> None:
        policy = RandomEvictionPolicy(RandKVConfig(budget=8, buffer_size=2))

        with self.assertRaisesRegex(
            ValueError, "cache is missing protected prompt positions: 1"
        ):
            policy.select([0, 2, *range(3, 20)], prompt_length=3)

    def test_rejects_unstable_request_identifier_type(self) -> None:
        policy = RandomEvictionPolicy(RandKVConfig(budget=8, buffer_size=2))

        with self.assertRaisesRegex(TypeError, "request_id must be"):
            policy.select(  # type: ignore[arg-type]
                range(20), prompt_length=3, request_id=("request", 1)
            )

    def test_rejects_invalid_positions(self) -> None:
        policy = RandomEvictionPolicy()

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            policy.select([0, 2, 1], prompt_length=1)


if __name__ == "__main__":
    unittest.main()
