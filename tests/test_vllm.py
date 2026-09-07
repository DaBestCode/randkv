import unittest

from randkv import RandKVConfig
from randkv.vllm import VLLMCompactionPlanner


class VLLMCompactionPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RandKVConfig(budget=4, buffer_size=2, seed=17)

    def test_returns_none_while_cache_is_within_capacity(self) -> None:
        planner = VLLMCompactionPlanner(self.config)
        positions = (tuple(range(6)),) * 2

        plan = planner.plan(
            positions,
            prompt_length=2,
            request_id="request-1",
            layer=0,
            eviction_index=0,
            absolute_tokens_seen=6,
            block_size=4,
        )

        self.assertIsNone(plan)

    def test_emits_independent_head_copy_offsets(self) -> None:
        planner = VLLMCompactionPlanner(self.config)
        source = tuple(range(12))

        plan = planner.plan(
            (source,) * 3,
            prompt_length=2,
            request_id="request-1",
            layer=4,
            eviction_index=0,
            absolute_tokens_seen=12,
            block_size=4,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.retained_tokens, 6)
        self.assertEqual(plan.evicted_tokens_per_head, 6)
        self.assertEqual(plan.destination_blocks, 2)
        self.assertEqual(plan.num_kv_heads, 3)
        self.assertEqual(len(set(plan.retained_positions_by_head)), 3)
        for positions, offsets in zip(
            plan.retained_positions_by_head,
            plan.source_offsets_by_head,
            strict=True,
        ):
            self.assertEqual(positions[:2], (0, 1))
            self.assertEqual(positions[-2:], (10, 11))
            self.assertEqual(tuple(source[offset] for offset in offsets), positions)

    def test_plan_is_deterministic_across_planners(self) -> None:
        arguments = {
            "prompt_length": 2,
            "request_id": "request-9",
            "layer": 3,
            "eviction_index": 2,
            "absolute_tokens_seen": 20,
            "block_size": 16,
            "head_offset": 8,
        }
        positions = (tuple(range(20)),) * 2

        first = VLLMCompactionPlanner(self.config).plan(positions, **arguments)
        second = VLLMCompactionPlanner(self.config).plan(positions, **arguments)

        self.assertEqual(first, second)

    def test_global_head_offset_changes_the_draw(self) -> None:
        planner = VLLMCompactionPlanner(self.config)
        positions = (tuple(range(30)),)
        arguments = {
            "prompt_length": 2,
            "request_id": "request-1",
            "layer": 0,
            "eviction_index": 0,
            "absolute_tokens_seen": 30,
            "block_size": 16,
        }

        first_rank = planner.plan(positions, head_offset=0, **arguments)
        second_rank = planner.plan(positions, head_offset=1, **arguments)

        self.assertIsNotNone(first_rank)
        self.assertIsNotNone(second_rank)
        assert first_rank is not None and second_rank is not None
        self.assertNotEqual(
            first_rank.retained_positions_by_head,
            second_rank.retained_positions_by_head,
        )

    def test_supports_positions_retained_by_an_earlier_round(self) -> None:
        planner = VLLMCompactionPlanner(self.config)
        initial = tuple(range(12))
        first = planner.plan(
            (initial,) * 2,
            prompt_length=2,
            request_id="request-1",
            layer=0,
            eviction_index=0,
            absolute_tokens_seen=12,
            block_size=4,
        )
        assert first is not None
        extended = tuple(
            positions + (12, 13, 14) for positions in first.retained_positions_by_head
        )

        second = planner.plan(
            extended,
            prompt_length=2,
            request_id="request-1",
            layer=0,
            eviction_index=1,
            absolute_tokens_seen=15,
            block_size=4,
        )

        self.assertIsNotNone(second)
        assert second is not None
        for positions in second.retained_positions_by_head:
            self.assertEqual(positions[:2], (0, 1))
            self.assertEqual(positions[-2:], (13, 14))

    def test_rejects_unequal_physical_head_lengths(self) -> None:
        planner = VLLMCompactionPlanner(self.config)

        with self.assertRaisesRegex(ValueError, "same physical length"):
            planner.plan(
                (tuple(range(8)), tuple(range(7))),
                prompt_length=2,
                request_id="request-1",
                layer=0,
                eviction_index=0,
                absolute_tokens_seen=8,
                block_size=4,
            )

    def test_validates_state_even_when_no_compaction_is_due(self) -> None:
        planner = VLLMCompactionPlanner(self.config)

        with self.assertRaisesRegex(ValueError, "protected prompt"):
            planner.plan(
                ((0, 2, 3),),
                prompt_length=2,
                request_id="request-1",
                layer=0,
                eviction_index=0,
                absolute_tokens_seen=4,
                block_size=4,
            )


if __name__ == "__main__":
    unittest.main()
