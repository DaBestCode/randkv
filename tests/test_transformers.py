import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from randkv import RandKVConfig
from randkv.transformers import RandKVCache, RandKVDynamicLayer, generate


class RandKVDynamicLayerTests(unittest.TestCase):
    @staticmethod
    def states(start: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.arange(start, start + length, dtype=torch.float32)
        values = values.view(1, 1, length, 1).expand(1, 4, length, 2).clone()
        return values, values + 1000

    def test_compacts_independently_after_prefill(self) -> None:
        layer = RandKVDynamicLayer(
            layer_idx=0,
            prompt_length=4,
            randkv_config=RandKVConfig(budget=8, buffer_size=2, seed=7),
            request_id="request-1",
        )
        prefill = self.states(0, 12)
        keys, _ = layer.update(*prefill)
        self.assertEqual(keys.shape[-2], 12)
        self.assertEqual(layer.get_mask_sizes(1), (10, 0))

        token = self.states(12, 1)
        keys, values = layer.update(*token)

        self.assertEqual(keys.shape, (1, 4, 10, 2))
        self.assertEqual(values.shape, keys.shape)
        self.assertEqual(layer.absolute_length, 13)
        self.assertEqual(layer.get_seq_length(), 10)
        self.assertEqual(len(set(layer.positions)), 4)
        for positions in layer.positions:
            self.assertTrue(set(range(4)).issubset(positions))
            self.assertTrue({11, 12}.issubset(positions))

    def test_waits_for_recent_buffer_before_next_eviction(self) -> None:
        layer = RandKVDynamicLayer(
            layer_idx=0,
            prompt_length=2,
            randkv_config=RandKVConfig(budget=6, buffer_size=2),
            request_id="request-1",
        )
        layer.update(*self.states(0, 9))
        layer.update(*self.states(9, 1))
        self.assertEqual(layer.get_seq_length(), 8)

        layer.update(*self.states(10, 1))
        self.assertEqual(layer.get_seq_length(), 9)
        self.assertEqual(layer.get_mask_sizes(1), (8, 0))

        layer.update(*self.states(11, 1))
        self.assertEqual(layer.get_seq_length(), 8)
        self.assertEqual(layer.eviction_index, 2)

    def test_rejects_batching(self) -> None:
        layer = RandKVDynamicLayer(
            layer_idx=0,
            prompt_length=2,
            randkv_config=RandKVConfig(budget=6),
            request_id="request-1",
        )
        states = torch.zeros(2, 4, 3, 8)
        with self.assertRaisesRegex(NotImplementedError, "batch size 1"):
            layer.update(states, states)


class RandKVCacheGenerationTests(unittest.TestCase):
    @staticmethod
    def model() -> Qwen3ForCausalLM:
        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
        return Qwen3ForCausalLM(config).eval()

    def test_matches_dense_generation_before_eviction(self) -> None:
        model = self.model()
        inputs = torch.tensor([[1, 5, 7, 9]])
        dense = model.generate(inputs, max_new_tokens=4, do_sample=False)
        cache = RandKVCache(
            model_config=model.config,
            prompt_length=inputs.shape[-1],
            randkv_config=RandKVConfig(budget=32, buffer_size=2),
        )

        actual = model.generate(
            inputs, max_new_tokens=4, do_sample=False, past_key_values=cache
        )

        torch.testing.assert_close(actual, dense)
        self.assertEqual(cache.get_seq_length(), 7)
        self.assertEqual(cache.physical_seq_length(), 7)

    def test_generates_past_physical_cache_budget(self) -> None:
        model = self.model()
        inputs = torch.tensor([[1, 5, 7, 9]])
        cache = RandKVCache(
            model_config=model.config,
            prompt_length=inputs.shape[-1],
            randkv_config=RandKVConfig(budget=6, buffer_size=2, seed=42),
        )

        output = model.generate(
            inputs, max_new_tokens=12, do_sample=False, past_key_values=cache
        )

        self.assertEqual(output.shape[-1], 16)
        self.assertEqual(cache.get_seq_length(), 15)
        self.assertLessEqual(cache.physical_seq_length(), 10)
        stats = cache.stats()
        self.assertEqual(stats.absolute_tokens_seen, 15)
        self.assertEqual(stats.eviction_rounds_per_layer, (4, 4))
        self.assertEqual(stats.physical_tokens_per_layer, (8, 8))
        self.assertGreater(stats.positions_evicted_total, 0)
        for layer in cache.layers:
            self.assertIsInstance(layer, RandKVDynamicLayer)
            self.assertTrue(set(range(4)).issubset(layer.positions[0]))

    def test_generate_helper_matches_dense_before_eviction(self) -> None:
        model = self.model()
        inputs = torch.tensor([[1, 5, 7, 9]])
        dense = model.generate(inputs, max_new_tokens=4, do_sample=False)

        actual = generate(
            model,
            inputs,
            randkv_config=RandKVConfig(budget=32),
            max_new_tokens=4,
            do_sample=False,
        )

        torch.testing.assert_close(actual, dense)

    def test_generate_helper_rejects_padding(self) -> None:
        model = self.model()
        inputs = torch.tensor([[0, 1, 5, 7]])

        with self.assertRaisesRegex(NotImplementedError, "padded inputs"):
            generate(model, inputs, attention_mask=torch.tensor([[0, 1, 1, 1]]))

    def test_cache_rejects_batch_expansion(self) -> None:
        model = self.model()
        cache = RandKVCache(model_config=model.config, prompt_length=4)

        with self.assertRaisesRegex(NotImplementedError, "multiple return"):
            cache.batch_repeat_interleave(2)


if __name__ == "__main__":
    unittest.main()
