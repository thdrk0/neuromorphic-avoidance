import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import numpy as np
    import torch
    from neuroavoid.snn import SmallSNN, SurrogateSpike, encode_events, lif_step
    from neuroavoid.snn_cli import classification_metrics


@unittest.skipUnless(HAS_TORCH, "Utiliser .venv-v2e pour les tests SNN")
class SNNChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_spikes_are_binary_with_finite_surrogate_gradient(self):
        voltage = torch.tensor([-0.2, 0.0, 0.2], requires_grad=True)
        spikes = SurrogateSpike.apply(voltage)
        self.assertEqual(spikes.tolist(), [0, 1, 1])
        spikes.sum().backward()
        self.assertTrue(torch.isfinite(voltage.grad).all())
        self.assertTrue((voltage.grad > 0).all())

    def test_neuron_integrates_then_resets(self):
        current = torch.tensor([0.3])
        spike, state = lif_step(current, torch.zeros(1))
        self.assertEqual(spike.item(), 0)
        spike, state = lif_step(current, state)
        self.assertEqual(spike.item(), 1)
        self.assertAlmostEqual(state.item(), 0.04, places=6)

    def test_encoding_keeps_time_polarity_and_spatial_blocks(self):
        counts = torch.zeros(1, 5, 2, 128, 128)
        counts[0, 3, 1, 9, 14] = 7
        encoded = encode_events(counts)
        self.assertEqual(tuple(encoded.shape), (1, 5, 2, 32, 32))
        self.assertEqual(encoded[0, 3, 1, 2, 3].item(), 1)
        self.assertEqual(encoded.sum().item(), 1)

    def test_model_resets_between_windows_and_backpropagates(self):
        model = SmallSNN()
        inputs = (torch.rand(2, 5, 2, 32, 32) > 0.8).float()
        logits, stats = model(inputs)
        repeated, _ = model(inputs)
        self.assertTrue(torch.equal(logits, repeated))
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.all(stats["hidden_spikes"] >= 0))
        self.assertTrue(torch.all(stats["hidden_spikes"] <= 5 * (12 * 16 * 16 + 24 * 8 * 8)))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
        self.assertGreater(model.conv1.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_metrics_do_not_hide_majority_class(self):
        metrics = classification_metrics([0, 1, 2], [1, 1, 1])
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 6)
        self.assertEqual(metrics["confusion_matrix"], [[0, 1, 0], [0, 1, 0], [0, 1, 0]])
