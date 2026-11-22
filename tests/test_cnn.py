import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from neuroavoid.cnn import SmallCNN, model_from_config


@unittest.skipUnless(HAS_TORCH, "Utiliser l'environnement ML")
class CNNChecks(unittest.TestCase):
    def test_shared_input_and_no_fictitious_spikes(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        model = SmallCNN()
        x = (torch.rand(2, 5, 2, 32, 32) > 0.9).float()
        logits, stats = model(x)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertIsNone(stats["hidden_spikes"])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10239)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
        self.assertGreater(model.features[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.equal(logits, model(x)[0]))

    def test_factory_preserves_existing_snn_checkpoints(self):
        from neuroavoid.snn import SmallSNN
        self.assertIsInstance(model_from_config({}), SmallSNN)
        self.assertIsInstance(model_from_config({"model_type": "cnn"}), SmallCNN)
