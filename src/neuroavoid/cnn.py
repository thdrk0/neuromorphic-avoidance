"""Conventional CNN on the same five encoded event bins as the SNN."""

from torch import nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(10, 12, 5, stride=2, padding=2), nn.ReLU(),
                                      nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.ReLU())
        self.readout = nn.Linear(24 * 8 * 8, 3)

    def forward(self, inputs):
        if inputs.ndim != 5 or tuple(inputs.shape[1:]) != (5, 2, 32, 32):
            raise ValueError("Entree CNN : [batch,5,2,32,32]")
        logits = self.readout(self.features(inputs.flatten(1, 2)).flatten(1))
        return logits, {"hidden_spikes": None, "input_spikes": inputs.detach().flatten(1).sum(1)}


def model_from_config(config):
    if config.get("model_type", "snn") == "cnn":
        return SmallCNN()
    from neuroavoid.snn import SmallSNN
    return SmallSNN(beta=config.get("beta", 0.8), threshold=config.get("threshold", 0.5))
