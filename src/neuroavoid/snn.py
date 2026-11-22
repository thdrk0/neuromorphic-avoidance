"""Small convolutional LIF SNN with a continuous three-class readout."""

import torch
from torch import nn
from torch.nn import functional as F


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (voltage,) = ctx.saved_tensors
        return gradient / (1 + 10 * voltage.abs()).square()


def lif_step(current, membrane, beta=0.8, threshold=0.5):
    voltage = beta * membrane + current
    spikes = SurrogateSpike.apply(voltage - threshold)
    return spikes, voltage - spikes.detach() * threshold


def encode_events(counts):
    """128x128 ON/OFF counts -> 32x32 binary occupancy spikes per time bin.

    Fixed 4x4 spatial pooling, no fitted statistics or random Poisson sampling.
    Event multiplicity is deliberately discarded in this initial encoding.
    """
    if counts.ndim != 5 or tuple(counts.shape[-3:]) != (2, 128, 128):
        raise ValueError("Entree attendue : [batch,time,2,128,128]")
    batch, steps = counts.shape[:2]
    pooled = F.max_pool2d(counts.float().reshape(batch * steps, 2, 128, 128), 4)
    return (pooled > 0).float().reshape(batch, steps, 2, 32, 32)


class SmallSNN(nn.Module):
    def __init__(self, beta=0.8, threshold=0.5):
        super().__init__()
        self.beta, self.threshold = beta, threshold
        self.conv1 = nn.Conv2d(2, 12, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(12, 24, 3, stride=2, padding=1)
        self.readout = nn.Linear(24 * 8 * 8, 3)

    def forward(self, inputs):
        if inputs.ndim != 5 or tuple(inputs.shape[-3:]) != (2, 32, 32) or inputs.shape[1] < 1:
            raise ValueError("Spikes attendus : [batch,time,2,32,32]")
        # States are local: independent dataset windows never share membrane history.
        membrane1 = inputs.new_zeros((len(inputs), 12, 16, 16))
        membrane2 = inputs.new_zeros((len(inputs), 24, 8, 8))
        logits = inputs.new_zeros((len(inputs), 3))
        spike_count = inputs.new_zeros(len(inputs))
        for step in range(inputs.shape[1]):
            spikes1, membrane1 = lif_step(self.conv1(inputs[:, step]), membrane1, self.beta, self.threshold)
            spikes2, membrane2 = lif_step(self.conv2(spikes1), membrane2, self.beta, self.threshold)
            logits = logits + self.readout(spikes2.flatten(1))
            spike_count = spike_count + spikes1.detach().flatten(1).sum(1) + spikes2.detach().flatten(1).sum(1)
        return logits / inputs.shape[1], {"hidden_spikes": spike_count,
                                         "input_spikes": inputs.detach().flatten(1).sum(1)}
