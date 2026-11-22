# Model cards

Two small project-trained checkpoints are included for demonstration without the
raw dataset. Both use PyTorch and event-only observations. Load with
`torch.load(..., weights_only=True)`; hashes are in SHA256SUMS.

| File | Model | Parameters | Selected epoch |
|---|---|---:|---:|
| snn.pt | Conv12-LIF-Conv24-LIF-linear3 | 7,839 | 22 |
| cnn.pt | Stack10-Conv12-ReLU-Conv24-ReLU-linear3 | 10,239 | 21 |

Inputs: uint16 counts `[5,2,128,128]`, five 20 ms bins, ON/OFF. Shared deterministic
encoding: 4x4 max pooling followed by binary occupancy, giving `[5,2,32,32]`.
Labels: LEFT, STRAIGHT, RIGHT. No robot pose, map or goal bearing is an input.
SNN hidden states reset per window; its output readout is continuous, not spiking.
The CNN stacks the five time bins as ten channels and has no internal spikes.

Training: 1,432 corrected-imitation examples, three map variants, Adam 0.001,
50 epochs, batch 32, seed 7, class-weighted cross entropy. Selection: validation
macro-F1 on 197 examples from one other map. The same training/validation data
and procedure are used for both models. Test: 288 examples from two held-out maps.
No checkpoint was retuned after the final test protocol was opened.

The SNN scores better on held-out recordings; the CNN is faster on CPU and wins
more of the eight closed-loop test episodes. The SNN's 4/4 development-map success
does not generalize to these held-out maps. See the root README for final results;
benchmark scripts regenerate detailed reports locally. These are simulation
research models, not validated real-robot pilots.

No real DVS, neuromorphic chip, energy measurement or snnTorch was used. Synthetic
events come from the vendored v2e core, without noise or learned interpolation.
These weights and the dataset were created for this project; no third-party
pretrained weights are redistributed.
