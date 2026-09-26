# Zephyr

Zephyr contains the CNN and temporal network used by the
[Breathing from Video Challenge](https://github.com/AllenNeuralDynamics/breathing-codabench-challenge).
It includes the model, loss, channel encoding, and training augmentations.

The challenge repository owns video preprocessing, datasets, training runs,
checkpoint loading, inference, evaluation, and submission packaging. Install
that repository to use those workflows. Zephyr's source history was extracted
from its former `baseline-cnn-tcn` package.

## Install

```bash
pip install git+https://github.com/AllenNeuralDynamics/zephyr.git
```

The main imports are `zephyr.model.BreathingNet`,
`zephyr.model.BreathingLoss`, `zephyr.channels.ChannelSet`, and
`zephyr.augment.AugmentConfig`.
