# MaxFactor Optimizer Summary

`MaxFactor` is a custom PyTorch optimizer with adaptive learning rates and specialized handling for matrix parameters. Key features include:

## Parameters
- `lr`: Base learning rate (default: 0.025)
- `beta2_decay`: Controls how EMA decay changes with steps (default: -0.8)
- `eps`: Small constants to prevent numerical instability (default: (1e-10, 1e-4))
- `weight_decay`: L2 regularization (default: 0.025)
- `gamma`: EMA decay factor (default: 0.99)
- `max`: When True, negates gradients for maximization (default: False)
- `min_lr`: Minimum learning rate (default: 1e-7)

## Key Characteristics
- **Adaptive learning rates** based on parameter norms and training step
- **Specialized matrix handling**:
  - Separate row and column variance estimates for 2D parameters
  - Matrix updates use sign-based scaling with max values
- **Vector handling** uses EMA of squared gradients (similar to RMSprop)
- **Update normalization** using infinity norm
- **Dynamic beta2** that changes with step count according to beta2_decay
- **Automatic learning rate annealing** that decreases with square root of step count
- **Parameter-specific updates** scaled by parameter norms

This optimizer combines elements from several optimization techniques with specialized matrix handling that could be beneficial for asr/nlp neural network architectures.


