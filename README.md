# MaxFactor Optimizer Summary

`MaxFactor` is a custom PyTorch optimizer with adaptive learning rates and specialized handling for matrix parameters.

## Key Characteristics
- Adaptive learning rates based on parameter norms and training step
- Specialized matrix handling:
  - Separate row and column variance estimates for 2D parameters
  - Matrix updates use sign-based scaling with max values
- Vector handling uses EMA of squared gradients (similar to RMSprop)
- Update normalization using infinity norm
- Dynamic beta2 that changes with step count according to beta2_decay
- Automatic learning rate annealing that decreases with square root of step count
- Parameter specific updates scaled by parameter norms

This optimizer combines elements from several optimization techniques with specialized matrix handling that could be beneficial for asr/nlp neural network architectures.

AdamW

<img width="640" alt="adamw" src="https://github.com/user-attachments/assets/068e4b2a-b0f3-47f1-8c28-21d2b6b968d3" />

MaxFActor @ 1/2 vram usage

<img width="640" alt="totd" src="https://github.com/user-attachments/assets/f2bb09ea-566c-430e-bd09-0797af37a855" />


