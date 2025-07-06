
`MaxFactor` is a custom PyTorch optimizer with adaptive learning rates and specialized handling for matrix parameters. I wrote it for the model in the asr_model repository. I needed something that performs well but has a light memory foot print since I do everything from my laptop. 

Characteristics
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
