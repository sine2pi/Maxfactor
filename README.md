
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

AdamW

<img width="640" alt="adamw" src="https://github.com/user-attachments/assets/068e4b2a-b0f3-47f1-8c28-21d2b6b968d3" />

MaxFActor @ 1/2 vram usage

<img width="640" alt="totd" src="https://github.com/user-attachments/assets/f2bb09ea-566c-430e-bd09-0797af37a855" />




```python

  optimizer = MaxFactor(model.parameters(), lr=0.025, beta2_decay=-0.8, eps=(1e-10, 1e-7), d=1.0, 
               weight_decay=0.025, gamma=0.99, max=False)

  optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, eps=1e-8, weight_decay=0.01, betas=(0.9, 0.999), 
  amsgrad=False, foreach=False, fused=False)
  
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-7, last_epoch=-1)

```

https://huggingface.co/Sin2pi/Echo17/tensorboard?params=scalars
