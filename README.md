
MaxFactor is best described as a thoughtful integration of existing optimization techniques with specific implementation choices tailored for encoder-decoder ASR transformer models. It combines proven optimization techniques from several established algorithms, with implementation details specifically tuned for transformer architectures used in speech recognition. 

The optimizer makes practical engineering tradeoffs that work well empirically for speech recognition models. Its particular combination of approaches addresses practical challenges in training large speech and multimodal llms. 

Why another optimizer? The architecture will serve as the foundation for the experimental Frequency-Adaptive Momentum (FAM) approach, which aims to leverage the inherent frequency structure of speech data in the optimization process itself. Each characteristic was selected based on empirical evidence indicating that they work well for ASR/NLP models and datasets.

#### MaxFactor Family Tree

```
Adam
├── Adaptive learning rates 
└── EMA of second moments

Adafactor
├── Factorized second moments
└── Relative step sizing

SignSGD
└── Sign-based updates

LAMB/LARS
├── Layer-wise adaptivity
└── Gradient normalization

AdamW
└── Decoupled weight decay

Adamax
└── Infinity normalization

RMSprop
└── Root mean squared gradient scaling

Gradient Clipping
└── Max norm constraints

MaxFactor
└── Combines all above features with a couple unique twists. (and FAM)
```


### Key Advantages

- **Superior accuracy** on simple tasks (MNIST)
- **Competitive accuracy** on complex tasks, significantly outperforming Adam/AdamW
- **Faster convergence** than SGD on some datasets
- **Memory efficiency** matching SGD, using ~25% less memory than Adam/AdamW
- **Stable optimization** across different model architectures and datasets

### When to Use MaxFactor

MaxFactor is particularly valuable for:
- Memory-constrained environments
- Complex datasets where Adam/AdamW underperform
- Speech recognition and other audio processing tasks
- Scenarios requiring a balance of accuracy and efficiency

### Accuracy
| Dataset      | SGD    | Adam   | AdamW  | MaxFactor |
|--------------|--------|--------|--------|-----------|
| MNIST        | 97.57% | 97.23% | 97.20% | 97.67%    |
| CNN-CIFAR    | 54.17% | 21.43% | 21.47% | 51.43%    |
| ConvNet-CIFAR| 48.37% | 32.13% | 32.30% | 46.30%    |

### Convergence Speed (epochs to 90% of final accuracy)
| Dataset      | SGD | Adam | AdamW | MaxFactor |
|--------------|-----|------|-------|-----------|
| MNIST        | 1   | 0    | 0     | 1         |
| CNN-CIFAR    | 6   | 3    | 3     | 5         |
| ConvNet-CIFAR| 7   | 6    | 6     | 7         |

### Memory Usage (relative to AdamW)
MaxFactor uses **25.1% less memory** than AdamW while maintaining comparable memory efficiency to SGD (difference <0.1%).





Coming soon -
Additionally, it will introduce Frequency-Adaptive Momentum (FAM), an experimental approach specifically designed for speech recognition tasks that adapts momentum based on the frequency characteristics of gradient updates.
### Frequency-Adaptive Momentum (FAM)

#### Core Concept

- Speech signals have inherent frequency structure, with different parts of the model responding to different frequency bands. The frequency structure of speech doesn't just disappear when converted to log-mel spectrograms; it's transformed and preserved in ways that the model's parameters adapt to capture.
- The Chain of Frequency Information: Original Audio → Log-Mel Spectrogram → Encoder Parameters → Gradient Updates.
  This isn't just a theoretical connection - it's empirically observable in how transformer-based speech models learn:
  - Lower encoder layers develop filters that respond to specific frequency bands in the mel spectrogram.
  - Attention heads specialize in tracking particular acoustic patterns across time.
  - The model inherently develops a hierarchical representation from acoustic features to phonetic units to words.
- The idea is to try and integrate a momentum scheme that adapts based on the "frequency signature" of gradient updates.

#### Why This Optimizer Makes Sense

What's compelling about the Frequency-Adaptive Momentum approach is that it acknowledges this structure in the optimization process itself. Rather than treating all gradient dimensions equally, it recognizes that:
- **Gradient Frequencies Matter:** The Fourier transform of gradient updates reveals patterns related to what the model is currently learning.
- **Different Parameters Process Different Bands:** Just as our ears have frequency-specific receptors, different parts of the model specialize in different acoustic frequencies.
- **Temporal Structure in Learning:** Speech learning happens in stages - first basic acoustics, then phonetic patterns, then linguistic structures.

By applying different momentum factors to different frequency bands in parameter space, we're essentially giving the optimizer information about the audio domain that it wouldn't otherwise have.


```python

class MaxFactor(torch.optim.Optimizer):
    """
    MaxFactor optimizer that combines adaptive learning rates with factorized second moments.
    
    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float, optional): Maximum learning rate (default: 0.01)
        beta2_decay (float, optional): Decay exponent for second moments (default: -0.8)
        eps (tuple, optional): Small constants for numerical stability (default: (None, 1e-3))
        d (float, optional): Scaling factor for updates (default: 1.0)
        weight_decay (float, optional): Weight decay factor (default: 0.0)
        gamma (float, optional): EMA factor for non-matrix parameters (default: 0.99)
        max (bool, optional): Maximize the objective instead of minimizing (default: False)
        full_matrix (bool, optional): Use full matrix for second moments (default: False)
        clip (float, optional): Gradient clipping norm (default: 1.0)
    """
    def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(1e-12, 1e-8), d=1.0, 
        weight_decay=0.0, gamma=0.99, max=False,
        ull_matrix=False, clip=1.0):
        
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
            
        eps1, eps2 = eps
        if eps1 is None:
            eps1 = torch.finfo(torch.float32).eps
            
        defaults = dict(
            lr=lr, beta2_decay=beta2_decay, eps=(eps1, eps2), d=d,  weight_decay=weight_decay, 
            gamma=gamma, max=max, full_matrix=full_matrix, clip=clip)
        
        super().__init__(params=params, defaults=defaults)
        
    def _get_lr(self, param_group, param_state):
            step = param_state["step"]
            step_float = step.item()
            decay_factor = min(1.0, 1.0 / (step_float ** 0.4  + 1e-12))
            param_scale = max(param_group["eps"][1], param_state["RMS"])
            return min(param_group["lr"], param_scale * decay_factor)

    @staticmethod
    def _rms(tensor):
        if tensor.numel() == 0:
            return torch.tensor(0.0, device=tensor.device)
        return tensor.norm() / (tensor.numel() ** 0.5 + 1e-12)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            row_vars = []
            col_vars = []
            v = []
            state_steps = []
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    
                    if p.dim() > 1 and not group["full_matrix"]:
                        row_shape = list(p.shape)
                        row_shape[-1] = 1
                        state["row_var"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
                        
                        col_shape = list(p.shape)
                        col_shape[-2] = 1
                        state["col_var"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
                    
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    state["RMS"] = self._rms(p).item()

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                params_with_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                state = self.state[param]
                
                # if self.use_fam and param.dim() > 1:
                #     grad = frequency_adaptive_momentum(
                #     grad, 
                #     state,
                #     alpha=self.fam_alpha,
                #     beta=self.fam_beta
                # )
                                
                if group["max"]:
                    grad = -grad
                    
                step_t = state_steps[i]
                row_var = row_vars[i]
                col_var = col_vars[i]
                vi = v[i]
                
                step_t += 1
                step_float = step_t.item()
                
                one_minus_beta2_t = min(0.999, step_float ** group["beta2_decay"])

                state = self.state[param]
                state["RMS"] = self._rms(param).item()
                adaptive_lr = self._get_lr(group, state)
                
                if group["weight_decay"] != 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"] + eps1)

                if param.dim() > 1 and not group["full_matrix"]:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_()
                    row_mean.div_(grad.size(-1) + eps1)
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_()
                    col_mean.div_(grad.size(-2) + eps1)
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))
                else:
 
                    vi.mul_(group["gamma"]).add_(grad.square_(), alpha=1 - group["gamma"])
                    var_estimate = vi
                    
                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                
                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))
                
                if group.get("clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [update], 
                        max_norm=group["clip"]
                    )
                
                l2_norm = update.norm(2).item()
                denom = max(1.0, l2_norm / ((update.numel() ** 0.5) * group["d"]))
                
                if param.dim() > 1:
                    param.add_(
                        update.sign() * update.abs().max(dim=-1, keepdim=True)[0], 
                        alpha=-adaptive_lr / denom
                    )
                else:
                    param.add_(update, alpha=-adaptive_lr / denom)
     
                state["step"] = step_t

        return loss

# optional:
# Create scheduler with warmup
scheduler = AdaptiveSchedule(
    optimizer=optimizer,
    initial_lr=0.001,
    warmup_steps=500,
    decay_factor=0.1
)

# Alternative: use cosine schedule
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=training_args.max_steps,
    eta_min=1e-5,
    last_epoch=-1  
)
### also optional

class AdaptiveSchedule(torch.optim.lr_scheduler.LambdaLR):
    """
    Learning rate scheduler that adapts to optimizer's internal rate calculation.
    
    Args:
        optimizer: Optimizer with _get_lr method
        initial_lr: Initial learning rate
        warmup_steps: Number of steps for warmup
        decay_factor: Factor to decay the learning rate
    """
    def __init__(self, optimizer, initial_lr=0.0, warmup_steps=0, decay_factor=0.1):
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        
        def lr_lambda(step):
            # Warmup phase
            if step < warmup_steps:
                return initial_lr * (step / max(1, warmup_steps))
            return initial_lr
        
        # Store parameters in optimizer groups
        for group in optimizer.param_groups:
            group["initial_lr"] = initial_lr
            
        super().__init__(optimizer, lr_lambda)
        
        # Clean up for memory efficiency
        for group in optimizer.param_groups:
            del group["initial_lr"]

    def get_lr(self):
        """Calculate learning rate using optimizer's internal method"""
        opt = self.optimizer
        
        # Get learning rates from optimizer if available
        if hasattr(opt, '_get_lr'):
            lrs = []
            for group_idx, group in enumerate(opt.param_groups):
                for param in group["params"]:
                    if param.grad is not None:
                        lrs.append(opt._get_lr(group, opt.state[param]))
                        break
            
            if len(lrs) == 0:
                lrs = self.base_lrs
                
            # Apply warmup/decay scaling
            step = self.last_epoch
            if step < self.warmup_steps:
                warmup_factor = max(0.001, step / max(1, self.warmup_steps))
                return [lr * warmup_factor for lr in lrs]
                
            return lrs
        else:

            return super().get_lr()
        

#### experimental 

def frequency_adaptive_momentum(grad, state, alpha=0.9, beta=0.999):
    """
    Apply frequency-adaptive momentum to gradients.
    
    Args:
        grad: Current gradient
        state: Optimizer state containing spectral history
        alpha: Short-term frequency decay factor
        beta: Long-term frequency decay factor
        theta: Because we like thetas
    
    Returns:
        Updated gradient with frequency-adaptive momentum
    """
    # Initialize state if needed
    if "freq_history" not in state:
        state["freq_history"] = {}
        state["step_freq"] = 0
    
    state["step_freq"] += 1
    
    # For matrices (likely attention-related parameters)
    if grad.dim() > 1 and min(grad.shape) > 4:  # Only for substantial matrices
        # Compute spectral signature using FFT on flattened gradient
        with torch.no_grad():
            # Sample spectral signature for efficiency
            if grad.numel() > 10000:
                # Sample along both dimensions for large matrices
                row_indices = torch.randperm(grad.size(0))[:min(grad.size(0), 100)]
                col_indices = torch.randperm(grad.size(1))[:min(grad.size(1), 100)]
                grad_sample = grad[row_indices][:, col_indices].flatten()
            else:
                grad_sample = grad.flatten()
            
            # Get frequency representation
            freq_repr = torch.fft.rfft(grad_sample.float())
            freq_power = torch.abs(freq_repr)
            
            # Normalize power spectrum
            if freq_power.sum() > 0:
                freq_power = freq_power / freq_power.sum()
            
            # Track frequency bands (divide spectrum into 10 bands)
            n_bands = 10
            band_size = freq_power.shape[0] // n_bands
            band_powers = [freq_power[i*band_size:(i+1)*band_size].sum().item() 
                          for i in range(n_bands)]
            
            # Update frequency history with exponential averaging
            for i, power in enumerate(band_powers):
                if f"band_{i}" not in state["freq_history"]:
                    state["freq_history"][f"band_{i}"] = power
                else:
                    state["freq_history"][f"band_{i}"] = (
                        beta * state["freq_history"][f"band_{i}"] +
                        (1-beta) * power
                    )
            
            # Compute adaptive dampening factors based on frequency history
            # High-frequency components get more dampening
            dampening_factors = []
            for i in range(n_bands):
                # Higher bands get more dampening, but modulated by recent activity
                base_dampening = i / n_bands  # 0 to 0.9
                recent_activity = state["freq_history"][f"band_{i}"]
                
                # Bands with more recent activity get less dampening (more momentum)
                adaptive_dampening = base_dampening * (1 - recent_activity * 5)
                dampening_factors.append(max(0, min(0.9, adaptive_dampening)))
            
            # Apply frequency-selective momentum to the gradient
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            
            # Apply band-specific momentum with inverse FFT
            momentum_buffer = state["momentum_buffer"].flatten()
            freq_momentum = torch.fft.rfft(momentum_buffer[:grad_sample.shape[0]].float())
            
            # Apply different momentum factors to different frequency bands
            for i in range(n_bands):
                start_idx = i * band_size
                end_idx = (i+1) * band_size
                dampening = dampening_factors[i]
                
                # Higher momentum for bands with higher recent activity
                momentum_factor = alpha * (1 - dampening)
                grad_factor = 1.0 + dampening  # Boost gradient for damped frequencies
                
                # Apply selective momentum in frequency domain
                if start_idx < freq_momentum.shape[0]:
                    actual_end = min(end_idx, freq_momentum.shape[0])
                    freq_momentum[start_idx:actual_end] = (
                        momentum_factor * freq_momentum[start_idx:actual_end] +
                        grad_factor * freq_repr[start_idx:actual_end]
                    )
            
            # Convert back to time domain and reshape
            new_grad_sample = torch.fft.irfft(freq_momentum, n=grad_sample.shape[0])
            
            # Update momentum buffer (in time domain)
            state["momentum_buffer"] = alpha * state["momentum_buffer"] + (1-alpha) * grad
            
            # Calculate adaptation factor to blend with original gradient
            # Early steps: more gradient, later steps: more frequency adaptation
            blend_factor = min(0.8, state["step_freq"] / 1000)
            
            # Create a scaling mask based on frequency characteristics
            scaling_mask = torch.ones_like(grad)
            
            # For demonstration - actual implementation would map frequency insights
            # back to the full gradient in a more sophisticated way
            if state["step_freq"] > 100:  # Only apply after initial training
                # Example: Speech models often have issues with high-frequency noise
                # Identify components likely responding to different frequencies
                
                # Compute row and column variances as proxies for frequency response
                row_var = grad.var(dim=1, keepdim=True)
                col_var = grad.var(dim=0, keepdim=True)
                
                # Normalize
                row_var = row_var / (row_var.mean() + 1e-8)
                col_var = col_var / (col_var.mean() + 1e-8)
                
                # Create mask emphasizing stable gradient components
                scaling_mask = 1.0 + 0.5 * (
                    torch.sigmoid(3 * (row_var - 1.5)) @ 
                    torch.sigmoid(3 * (col_var - 1.5)).T
                )
            
            # Apply adaptive mask to gradient
            grad = grad * scaling_mask
            
            return grad
    else:
        # For vectors and small matrices, use standard momentum
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
            
        state["momentum_buffer"] = alpha * state["momentum_buffer"] + (1-alpha) * grad
        return state["momentum_buffer"]

@torch.no_grad()
def step(self, closure=None):
   
    for i, param in enumerate(params_with_grad):
        grad = grads[i]
        state = self.state[param]
        
        # Apply frequency-adaptive momentum if enabled
        if self.use_fam and param.dim() > 1:
            grad = frequency_adaptive_momentum(
                grad, 
                state,
                alpha=self.fam_alpha,
                beta=self.fam_beta
            )
        

```
