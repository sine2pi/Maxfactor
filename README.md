
**MaxFactor: A Robust Foundation

MaxFactor is best described as a thoughtful integration of existing optimization techniques with specific implementation choices tailored for encoder-decoder ASR transformer models. It combines proven optimization methods from several established algorithms with implementation details specifically fine-tuned for transformer architectures used in speech recognition.

While MaxFactor is an effective optimizer, making practical engineering tradeoffs that work well empirically for speech recognition models, it's purpose is to serve as the backbone for the Frequency-Adaptive Momentum (FAM) approach. FAM aims to harness the natural frequency structure of speech data within the optimization process. Each attribute of MaxFactor (core) has been chosen based on empirical evidence demonstrating its effectiveness for ASR and NLP models and datasets. As our test data grows to encompass datasets and models that better reflect our target group, we expect MaxFactor to significantly outperform other optimizers in its domain.

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

---

**Coming Soon: Frequency-Adaptive Momentum (FAM)**

An experimental approach specifically designed for speech recognition tasks, FAM adapts momentum based on the frequency characteristics of gradient updates.

### Frequency-Adaptive Momentum (FAM)

#### Core Concept

- Speech signals possess an inherent frequency structure, with different parts of the model responding to various frequency bands. This frequency structure remains preserved, albeit transformed, when converted to log-mel spectrograms, with model parameters adapting to capture this structure.
- The Chain of Frequency Information: Original Audio → Log-Mel Spectrogram → Encoder Parameters → Gradient Updates.
- Empirical observations reveal that transformer-based speech models develop:
  - Lower encoder layers with filters responsive to specific frequency bands in the mel spectrogram.
  - Attention heads tracking particular acoustic patterns over time.
  - A hierarchical representation from acoustic features to phonetic units to words.
- FAM aims to integrate a momentum scheme that adapts based on the "frequency signature" of gradient updates.

#### Why This Optimizer Makes Sense

FAM acknowledges the frequency structure within the optimization process itself, recognizing that:
- **Gradient Frequencies Matter:** The Fourier transform of gradient updates reveals patterns linked to the model's current learning phase.
- **Different Parameters Process Different Bands:** Similar to how our ears have frequency-specific receptors, different parts of the model specialize in various acoustic frequencies.
- **Temporal Structure in Learning:** Speech learning progresses through stages - from basic acoustics to phonetic patterns to linguistic structures.

By applying distinct momentum factors to different frequency bands in parameter space, FAM provides the optimizer with domain-specific audio information that it otherwise wouldn't have.

---
MaxFactor family tree:
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
└── Combines all above features and eventually FAM 
```


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

## Use any scheduler you like such as:

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=max_steps,
    eta_min=0.0,
    last_epoch=-1  
)

## Dummy scheduler for hugging face trainer : scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)        

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
