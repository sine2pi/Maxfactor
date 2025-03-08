in progress...

MaxFactor is best described as a thoughtful integration of existing optimization techniques with specific implementation choices tailored for encoder-decoder ASR transformer models. 

It combines proven optimization methods from several established algorithms with implementation details specifically fine-tuned for transformer architectures used in speech recognition.

When MaxFactor might be a good alternative:
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

Memory Usage (relative to AdamW)

MaxFactor uses **25.1% less memory** than AdamW while maintaining comparable memory efficiency to SGD (difference <0.1%).

Each optimizer attribute was chosen based on empirical evidence demonstrating its effectiveness for ASR and NLP models and datasets. On it's own it's an effective optimizer, making practical engineering tradeoffs that work well empirically for speech recognition models, but it's purpose will be to serve as the backbone for the Frequency-Adaptive Momentum (FAM) approach that I'm experimenting with. FAM aims to harness the natural frequency structure of speech data within the optimization process itself. It's a work in progress. 

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

## internal lr adjustment removed for now
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
    # working - no inf norm
    def __init__(self, params, lr=0.025, beta2_decay=-0.8, eps=(1e-10, 1e-4), d=1.0, 
                 weight_decay=0.025, gamma=0.99, max=False, clip_threshold=1.0):
        
        defaults = dict(lr=lr, beta2_decay=beta2_decay, eps=eps, d=d, weight_decay=weight_decay, 
                        gamma=gamma, max=max, clip_threshold=clip_threshold)
        super().__init__(params=params, defaults=defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm() / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            clip_threshold = group.get("clip_threshold", 1.0)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"] = p.grad.new_zeros(row_shape)
                        state["col_var"] = p.grad.new_zeros(col_shape)
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
                row_var, col_var, vi = row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                step_t += 1
                step_float = step_t.item()
                
                beta2t = 1.0 - min(0.9, max(0.1, math.pow(step_float, group["beta2_decay"])))
                state["RMS"] = self._rms(param).item()
                rho_t = min(group["lr"], 1.0 / (step_float ** 0.5))
                alpha = max(eps2, state["RMS"]) * rho_t

                if group["weight_decay"] > 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])

                if grad.dim() > 1:
                    row_mean = (grad ** 2).mean(dim=-1, keepdim=True)
                    row_var.mul_(1-beta2t).add_(row_mean, alpha=beta2t)
                    col_mean = (grad ** 2).mean(dim=-2, keepdim=True)
                    col_var.mul_(1-beta2t).add_(col_mean, alpha=beta2t)
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_((grad ** 2), alpha=1 - group["gamma"])
                    var_estimate = vi

                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                update_norm = self._rms(update)
                if update_norm > 0 and clip_threshold > 0:
                    update.mul_(min(1.0, clip_threshold / (update_norm + eps1)))
                
                denom = max(1.0, update.norm() / ((update.numel() ** 0.5) * group["d"]))
                if param.dim() > 1:
                    max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                    param.add_(-alpha / denom * update.sign() * max_vals)
                else:
                    param.add_(-alpha / denom * update)              
                state["step"] = step_t
                
        return loss
    

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


## rough idea

import torch
from torch.optim.optimizer import Optimizer


class FAMOptimizer(torch.optim.Optimizer):
    """
    Frequency-Adaptive Momentum (FAM) optimizer
    
    Applies momentum with different factors based on frequency characteristics of gradients,
    particularly useful for speech recognition models.
    
    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float, optional): Learning rate (default: 1e-3)
        alpha (float, optional): Momentum factor (default: 0.9)
        beta (float, optional): Frequency history decay factor (default: 0.99)
        eps (float, optional): Term for numerical stability (default: 1e-8)
        weight_decay (float, optional): Weight decay factor (default: 0)
        n_bands (int, optional): Number of frequency bands to analyze (default: 8)
        fam_start_step (int, optional): Step to start applying FAM (default: 100)
        layer_boost (bool, optional): Whether to apply layer-specific boosts (default: True)
        min_size (int, optional): Minimum parameter size to apply FAM (default: 256)
        debug (bool, optional): Whether to collect debug information (default: False)
    """
    def __init__(self, params, lr=1e-3, alpha=0.9, beta=0.99, eps=1e-8,
                 weight_decay=0, n_bands=8, fam_start_step=100,
                 layer_boost=True, min_size=256, debug=False):
        defaults = dict(lr=lr, alpha=alpha, beta=beta, eps=eps,
                       weight_decay=weight_decay, n_bands=n_bands,
                       fam_start_step=fam_start_step, 
                       layer_boost=layer_boost, min_size=min_size)
        self.debug = debug
        self.debug_info = {} if debug else None
        super(FAMOptimizer, self).__init__(params, defaults)
        
        print(f"FAM Optimizer initialized with:")
        print(f"  lr={lr}, alpha={alpha}, beta={beta}, n_bands={n_bands}")
        print(f"  fam_start_step={fam_start_step}, min_size={min_size}")
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('FAMOptimizer does not support sparse gradients')
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['freq_history'] = {}
                    state['param_name'] = f"param_{p_idx}"
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                exp_avg = state['exp_avg']
                alpha = group['alpha']
                beta = group['beta']
                lr = group['lr']
                n_bands = group['n_bands']
                
                should_apply_fam = (
                    state['step'] > group['fam_start_step'] and
                    p.numel() > group['min_size']
                )
                
                if should_apply_fam:
                    try:
                        if p.numel() > 10000:
                            if p.dim() > 1:
                                row_indices = torch.randperm(p.size(0))[:min(p.size(0), 64)]
                                col_indices = torch.randperm(p.size(1))[:min(p.size(1), 64)]
                                grad_sample = grad[row_indices][:, col_indices].flatten()
                            else:
                                sample_idx = torch.randperm(p.numel())[:1000]
                                grad_sample = grad.flatten()[sample_idx]
                        else:
                            grad_sample = grad.flatten()
                        
                        freq_repr = torch.fft.rfft(grad_sample.float())
                        freq_power = torch.abs(freq_repr)
                        
                        if freq_power.sum() > 0:
                            freq_power = freq_power / (freq_power.sum() + group['eps'])
                        
                        band_size = freq_power.shape[0] // n_bands
                        if band_size > 0:
                            band_powers = []
                            for i in range(n_bands):
                                start_idx = i * band_size
                                end_idx = min((i+1) * band_size, freq_power.shape[0])
                                if start_idx < end_idx:
                                    band_power = freq_power[start_idx:end_idx].sum().item()
                                    band_powers.append(band_power)
                                else:
                                    band_powers.append(0.0)
                            
                            for i, power in enumerate(band_powers):
                                band_key = f'band_{i}'
                                if band_key not in state['freq_history']:
                                    state['freq_history'][band_key] = power
                                else:
                                    state['freq_history'][band_key] = (
                                        beta * state['freq_history'][band_key] +
                                        (1-beta) * power
                                    )
                            
                            adaptivity = torch.ones_like(grad)
                            
                            effective_alpha = alpha
                            
                            high_freq_activity = sum(state['freq_history'].get(f'band_{i}', 0) 
                                                    for i in range(n_bands//2, n_bands))
                            
                            if high_freq_activity > 0.3:
                                effective_alpha = min(0.95, alpha + 0.05)
                            
                            band_values = [state['freq_history'].get(f'band_{i}', 0) 
                                          for i in range(n_bands)]
                            max_band = max(range(n_bands), key=lambda i: band_values[i])
                            
                            if group['layer_boost']:
                                if p.dim() > 1:
                                    row_factor = 1.0
                                    col_factor = 1.0
                                    
                                    if max_band < n_bands // 3:
                                        effective_alpha *= 0.95
                                    
                                    elif max_band < 2 * n_bands // 3:
                                        pass
                                    
                                    else:
                                        effective_alpha = min(0.98, effective_alpha * 1.05)
                            
                            if self.debug:
                                param_name = state['param_name']
                                if param_name not in self.debug_info:
                                    self.debug_info[param_name] = {'steps': [], 'bands': []}
                                
                                if state['step'] % 10 == 0:
                                    self.debug_info[param_name]['steps'].append(state['step'])
                                    self.debug_info[param_name]['bands'].append(band_values)
                            
                            exp_avg.mul_(effective_alpha).add_(grad * adaptivity, alpha=1-effective_alpha)
                        else:
                            exp_avg.mul_(alpha).add_(grad, alpha=1-alpha)
                    except Exception as e:
                        print(f"Error in FAM processing: {e}")
                        exp_avg.mul_(alpha).add_(grad, alpha=1-alpha)
                else:
                    exp_avg.mul_(alpha).add_(grad, alpha=1-alpha)
                
                p.add_(exp_avg, alpha=-lr)
        
        return loss
    
    optimizer = FAMOptimizer(
        model.parameters(),
        lr=0.001,
        alpha=0.9,
        beta=0.99,
        n_bands=8,
        fam_start_step=10,
        debug=True
    )

```
