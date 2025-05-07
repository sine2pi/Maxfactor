Make sure to start with the default values and then tweak if needed. Learning rates and values used with other optimizers like AdamW will likely NaN out. depending on your architecture.*

**Parameter-specific learning rate scaling is crucial for transformer-based ASR models.
Uniform optimization techniques often struggle with ASR - they're trying to apply the same rules to fundamentally different computational structures.**



MaxFactor can be characterized as a memory-efficient adaptive optimizer that combines several innovative techniques:

1. **Factored Second Moments for Matrices**
   - Like Adafactor, it uses row and column statistics rather than storing full matrices
   - Significantly reduces memory requirements compared to Adam-style optimizers

2. **Sign-Based Matrix Updates with Max-Pooling**
   - For matrix parameters, takes sign of updates and scales by the maximum value along rows
   - This unique approach bridges sign-based methods and adaptive learning

3. **Dual Normalization Strategy**
   - RMS-based clipping controls overall update magnitude
   - Optional infinity norm normalization ensures balanced updates across dimensions

4. **Adaptive Learning Rate Scheduling**
   - Incorporates automatic learning rate decay based on step count
   - Parameter-specific scaling based on RMS values

## Technical Strengths

1. **Memory Efficiency**
   - O(n) storage for second moments rather than O(n²) like Adam
   - Especially beneficial for large language models with massive embedding matrices

2. **Numerical Stability**
   - Multiple safeguards against exploding/vanishing gradients
   - Bounded beta values prevent extreme adaptation rates

3. **Flexibility**
   - Works in both minimization and maximization modes
   - Configurable for different parameter shapes (vectors vs matrices)

MaxFactor essentially represents a hybrid approach that combines the memory efficiency of Adafactor, the adaptivity of Adam, and the robustness of sign-based methods, with its own unique max-pooling innovation for matrix parameters. It's particularly well-suited for training large models where memory constraints are significant.

---
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
    MaxFactor optimizer: A memory-efficient optimizer with max-based normalization
    for training speech recognition models.
    
    Key features:
    - Factorized second moments (memory efficient)
    - Max-based normalization (better for attention & rotary params)
    - Infinity norm clipping (prevents extreme updates)
    - Per-parameter adaptive learning rates
    
    Args:
        params: Model parameters or param groups
        lr: Learning rate (default: 0.01)
        beta2_decay: Power for step-size decay (default: -0.8)
        eps: Small constants for numerical stability (default: (1e-10, 1e-4))
        d: Update scale control factor (default: 1.0)
        weight_decay: Weight decay factor (default: 0.01)
        gamma: EMA decay rate for non-factorized tensors (default: 0.99)
        max_norm: Whether to use max normalization (default: True)
        min_lr: Minimum learning rate (default: 1e-7)
    """
    def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(1e-10, 1e-4), 
                 d=1.0, weight_decay=0.01, gamma=0.99, max_norm=True, min_lr=1e-7, scalar_boost=2.0):
        
        defaults = dict(
            lr=lr, 
            beta2_decay=beta2_decay, 
            eps=eps, 
            d=d, 
            weight_decay=weight_decay, 
            gamma=gamma, 
            max_norm=max_norm, 
            min_lr=min_lr,
            scalar_boost=scalar_boost
        )
        super().__init__(params=params, defaults=defaults)
        print(f"MaxFactor optimizer initialized with lr={lr}, beta2_decay={beta2_decay}")

    def _get_lr(self):
        """Return the current learning rates as a dictionary."""
        return {i: group['lr'] for i, group in enumerate(self.param_groups)}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps1, eps2 = group["eps"]
            min_lr = group.get("min_lr", 1e-7)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.float()
                state = self.state[p]
                
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, device=p.device)
                    
                    if p.dim() > 1:
                        row_shape, col_shape = list(p.shape), list(p.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"] = torch.zeros(row_shape, device=p.device)
                        state["col_var"] = torch.zeros(col_shape, device=p.device)
                    
                    state["v"] = torch.zeros_like(p)
                
                state["step"] += 1
                step_float = state["step"].item()
                
                one_minus_beta2_t = min(0.999, max(0.001, step_float ** group["beta2_decay"]))
                rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
                param_scale = (p.norm() / (p.numel() ** 0.5 + 1e-12)).item()
                alpha = max(eps2, param_scale) * rho_t
                
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                
                if p.dim() > 1:
                    row_var = state["row_var"]
                    col_var = state["col_var"]
                    
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_()
                    row_mean.div_(grad.size(-1) + eps1)
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_()
                    col_mean.div_(grad.size(-2) + eps1)
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    
                    var_estimate = row_var @ col_var
                    
                    if group["max_norm"]:
                        max_row_var = row_var.max(dim=-2, keepdim=True)[0]
                        var_estimate.div_(max_row_var.clamp_(min=eps1))
                
                else:
                    vi = state["v"]
                    vi.mul_(group["gamma"]).add_(grad.square_(), alpha=1 - group["gamma"])
                    var_estimate = vi


                if p.numel() == 1:
                    update = grad / (var_estimate.sqrt() + eps1)
                    scalar_boost = group.get("scalar_boost", 2.0)
                    p.add_(-alpha * scalar_boost * update)
                else:
                    update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                    
                    inf_norm = torch.norm(update, float('inf'))
                    if inf_norm > 0:
                        update.div_(inf_norm.clamp_(min=eps1))
                    
                    denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                    
                    if p.dim() > 1 and group["max_norm"]:
                        max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                        p.add_(-alpha / denom * update.sign() * max_vals)
                    else:
                        p.add_(-alpha / denom * update)
                    
            return loss
    



