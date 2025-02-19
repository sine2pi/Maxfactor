Experimental optimizer(wip) - ASR/NLP - A mix of things from other optimizers I found worked well for ASR models. Part Adafactor part RMSprop part Adamax part something else I can't remember where it came from.

#### Maxfactor (not ready)
                      
      
      class MaxFactor(torch.optim.Optimizer):
          def __init__(
              self, params, lr=None, eps=(1e-10, 1e-3), clip_threshold=1.0,
              decay_rate=-0.8, beta1=None, weight_decay=0.0, gamma=0.99,
              eps_rms=1e-8, maximize=False, **kwargs):
      
              defaults = {
                  "lr": lr,
                  "eps": eps,
                  "clip_threshold": clip_threshold,
                  "decay_rate": decay_rate,
                  "beta1": beta1,
                  "weight_decay": weight_decay,
                  "gamma": gamma,
                  "eps_rms": eps_rms,
                  "maximize": maximize,
                  "d": 1.0,
              }
              defaults.update(kwargs)
              super().__init__(params, defaults)
              
          
          def _get_lr(self, param_group, param_state):
              step = param_state["step"]
              min_step = 1e-6 * step
              rel_step_sz = min(min_step, 1.0 / torch.sqrt(torch.tensor(step)).item())
              param_scale = max(param_group["eps"][1], param_state["RMS"])
              return param_scale * rel_step_sz
      
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
                  params_with_grad = []
                  grads = []
                  row_vars = []
                  col_vars = []
                  vs = []
                  state_steps = []
                  eps1, eps2 = group["eps"]
      
                  for param in group["params"]:
                      if param.grad is None:
                          continue
                      grad = param.grad
                      if grad.dtype in {torch.float16, torch.bfloat16}:
                          grad = grad.float()
      
                      state = self.state[param]
                      if len(state) == 0:
                          state["step"] = torch.tensor(0.0, dtype=torch.float32)
                          if grad.dim() > 1:
                              row_shape = list(grad.shape)
                              col_shape = list(grad.shape)
                              row_shape[-1] = 1
                              col_shape[-2] = 1
                              state["row_var"] = grad.new_zeros(row_shape)
                              state["col_var"] = grad.new_zeros(col_shape)
                          state["v"] = torch.zeros_like(param, memory_format=torch.preserve_format)
      
                      row_vars.append(state.get("row_var", None))
                      col_vars.append(state.get("col_var", None))
                      vs.append(state["v"])
                      state_steps.append(state["step"])
                      params_with_grad.append(param)
                      grads.append(grad)
      
                  for i, param in enumerate(params_with_grad):
                      grad = grads[i]
      
                      if group["maximize"]:
                          grad = -grad
                      step_t = state_steps[i]
                      row_var = row_vars[i]
                      col_var = col_vars[i]
                      v = vs[i]
      
                      step_t += 1
                      step_float = step_t.item()
                      state_steps[i] = step_t
                      state = self.state[param]
                      state["step"] = step_t
                      state["RMS"] = self._rms(param)
                      lr = self._get_lr(group, state)
                      one_minus_beta2_t = step_float ** group["decay_rate"]
                      rho_t = min(group["lr"] or 1.0, 1 / (step_float ** 0.5))
                      alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t
      
                      if group["weight_decay"] != 0:
                          param.mul_(1 - group["lr"] * group["weight_decay"])
      
                      
                      if grad.dim() > 1:
                          row_mean = torch.norm(grad, dim=-1, keepdim=True).square().div_(grad.size(-1) + 1e-8)
                          row_var.lerp_(row_mean, one_minus_beta2_t)
                          col_mean = torch.norm(grad, dim=-2, keepdim=True).square().div_(grad.size(-2) + 1e-8)
                          col_var.lerp_(col_mean, one_minus_beta2_t)
                          
                          var_estimate = row_var @ col_var  
                          
                          max_row_var = row_var.amax(dim=-1, keepdim=True)
                          var_estimate.div_(max_row_var.clamp_(min=eps1))
                      else:
                          v.mul_(group["gamma"]).add_(grad.square(), alpha=1 - group["gamma"])
                          var_estimate = v
      
      
                      update = var_estimate.clamp_(min=eps1 ** 2).rsqrt_().mul_(grad)
                      max_update = update.abs().amax()
                      if max_update < eps1:
                          max_update = eps1
                      update.div_(max_update)
                      denom_factor = (update.numel() ** 0.5) * group.get("d", 1.0)
                      denom = max(1.0, update.norm(2).item() / denom_factor)
                      param.add_(-alpha / denom * update)
      
              return loss
      
              
      optimizer = MaxFactor(
          model.parameters(), 
          lr=0.025,  
          beta2_decay=-0.8,
          eps=(1e-10, 1e-4),
          d=0.98,
          weight_decay=0.025,
          gamma=0.99, 
          eps_rms=1e-8,
          maximize=False,
          )
      
      # scheduler = torch.optim.lr_scheduler.LambdaLR(
      #     optimizer = optimizer,
      #     lr_lambda=lambda step: 0.9 ** (step / training_args.max_steps),
      #     last_epoch=-1  
      
      # )
      
      # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      #     optimizer=optimizer,
      #     T_max=training_args.max_steps,
      #     eta_min=0.00025,
      #     last_epoch=-1  
      # )
      

        
