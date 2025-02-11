Experimental optimizer - wip

#### Maxfactor


    class Maxfactor(Optimizer):
        def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
            defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
            super(Maxfactor, self).__init__(params, defaults)
    
        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()
    
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    
                    grad = param.grad.data
                    state = self.state[param]
    
                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(param.data)
                        state['exp_inf'] = torch.zeros_like(param.data)
                        state['grad_avg'] = torch.zeros_like(param.data)
    
                    exp_avg, exp_inf, grad_avg = state['exp_avg'], state['exp_inf'], state['grad_avg']
                    beta1, beta2 = group['beta1'], group['beta2']
    
                    state['step'] += 1
    
                    # Adafactor-like scaling of gradient
                    grad_sq = grad * grad + group['eps']
                    factor = torch.sqrt(torch.mean(grad_sq))
                    grad /= factor
    
                    # AdaMax update
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_inf = torch.max(exp_inf, grad.abs())
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
    
                    param.data.addcdiv_(-step_size, exp_avg, exp_inf + group['eps'])
    
                    # Apply weight decay if specified
                    if group['weight_decay'] != 0:
                        param.data.add_(-group['lr'] * group['weight_decay'], param.data)
    
            return loss
    
    optimizer = Maxfactor(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0)

