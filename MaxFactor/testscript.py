
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
from torch.optim import Optimizer
import os, math 
from datetime import datetime


import matplotlib.pyplot as plt
import time
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


class MaxFactor(torch.optim.Optimizer):

    def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(1e-12, 1e-8), d=1.0, 
                 weight_decay=0.25, gamma=0.99, max=False,
                 full_matrix=False, clip=0.95):
                   
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





def optimizer_benchmark(model_fn, dataset_name='mnist', batch_size=128, 
                        epochs=10, learning_rates=None, seeds=None,
                        optimizers_dict=None, save_plots=True):
    """
    Benchmark different optimizers on a given model and dataset.
    
    Args:
        model_fn: Function that returns a model
        dataset_name: Name of dataset to use ('mnist', 'cifar10')
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rates: Dictionary mapping optimizer names to learning rates
        seeds: List of random seeds to use
        optimizers_dict: Dictionary mapping names to optimizer constructors
        save_plots: Whether to save plots to files
        
    Returns:
        Dictionary with benchmark results
    """
    if seeds is None:
        seeds = [42, 123, 456]
        
    if learning_rates is None:
        learning_rates = {
            'SGD': 0.01,
            'Adam': 0.001,
            'AdamW': 0.001,
            'MaxFactor': 0.025
        }
    
    if optimizers_dict is None:
        optimizers_dict = {
            'SGD': lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
            'Adam': lambda params, lr: torch.optim.Adam(params, lr=lr),
            'AdamW': lambda params, lr: torch.optim.AdamW(params, lr=lr, weight_decay=0.0025),
            'MaxFactor': lambda params, lr: MaxFactor(params=params, lr=lr)

        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if dataset_name.lower() == 'mnist':
        train_dataset = datasets.MNIST('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]))
        test_dataset = datasets.MNIST('./data', train=False, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]))
        input_channels = 1
        num_classes = 10
    elif dataset_name.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                (0.2023, 0.1994, 0.2010))
                                        ]))
        test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                       ]))
        input_channels = 3
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_dataset = Subset(train_dataset, range(5000))
    test_dataset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    results = {
        'train_loss': {},
        'train_acc': {},
        'test_loss': {},
        'test_acc': {},
        'time_per_epoch': {},
        'param_update_norm': {},
        'grad_norm': {},
    }
    
    for opt_name in optimizers_dict.keys():
        results['train_loss'][opt_name] = []
        results['train_acc'][opt_name] = []
        results['test_loss'][opt_name] = []
        results['test_acc'][opt_name] = []
        results['time_per_epoch'][opt_name] = []
        results['param_update_norm'][opt_name] = []
        results['grad_norm'][opt_name] = []
    
    for opt_name, opt_constructor in optimizers_dict.items():
        print(f"\nBenchmarking optimizer: {opt_name}")
        
        seed_train_losses = []
        seed_train_accs = []
        seed_test_losses = []
        seed_test_accs = []
        seed_times = []
        seed_update_norms = []
        seed_grad_norms = []
        
        for seed_idx, seed in enumerate(seeds):
            print(f"  Running with seed {seed} ({seed_idx+1}/{len(seeds)})")
            set_seed(seed)
            
            model = model_fn(input_channels, num_classes).to(device)
            
            lr = learning_rates.get(opt_name, 0.01)
            optimizer = opt_constructor(model.parameters(), lr)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr/10
            )
            
            train_losses = []
            train_accs = []
            test_losses = []
            test_accs = []
            epoch_times = []
            update_norms = []
            grad_norms = []
            
            for epoch in range(epochs):
                start_time = time.time()
                
                model.train()
                train_loss, train_acc, epoch_update_norm, epoch_grad_norm = train_epoch(
                    model, train_loader, optimizer, device
                )
                
                model.eval()
                test_loss, test_acc = evaluate(model, test_loader, device)
                
                scheduler.step()
                
                epoch_time = time.time() - start_time
                
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                epoch_times.append(epoch_time)
                update_norms.append(epoch_update_norm)
                grad_norms.append(epoch_grad_norm)
                
                print(f"    Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
                      f"Time: {epoch_time:.2f}s")
            
            seed_train_losses.append(train_losses)
            seed_train_accs.append(train_accs)
            seed_test_losses.append(test_losses)
            seed_test_accs.append(test_accs)
            seed_times.append(epoch_times)
            seed_update_norms.append(update_norms)
            seed_grad_norms.append(grad_norms)
        
        results['train_loss'][opt_name] = np.mean(seed_train_losses, axis=0)
        results['train_acc'][opt_name] = np.mean(seed_train_accs, axis=0)
        results['test_loss'][opt_name] = np.mean(seed_test_losses, axis=0)
        results['test_acc'][opt_name] = np.mean(seed_test_accs, axis=0)
        results['time_per_epoch'][opt_name] = np.mean(seed_times, axis=0)
        results['param_update_norm'][opt_name] = np.mean(seed_update_norms, axis=0)
        results['grad_norm'][opt_name] = np.mean(seed_grad_norms, axis=0)
    
    if save_plots:
        plot_metrics(results, dataset_name)
    
    return results


def train_epoch(model, train_loader, optimizer, device):
    """Train model for one epoch and return metrics"""
    train_loss = 0
    correct = 0
    total = 0
    
    param_update_norm = 0
    grad_norm = 0
    update_samples = 0
    
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.clone().detach()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        batch_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                batch_grad_norm += param.grad.norm(2).item() ** 2
        grad_norm += batch_grad_norm ** 0.5
        
        optimizer.step()
        
        if batch_idx % 5 == 0:
            batch_update_norm = 0
            for name, param in model.named_parameters():
                if name in initial_params:
                    update = param.detach() - initial_params[name]
                    batch_update_norm += update.norm(2).item() ** 2
                    initial_params[name] = param.clone().detach()
            param_update_norm += batch_update_norm ** 0.5
            update_samples += 1
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    train_loss /= len(train_loader)
    accuracy = 100. * correct / total
    param_update_norm = param_update_norm / max(1, update_samples)
    grad_norm /= len(train_loader)
    
    return train_loss, accuracy, param_update_norm, grad_norm


def evaluate(model, test_loader, device):
    """Evaluate model on test data"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    return test_loss, accuracy





def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_metrics(results, dataset_name):
    """Plot benchmark metrics"""
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    for opt_name, losses in results['train_loss'].items():
        plt.plot(losses, label=opt_name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    for opt_name, losses in results['test_loss'].items():
        plt.plot(losses, label=opt_name)
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    for opt_name, accs in results['test_acc'].items():
        plt.plot(accs, label=opt_name)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    for opt_name, times in results['time_per_epoch'].items():
        plt.plot(times, label=opt_name)
    plt.title('Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    for opt_name, norms in results['param_update_norm'].items():
        plt.plot(norms, label=opt_name)
    plt.title('Parameter Update Norm')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    for opt_name, norms in results['grad_norm'].items():
        plt.plot(norms, label=opt_name)
    plt.title('Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./test/optimizer_benchmark_{dataset_name}.png')
    plt.close()


def create_mlp(input_channels, num_classes):
    """Create a simple MLP model"""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_channels * 28 * 28 if input_channels == 1 else input_channels * 32 * 32, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )


def create_cnn(input_channels, num_classes):
    """Create a simple CNN model"""
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )


class ConvNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        if input_channels == 1:
            fc_size = 128 * 3 * 3
        else:
            fc_size = 128 * 4 * 4
            
        self.fc1 = nn.Linear(fc_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x





def learning_rate_search():
    """Test different learning rates for each optimizer"""
    test_lrs = {
        'SGD': [0.1, 0.05, 0.01, 0.005, 0.001],
        'Adam': [0.01, 0.005, 0.001, 0.0005, 0.0001],
        'AdamW': [0.01, 0.005, 0.001, 0.0005, 0.0001],
        'MaxFactor': [0.05, 0.01, 0.005, 0.001, 0.0005]
    }
    
    best_lrs = {}
    best_accuracies = {}
    
    for opt_name, lrs in test_lrs.items():
        print(f"\nTesting learning rates for {opt_name}")
        best_acc = 0
        best_lr = None
        
        for lr in lrs:
            print(f"  Testing lr={lr}")
            
            if opt_name == 'SGD':
                opt_constructor = lambda params, _: torch.optim.SGD(params, lr=lr, momentum=0.9)
            elif opt_name == 'Adam':
                opt_constructor = lambda params, _: torch.optim.Adam(params, lr=lr)
            elif opt_name == 'AdamW':
                opt_constructor = lambda params, _: torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
            elif opt_name == 'MaxFactor':
                opt_constructor = lambda params, _: MaxFactor(
                    params, lr=lr)
            
            optimizers_dict = {opt_name: opt_constructor}
            
            lr_dict = {opt_name: lr}
            
            results = optimizer_benchmark(
                create_cnn, 'mnist', batch_size=128, epochs=5, 
                learning_rates=lr_dict, seeds=[42], 
                optimizers_dict=optimizers_dict, save_plots=False
            )
            
            final_acc = results['test_acc'][opt_name][-1]
            if final_acc > best_acc:
                best_acc = final_acc
                best_lr = lr
        
        best_lrs[opt_name] = best_lr
        best_accuracies[opt_name] = best_acc
        print(f"Best learning rate for {opt_name}: {best_lr} (Accuracy: {best_acc:.2f}%)")
    
    return best_lrs, best_accuracies


def advanced_test():
    """Run comprehensive benchmarks"""
    best_lrs, _ = learning_rate_search()
    
    optimizers_dict = {
        'SGD': lambda params, lr: torch.optim.SGD(params, lr=best_lrs['SGD'], momentum=0.9),
        'Adam': lambda params, lr: torch.optim.Adam(params, lr=best_lrs['Adam']),
        'AdamW': lambda params, lr: torch.optim.AdamW(params, lr=best_lrs['AdamW'], weight_decay=0.01),
        'MaxFactor': lambda params, lr: MaxFactor(params, lr=best_lrs['MaxFactor']
        )
    }
    
    print("\nRunning comprehensive benchmarks...")
    
    print("\nBenchmarking CNN on MNIST")
    cnn_mnist_results = optimizer_benchmark(
        create_cnn, 'mnist', batch_size=128, epochs=10, 
        learning_rates=best_lrs, seeds=[42, 123, 456], 
        optimizers_dict=optimizers_dict, save_plots=True
    )
    
    print("\nBenchmarking CNN on CIFAR10")
    cnn_cifar_results = optimizer_benchmark(
        create_cnn, 'cifar10', batch_size=128, epochs=10, 
        learning_rates=best_lrs, seeds=[42, 123, 456], 
        optimizers_dict=optimizers_dict, save_plots=True
    )
    
    print("\nBenchmarking ConvNet on CIFAR10")
    convnet_cifar_results = optimizer_benchmark(
        ConvNet, 'cifar10', batch_size=128, epochs=10, 
        learning_rates=best_lrs, seeds=[42, 123, 456], 
        optimizers_dict=optimizers_dict, save_plots=True
    )
    
    return {
        'cnn_mnist': cnn_mnist_results,
        'cnn_cifar': cnn_cifar_results,
        'convnet_cifar': convnet_cifar_results
    }
    
def memory_usage_test():
    """Compare memory usage of different optimizers"""
    import torch
    import gc
    import numpy as np
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    feature_dims = [100, 200, 400, 800, 1600]
    optimizers = ['SGD', 'Adam', 'AdamW', 'MaxFactor']
    
    memory_usage = {opt: [] for opt in optimizers}
    feature_dims_list = []
    
    def get_gpu_memory_usage():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return 0
    
    hidden_dim = 2048
    batch_size = 128
    
    print("Measuring memory usage...")
    
    for feature_dim in feature_dims:
        print(f"\nTesting with feature dimension: {feature_dim}")
        feature_dims_list.append(feature_dim)
        
        x = torch.randn(1000, feature_dim, device=device)
        y = torch.randint(0, 10, (1000,), device=device)
        
        for opt_name in optimizers:
            print(f"  Testing {opt_name}...")
            
            del x, y
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            x = torch.randn(1000, feature_dim, device=device)
            y = torch.randint(0, 10, (1000,), device=device)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            base_memory = get_gpu_memory_usage()
            
            model = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim // 2, 10)
            ).to(device)
            
            if opt_name == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            elif opt_name == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            elif opt_name == 'AdamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            elif opt_name == 'MaxFactor':
                optimizer = MaxFactor(params=model.parameters(), lr=0.01, weight_decay=0.01)
            
            inputs, targets = x[:batch_size], y[:batch_size]
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            current_memory = get_gpu_memory_usage()
            
            memory_used = current_memory - base_memory
            memory_usage[opt_name].append(memory_used)
            
            print(f"    Memory used: {memory_used:.2f} MB")
            
            del model, optimizer, inputs, targets, outputs, loss
    
    result = {
        'feature_dims': feature_dims_list,
        'memory_usage': memory_usage
    }
    plot_memory_usage(memory_data=result)
    return result

def plot_memory_usage(memory_data):
    """Plot memory usage comparison"""
    feature_dims = memory_data['feature_dims']
    memory_usage = memory_data['memory_usage']
    optimizers = list(memory_usage.keys())
    
    plt.figure(figsize=(10, 6))
    
    width = 0.2
    x = np.arange(len(feature_dims))
    
    for i, opt_name in enumerate(optimizers):
        plt.bar(x + i*width, memory_usage[opt_name], width, label=opt_name)
    
    plt.xlabel('Feature Dimension')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage by Optimizer')
    plt.xticks(x + width*1.5, feature_dims)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./test/memory_usage_comparison.png')
    
    return {
        'feature_dims': feature_dims,
        'memory_usage': memory_usage
    }

def simple_test():
    """Run a simpler test for quick validation"""
    learning_rates = {
        'SGD': 0.01,
        'Adam': 0.001,
        'MaxFactor': 0.01
    }
    
    optimizers_dict = {
        'SGD': lambda params, lr: torch.optim.SGD(params, lr=learning_rates['SGD'], momentum=0.9),
        'Adam': lambda params, lr: torch.optim.Adam(params, lr=learning_rates['Adam']),
        'MaxFactor': lambda params, lr: MaxFactor(
            params, lr=learning_rates['MaxFactor']
        )
    }
    
    results = optimizer_benchmark(
        create_cnn, 'mnist', batch_size=128, epochs=5, 
        learning_rates=learning_rates, seeds=[42], 
        optimizers_dict=optimizers_dict, save_plots=True
    )
    
    return results

def add_memory_usage_to_report(report, memory_data):
    """Add memory usage information to the benchmark report"""
    report += "\nMemory Usage Comparison\n"
    report += "=====================\n\n"
    
    feature_dims = memory_data['feature_dims']
    memory_usage = memory_data['memory_usage']
    
    for dim in range(len(feature_dims)):
        report += f"Feature Dimension: {feature_dims[dim]}\n"
        report += "--------------------------\n"
        for opt_name in memory_usage:
            report += f"  {opt_name}: {memory_usage[opt_name][dim]:.2f} MB\n"
        report += "\n"
    
    if 'MaxFactor' in memory_usage and 'AdamW' in memory_usage:
        avg_maxfactor = np.mean(memory_usage['MaxFactor'])
        avg_adamw = np.mean(memory_usage['AdamW'])
        savings_pct = 100 * (avg_adamw - avg_maxfactor) / avg_adamw
        report += f"MaxFactor uses {savings_pct:.1f}% less memory than AdamW on average.\n\n"
    
    return report

def summary_report(results):
    """Generate a summary report of benchmark results"""
    report = "Optimizer Benchmark Summary\n"
    report += "=========================\n\n"
    
    for dataset, dataset_results in results.items():
        report += f"Dataset: {dataset}\n"
        report += "-----------------\n"
        
        report += "Final Test Accuracy:\n"
        for opt_name, accs in dataset_results['test_acc'].items():
            final_acc = accs[-1]
            report += f"  {opt_name}: {final_acc:.2f}%\n"
        
        report += "\nConvergence Speed (epochs to 90% of final accuracy):\n"
        for opt_name, accs in dataset_results['test_acc'].items():
            final_acc = accs[-1]
            target_acc = 0.9 * final_acc
            epochs_to_target = next((i for i, acc in enumerate(accs) if acc >= target_acc), len(accs))
            report += f"  {opt_name}: {epochs_to_target} epochs\n"
        
        report += "\nAverage Time per Epoch:\n"
        for opt_name, times in dataset_results['time_per_epoch'].items():
            avg_time = np.mean(times)
            report += f"  {opt_name}: {avg_time:.2f}s\n"
        
        report += "\nAverage Parameter Update Norm:\n"
        for opt_name, norms in dataset_results['param_update_norm'].items():
            avg_norm = np.mean(norms)
            report += f"  {opt_name}: {avg_norm:.4f}\n"
        
        report += "\n\n"
    
    return report





    
    



if __name__ == "__main__":
    print("Running simple test...")
    simple_results = simple_test()
    
    print("\nRunning memory usage test...")
    memory_data = memory_usage_test()
    
    print("\nRunning comprehensive benchmarks...")
    advanced_results = advanced_test()
    
    report = summary_report(advanced_results)
    


    report = add_memory_usage_to_report(report, memory_data)
    
    print(report)
    
    log_dir = os.path.join('./test/benchmark/', datetime.now().strftime(format='%m-%d_%H'))
    os.makedirs(name=log_dir, exist_ok=True)
    with open(log_dir+"/benchmarks.txt", "w") as f:
        f.write(report)



