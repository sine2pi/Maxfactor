import time
from torch.optim import Optimizer
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class MaxFactor(Optimizer):
    def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(None, 1e-3), d=1.0, 
                 weight_decay=0.0, gamma=0.99, eps_rms=1e-8, maximize=False):
        
        defaults = dict(lr=lr, beta2_decay=beta2_decay, eps=eps, d=d, weight_decay=weight_decay, 
                        gamma=gamma, eps_rms=eps_rms, maximize=maximize)
        super().__init__(params=params, defaults=defaults)

    def _get_lr(self, param_group, param_state):
        step = param_state["step"]
        min_step = 1e-6 * step
        rel_step_sz = min(min_step, 1.0 / step.sqrt())
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
            params_with_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
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
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                params_with_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(params_with_grad):
                grad = grads[i]

                if group["maximize"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                step_t += 1
                step_float = step_t.item()
                one_minus_beta2_t = step_float ** group["beta2_decay"]
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["weight_decay"]!= 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
                    var_estimate = vi

                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])
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
    # Default parameters
    if seeds is None:
        seeds = [42, 123, 456]  # Multiple seeds for statistical significance
        
    if learning_rates is None:
        learning_rates = {
            'SGD': 0.01,
            'Adam': 0.001,
            'AdamW': 0.001,
            'MaxFactor': 0.01
        }
    
    if optimizers_dict is None:
        optimizers_dict = {
            'SGD': lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
            'Adam': lambda params, lr: torch.optim.Adam(params, lr=lr),
            'AdamW': lambda params, lr: torch.optim.AdamW(params, lr=lr, weight_decay=0.0025),
            'MaxFactor': lambda params, lr: MaxFactor(
                params=params, lr=lr, beta2_decay=-0.7, eps=(1e-8, 1e-4), d=1.0,
                weight_decay=0.01, gamma=0.99, eps_rms=1e-8, maximize=False)

        }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
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
    
    # For quicker iterations, use a subset of the data during development
    # Remove this for full benchmarks
    train_dataset = Subset(train_dataset, range(5000))  # 5000 samples for training
    test_dataset = Subset(test_dataset, range(1000))    # 1000 samples for testing
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Dictionary to store results
    results = {
        'train_loss': {},
        'train_acc': {},
        'test_loss': {},
        'test_acc': {},
        'time_per_epoch': {},
        'param_update_norm': {},
        'grad_norm': {},
    }
    
    # Initialize results dictionaries
    for opt_name in optimizers_dict.keys():
        results['train_loss'][opt_name] = []
        results['train_acc'][opt_name] = []
        results['test_loss'][opt_name] = []
        results['test_acc'][opt_name] = []
        results['time_per_epoch'][opt_name] = []
        results['param_update_norm'][opt_name] = []
        results['grad_norm'][opt_name] = []
    
    # Run benchmark for each optimizer
    for opt_name, opt_constructor in optimizers_dict.items():
        print(f"\nBenchmarking optimizer: {opt_name}")
        
        # Store results across seeds
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
            
            # Create model
            model = model_fn(input_channels, num_classes).to(device)
            
            # Create optimizer
            lr = learning_rates.get(opt_name, 0.01)
            optimizer = opt_constructor(model.parameters(), lr)
            
            # Create scheduler (cosine annealing)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr/10
            )
            
            # Lists to store metrics for this run
            train_losses = []
            train_accs = []
            test_losses = []
            test_accs = []
            epoch_times = []
            update_norms = []
            grad_norms = []
            
            # Training loop
            for epoch in range(epochs):
                start_time = time.time()
                
                # Train
                model.train()
                train_loss, train_acc, epoch_update_norm, epoch_grad_norm = train_epoch(
                    model, train_loader, optimizer, device
                )
                
                # Evaluate
                model.eval()
                test_loss, test_acc = evaluate(model, test_loader, device)
                
                # Step scheduler
                scheduler.step()
                
                # Record time
                epoch_time = time.time() - start_time
                
                # Store metrics
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                epoch_times.append(epoch_time)
                update_norms.append(epoch_update_norm)
                grad_norms.append(epoch_grad_norm)
                
                # Print progress
                print(f"    Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
                      f"Time: {epoch_time:.2f}s")
            
            # Store results for this seed
            seed_train_losses.append(train_losses)
            seed_train_accs.append(train_accs)
            seed_test_losses.append(test_losses)
            seed_test_accs.append(test_accs)
            seed_times.append(epoch_times)
            seed_update_norms.append(update_norms)
            seed_grad_norms.append(grad_norms)
        
        # Average results across seeds
        results['train_loss'][opt_name] = np.mean(seed_train_losses, axis=0)
        results['train_acc'][opt_name] = np.mean(seed_train_accs, axis=0)
        results['test_loss'][opt_name] = np.mean(seed_test_losses, axis=0)
        results['test_acc'][opt_name] = np.mean(seed_test_accs, axis=0)
        results['time_per_epoch'][opt_name] = np.mean(seed_times, axis=0)
        results['param_update_norm'][opt_name] = np.mean(seed_update_norms, axis=0)
        results['grad_norm'][opt_name] = np.mean(seed_grad_norms, axis=0)
    
    # Plot results
    if save_plots:
        plot_metrics(results, dataset_name)
    
    return results


def train_epoch(model, train_loader, optimizer, device):
    """Train model for one epoch and return metrics"""
    train_loss = 0
    correct = 0
    total = 0
    
    # For tracking parameter updates and gradient norms
    param_update_norm = 0
    grad_norm = 0
    update_samples = 0
    
    # Store initial parameters for computing updates
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.clone().detach()
    
    # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Calculate gradient norm before optimizer step
        batch_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                batch_grad_norm += param.grad.norm(2).item() ** 2
        grad_norm += batch_grad_norm ** 0.5
        
        optimizer.step()
        
        # Sample parameter updates to avoid excessive memory usage
        if batch_idx % 5 == 0:  # Sample every 5 batches
            batch_update_norm = 0
            for name, param in model.named_parameters():
                if name in initial_params:
                    update = param.detach() - initial_params[name]
                    batch_update_norm += update.norm(2).item() ** 2
                    # Update initial params for next comparison
                    initial_params[name] = param.clone().detach()
            param_update_norm += batch_update_norm ** 0.5
            update_samples += 1
        
        # Compute metrics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    # Normalize metrics
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
    
    # Plot training loss
    plt.subplot(2, 3, 1)
    for opt_name, losses in results['train_loss'].items():
        plt.plot(losses, label=opt_name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot test loss
    plt.subplot(2, 3, 2)
    for opt_name, losses in results['test_loss'].items():
        plt.plot(losses, label=opt_name)
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot test accuracy
    plt.subplot(2, 3, 3)
    for opt_name, accs in results['test_acc'].items():
        plt.plot(accs, label=opt_name)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot time per epoch
    plt.subplot(2, 3, 4)
    for opt_name, times in results['time_per_epoch'].items():
        plt.plot(times, label=opt_name)
    plt.title('Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot parameter update norm
    plt.subplot(2, 3, 5)
    for opt_name, norms in results['param_update_norm'].items():
        plt.plot(norms, label=opt_name)
    plt.title('Parameter Update Norm')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot gradient norm
    plt.subplot(2, 3, 6)
    for opt_name, norms in results['grad_norm'].items():
        plt.plot(norms, label=opt_name)
    plt.title('Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'optimizer_benchmark_{dataset_name}.png')
    plt.close()


# Define model architectures for testing
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


# Define a more challenging convnet model
class ConvNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate size after convolutions and pooling
        if input_channels == 1:  # MNIST: 28x28
            fc_size = 128 * 3 * 3
        else:  # CIFAR10: 32x32
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





# Test all optimizers with different learning rates to find optimal settings
def learning_rate_search():
    """Test different learning rates for each optimizer"""
    test_lrs = {
        'SGD': [0.1, 0.05, 0.01, 0.005, 0.001],
        'Adam': [0.01, 0.005, 0.001, 0.0005, 0.0001],
        'AdamW': [0.01, 0.005, 0.001, 0.0005, 0.0001],
        'MaxFactor': [0.05, 0.01, 0.005, 0.001, 0.0005]
    }
    
    # Results storage
    best_lrs = {}
    best_accuracies = {}
    
    for opt_name, lrs in test_lrs.items():
        print(f"\nTesting learning rates for {opt_name}")
        best_acc = 0
        best_lr = None
        
        for lr in lrs:
            print(f"  Testing lr={lr}")
            
            # Define optimizer constructor with current learning rate
            if opt_name == 'SGD':
                opt_constructor = lambda params, _: torch.optim.SGD(params, lr=lr, momentum=0.9)
            elif opt_name == 'Adam':
                opt_constructor = lambda params, _: torch.optim.Adam(params, lr=lr)
            elif opt_name == 'AdamW':
                opt_constructor = lambda params, _: torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
            elif opt_name == 'MaxFactor':
                opt_constructor = lambda params, _: MaxFactor(
                    params, lr=lr, beta2_decay=-0.8, eps=(1e-10, 1e-4), d=1.0,
                    weight_decay=0.01, gamma=0.999, eps_rms=1e-8
                )
            
            # Test with this optimizer only
            optimizers_dict = {opt_name: opt_constructor}
            
            # Create learning rate dict
            lr_dict = {opt_name: lr}
            
            # Run shorter benchmark
            results = optimizer_benchmark(
                create_cnn, 'mnist', batch_size=128, epochs=5, 
                learning_rates=lr_dict, seeds=[42], 
                optimizers_dict=optimizers_dict, save_plots=False
            )
            
            # Check final accuracy
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
    # First find optimal learning rates
    best_lrs, _ = learning_rate_search()
    
    # Define optimizers with best learning rates
    optimizers_dict = {
        'SGD': lambda params, lr: torch.optim.SGD(params, lr=best_lrs['SGD'], momentum=0.9),
        'Adam': lambda params, lr: torch.optim.Adam(params, lr=best_lrs['Adam']),
        'AdamW': lambda params, lr: torch.optim.AdamW(params, lr=best_lrs['AdamW'], weight_decay=0.01),
        'MaxFactor': lambda params, lr: MaxFactor(
            params, lr=best_lrs['MaxFactor'], beta2_decay=-0.8, eps=(1e-10, 1e-4), 
            d=1.0, weight_decay=0.01, gamma=0.99, eps_rms=1e-8
        )
    }
    
    # Run benchmarks on different models and datasets
    print("\nRunning comprehensive benchmarks...")
    
    # CNN on MNIST
    print("\nBenchmarking CNN on MNIST")
    cnn_mnist_results = optimizer_benchmark(
        create_cnn, 'mnist', batch_size=128, epochs=10, 
        learning_rates=best_lrs, seeds=[42, 123, 456], 
        optimizers_dict=optimizers_dict, save_plots=True
    )
    
    # CNN on CIFAR10
    print("\nBenchmarking CNN on CIFAR10")
    cnn_cifar_results = optimizer_benchmark(
        create_cnn, 'cifar10', batch_size=128, epochs=10, 
        learning_rates=best_lrs, seeds=[42, 123, 456], 
        optimizers_dict=optimizers_dict, save_plots=True
    )
    
    # ConvNet on CIFAR10
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
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create feature dimensions to test
    feature_dims = [100, 200, 400, 800, 1600]
    optimizers = ['SGD', 'Adam', 'AdamW', 'MaxFactor']
    
    # Dictionary to store absolute memory usage (in MB)
    memory_usage = {opt: [] for opt in optimizers}
    feature_dims_list = []
    
    # Function to measure GPU memory
    def get_gpu_memory_usage():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return 0
    
    # Fixed model architecture
    hidden_dim = 2048  # Large enough to see memory differences
    batch_size = 128   # Fixed batch size
    
    print("Measuring memory usage...")
    
    # For each feature dimension
    for feature_dim in feature_dims:
        print(f"\nTesting with feature dimension: {feature_dim}")
        feature_dims_list.append(feature_dim)
        
        # Create fake dataset
        x = torch.randn(1000, feature_dim, device=device)
        y = torch.randint(0, 10, (1000,), device=device)
        
        # Test each optimizer with a clean slate
        for opt_name in optimizers:
            print(f"  Testing {opt_name}...")
            
            # Clear any existing models and optimizers
            del x, y  # Delete previous dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Recreate dataset
            x = torch.randn(1000, feature_dim, device=device)
            y = torch.randint(0, 10, (1000,), device=device)
            
            # Starting memory - measure after dataset creation
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            base_memory = get_gpu_memory_usage()
            
            # Create model
            model = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim // 2, 10)
            ).to(device)
            
            # Create optimizer
            if opt_name == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            elif opt_name == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            elif opt_name == 'AdamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            elif opt_name == 'MaxFactor':
                optimizer = MaxFactor(
                    params=model.parameters(), 
                    lr=0.01,  
                    beta2_decay=-0.8,
                    eps=(1e-10, 1e-4),
                    d=1.0,
                    weight_decay=0.01,
                    gamma=0.99, 
                    eps_rms=1e-8,
                    maximize=False
                )
            
            # Do a single batch update to initialize all optimizer states
            inputs, targets = x[:batch_size], y[:batch_size]
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Ensure all operations are complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Measure memory after optimizer step
            current_memory = get_gpu_memory_usage()
            
            # Calculate absolute memory used
            memory_used = current_memory - base_memory
            memory_usage[opt_name].append(memory_used)
            
            print(f"    Memory used: {memory_used:.2f} MB")
            
            # Clean up
            del model, optimizer, inputs, targets, outputs, loss
    
    # Create result dictionary
    result = {
        'feature_dims': feature_dims_list,
        'memory_usage': memory_usage
    }
    
    return result

def plot_memory_usage(memory_data):
    """Plot memory usage comparison"""
    feature_dims = memory_data['feature_dims']
    memory_usage = memory_data['memory_usage']
    optimizers = list(memory_usage.keys())
    
    # Plot memory usage
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
    plt.savefig('memory_usage_comparison.png')
    
    # Return the data
    return {
        'feature_dims': feature_dims,
        'memory_usage': memory_usage
    }

# Simple test harness
def simple_test():
    """Run a simpler test for quick validation"""
    # Use fixed learning rates
    learning_rates = {
        'SGD': 0.01,
        'Adam': 0.001,
        'MaxFactor': 0.01
    }
    
    # Define optimizers
    optimizers_dict = {
        'SGD': lambda params, lr: torch.optim.SGD(params, lr=learning_rates['SGD'], momentum=0.9),
        'Adam': lambda params, lr: torch.optim.Adam(params, lr=learning_rates['Adam']),
        'MaxFactor': lambda params, lr: MaxFactor(
            params, lr=learning_rates['MaxFactor'], beta2_decay=-0.8, eps=(1e-8, 1e-4), 
            d=1.0, weight_decay=0.025, gamma=0.99, eps_rms=1e-8
        )
    }
    
    # Run short benchmark on MNIST
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
    
    # Calculate average memory savings
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
        
        # Final test accuracy
        report += "Final Test Accuracy:\n"
        for opt_name, accs in dataset_results['test_acc'].items():
            final_acc = accs[-1]
            report += f"  {opt_name}: {final_acc:.2f}%\n"
        
        # Convergence speed (epochs to reach 90% of final accuracy)
        report += "\nConvergence Speed (epochs to 90% of final accuracy):\n"
        for opt_name, accs in dataset_results['test_acc'].items():
            final_acc = accs[-1]
            target_acc = 0.9 * final_acc
            epochs_to_target = next((i for i, acc in enumerate(accs) if acc >= target_acc), len(accs))
            report += f"  {opt_name}: {epochs_to_target} epochs\n"
        
        # Average time per epoch
        report += "\nAverage Time per Epoch:\n"
        for opt_name, times in dataset_results['time_per_epoch'].items():
            avg_time = np.mean(times)
            report += f"  {opt_name}: {avg_time:.2f}s\n"
        
        # Parameter update statistics
        report += "\nAverage Parameter Update Norm:\n"
        for opt_name, norms in dataset_results['param_update_norm'].items():
            avg_norm = np.mean(norms)
            report += f"  {opt_name}: {avg_norm:.4f}\n"
        
        report += "\n\n"
    
    return report





# if __name__ == "__main__":
#     # For quick testing, run simple_test()
#     print("Running simple test...")
#     simple_results = simple_test()
    
#     # For comprehensive benchmarks, uncomment the following:
#     print("Running comprehensive benchmarks...")
#     advanced_results = advanced_test()
#     report = summary_report(advanced_results)
#     print(report)
    

#     log_dir = os.path.join('./test/benchmark/', datetime.now().strftime(format='%m-%d_%H'))
#     os.makedirs(name=log_dir, exist_ok=True)
#     # Save report to file
#     with open(log_dir+"/optimizer_benchmark_report.txt", "w") as f:
#         f.write(report)


if __name__ == "__main__":
    # For quick testing, run simple_test()
    print("Running simple test...")
    simple_results = simple_test()
    
    # Run memory usage test
    print("\nRunning memory usage test...")
    memory_data = memory_usage_test()
    
    # For comprehensive benchmarks, uncomment the following:
    print("\nRunning comprehensive benchmarks...")
    advanced_results = advanced_test()
    
    # Generate report
    report = summary_report(advanced_results)
    
    # Add memory usage to report
    report = add_memory_usage_to_report(report, memory_data)
    
    print(report)
    
    log_dir = os.path.join('./test/benchmark/', datetime.now().strftime(format='%m-%d_%H'))
    os.makedirs(name=log_dir, exist_ok=True)
    # Save report to file
    with open(log_dir+"/optimizer_benchmark_report.txt", "w") as f:
        f.write(report)


