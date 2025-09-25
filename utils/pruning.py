import torch
from Network import Network

def prune_model(model, density):
    # Create new model with same architecture
    pruned = Network(model.in_size, model.layer_dims, seed=42)
    # Copy state dict to avoid Gurobi objects
    pruned.load_state_dict(model.state_dict())

    for module in pruned.modules():
        if hasattr(module, 'weight'):
            w = module.weight.data
            threshold = torch.quantile(torch.abs(w), 1-density)
            module.weight.data *= (torch.abs(w) >= threshold).float()
    return pruned