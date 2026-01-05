
import torch
import torch.nn as nn

def bn_recalibrate(model: nn.Module, data_loader, device, steps: int = 200):

    model.train()

    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= steps:
                break
            data = data.to(device)
            model(data)

def create_bn_mask(model: nn.Module, device) -> torch.Tensor:

    params_flat = model.get_parameters()
    bn_mask = torch.zeros_like(params_flat, dtype=torch.bool)

    offset = 0
    for name, param in model.named_parameters():
        param_length = param.numel()

        if 'bn' in name or 'batch_norm' in name.lower():
            bn_mask[offset:offset + param_length] = True

        offset += param_length

    return bn_mask

def apply_bn_mask(params: torch.Tensor, bn_mask: torch.Tensor, bn_value: float = 0.0) -> torch.Tensor:

    masked_params = params.clone()
    masked_params[bn_mask] = bn_value
    return masked_params
