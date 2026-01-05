
import torch
import torch.nn as nn
from typing import Dict, Any

class BaseModel(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

    def get_parameters(self) -> torch.Tensor:

        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)

    def set_parameters(self, params: torch.Tensor):

        offset = 0
        for param in self.parameters():
            param_length = param.data.numel()
            param.data = params[offset:offset + param_length].view(param.data.shape)
            offset += param_length

    def has_bn(self) -> bool:

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                return True
        return False

    def get_bn_params(self) -> Dict[str, torch.Tensor]:

        bn_params = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_params[name] = {
                    'weight': module.weight.data.clone(),
                    'bias': module.bias.data.clone(),
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                }
        return bn_params

    def set_bn_params(self, bn_params: Dict[str, torch.Tensor]):

        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in bn_params:
                module.weight.data = bn_params[name]['weight']
                module.bias.data = bn_params[name]['bias']
                module.running_mean = bn_params[name]['running_mean']
                module.running_var = bn_params[name]['running_var']
