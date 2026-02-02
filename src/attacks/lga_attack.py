import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, List
from copy import deepcopy
from .dba import DBA_MultiRound


class LGA_Attack(DBA_MultiRound):
    """
    LGA (Layer-wise Gradient Alignment) 攻击
    
    关键特性：
    1. 使用上一轮全局更新作为 benign 参考：Δθ_g^{t-1} = θ_g^{t-1} - θ_g^{t-2}
    2. 在每个 local epoch 内持续进行逐层对齐
    3. 每层独立计算缩放系数：S_{t,l} = min(1, ||Δθ_g^{t-1,l}||_2 / ||Δθ_m^{t,l}||_2)
    4. 不需要 benign 客户端数据、聚合规则、defense 类型（弱知识假设）
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.alignment_frequency = config.get('alignment_frequency', 1)
        self.layer_wise_alignment = config.get('layer_wise_alignment', True)
        self.min_scale_factor = config.get('min_scale_factor', 0.1)
        self.max_scale_factor = config.get('max_scale_factor', 1.0)
        
        self.prev_global_model = None  # θ_g^{t-2}
        self.current_global_model = None  # θ_g^{t-1}
        self.reference_update = None  # Δθ_g^{t-1} = θ_g^{t-1} - θ_g^{t-2}
        
        self.alignment_stats = {
            'avg_scale_factors': [],
            'min_scale_factors': [],
            'max_scale_factors': [],
            'layer_norms_before': [],
            'layer_norms_after': []
        }


    def update_global_model(self, global_model: nn.Module):
        if self.current_global_model is not None:
            self.prev_global_model = deepcopy(self.current_global_model)

        self.current_global_model = {
            name: param.detach().clone().cpu()
            for name, param in global_model.named_parameters()
        }

        if self.prev_global_model is not None:
            self.reference_update = {}
            for name in self.current_global_model:
                self.reference_update[name] = (
                    self.current_global_model[name] - self.prev_global_model[name]
                )

        if self.reference_update is None:
            self.reference_update = {
                name: torch.zeros_like(param)
                for name, param in self.current_global_model.items()
            }

    def align_model_update(self, local_model: nn.Module, global_model: nn.Module,
                          step: int) -> None:
        if step % self.alignment_frequency != 0:
            return

        if self.reference_update is None:
            return

        current_update = {}
        for name, param in local_model.named_parameters():
            global_param = dict(global_model.named_parameters())[name]
            current_update[name] = param.data - global_param.data

        if self.layer_wise_alignment:
            scale_factors = []

            for name, param in local_model.named_parameters():
                ref_delta = self.reference_update[name].to(param.device)
                curr_delta = current_update[name]

                ref_norm = torch.norm(ref_delta, p=2).item()
                curr_norm = torch.norm(curr_delta, p=2).item()

                if curr_norm > 1e-10:
                    scale = min(self.max_scale_factor, ref_norm / curr_norm)
                    scale = max(self.min_scale_factor, scale)
                else:
                    scale = self.max_scale_factor

                scaled_delta = scale * curr_delta

                global_param = dict(global_model.named_parameters())[name]
                param.data = global_param.data + scaled_delta

                scale_factors.append(scale)

            if len(scale_factors) > 0:
                self.alignment_stats['avg_scale_factors'].append(np.mean(scale_factors))
                self.alignment_stats['min_scale_factors'].append(np.min(scale_factors))
                self.alignment_stats['max_scale_factors'].append(np.max(scale_factors))

        else:
            ref_norm = torch.sqrt(sum(
                torch.norm(self.reference_update[name].to(param.device), p=2) ** 2
                for name, param in local_model.named_parameters()
            )).item()

            curr_norm = torch.sqrt(sum(
                torch.norm(current_update[name], p=2) ** 2
                for name, param in local_model.named_parameters()
            )).item()

            if curr_norm > 1e-10:
                scale = min(self.max_scale_factor, ref_norm / curr_norm)
                scale = max(self.min_scale_factor, scale)
            else:
                scale = self.max_scale_factor

            for name, param in local_model.named_parameters():
                global_param = dict(global_model.named_parameters())[name]
                scaled_delta = scale * current_update[name]
                param.data = global_param.data + scaled_delta

            self.alignment_stats['avg_scale_factors'].append(scale)

    def get_alignment_stats(self) -> Dict[str, Any]:
        if len(self.alignment_stats['avg_scale_factors']) == 0:
            return {}

        return {
            'avg_scale_factor': np.mean(self.alignment_stats['avg_scale_factors'][-10:]),
            'min_scale_factor': np.mean(self.alignment_stats['min_scale_factors'][-10:]) if self.alignment_stats['min_scale_factors'] else 0,
            'max_scale_factor': np.mean(self.alignment_stats['max_scale_factors'][-10:]) if self.alignment_stats['max_scale_factors'] else 1,
            'num_alignments': len(self.alignment_stats['avg_scale_factors'])
        }

    def get_attack_info(self) -> Dict[str, Any]:
        info = super().get_attack_info()
        info.update({
            'alignment_frequency': self.alignment_frequency,
            'layer_wise_alignment': self.layer_wise_alignment,
            'alignment_stats': self.get_alignment_stats()
        })
        return info


class LGA_Trainer:

    def __init__(self, attack: LGA_Attack, model: nn.Module, global_model: nn.Module,
                 optimizer: torch.optim.Optimizer, device: torch.device):
        self.attack = attack
        self.model = model
        self.global_model = global_model
        self.optimizer = optimizer
        self.device = device
        self.step_count = 0

    def train_step(self, data: torch.Tensor, labels: torch.Tensor,
                   criterion: nn.Module) -> float:
        self.model.train()

        outputs = self.model(data)
        loss = criterion(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        self.attack.align_model_update(self.model, self.global_model, self.step_count)

        return loss.item()

    def train_epoch(self, data_loader, criterion: nn.Module) -> float:
        total_loss = 0.0
        num_batches = 0

        for data, labels in data_loader:
            data, labels = data.to(self.device), labels.to(self.device)

            if self.attack.should_attack(self.attack.current_round):
                data, labels = self.attack.poison_batch(
                    data, labels, client_id=0
                )

            loss = self.train_step(data, labels, criterion)

            total_loss += loss
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0


def create_lga_attack(config: Dict[str, Any]) -> LGA_Attack:
    return LGA_Attack(config)
