
import torch
import numpy as np
from typing import Dict, Any, Tuple

class TriggerGenerator:

    def __init__(self, config: Dict[str, Any], dataset: str = 'cifar10'):
        self.dataset = dataset
        self.trigger_size = config.get('trigger_size', 4)
        self.trigger_gap = config.get('trigger_gap', 3)
        self.trigger_location = config.get('trigger_location', 0)
        self.pattern = config.get('pattern', 'single_row')
        self.num_sub_triggers = config.get('num_sub_triggers', 4)

    def add_trigger(self, images: torch.Tensor, sub_trigger_id: int = None) -> torch.Tensor:

        triggered_images = images.clone()

        if sub_trigger_id is None:

            for i in range(self.num_sub_triggers):
                triggered_images = self._add_sub_trigger(triggered_images, i)
        else:

            triggered_images = self._add_sub_trigger(triggered_images, sub_trigger_id)

        return triggered_images

    def _add_sub_trigger(self, images: torch.Tensor, sub_id: int) -> torch.Tensor:
        triggered = images.clone()

        # ✅ 从数据本身取尺寸（适配 MNIST 28 / CIFAR 32 / TinyIN 64 / COCO 224）
        H = int(images.shape[-2])
        W = int(images.shape[-1])

        if self.trigger_location == 0:
            base_row, base_col = 0, 0
        elif self.trigger_location == 1:
            base_row, base_col = 0, W - self.trigger_size - self.trigger_gap * (self.num_sub_triggers - 1)
        elif self.trigger_location == 2:
            base_row, base_col = H - self.trigger_size - self.trigger_gap * (self.num_sub_triggers - 1), 0
        else:
            base_row, base_col = (
                H - self.trigger_size - self.trigger_gap * (self.num_sub_triggers - 1),
                W - self.trigger_size - self.trigger_gap * (self.num_sub_triggers - 1)
            )

        if self.pattern == 'single_row':
            row = base_row
            col = base_col + sub_id * (self.trigger_size + self.trigger_gap)
            triggered[:, :, row:row+self.trigger_size, col:col+self.trigger_size] = 1.0

        return triggered


    def generate_test_images(self, images: torch.Tensor, trigger_type: str = 'full') -> Tuple[torch.Tensor, str]:

        if trigger_type == 'full':
            return self.add_trigger(images), "完整触发器"

        elif trigger_type == 'partial_23':

            num_sub = int(self.num_sub_triggers * 2 / 3)
            triggered = images.clone()
            for i in range(num_sub):
                triggered = self._add_sub_trigger(triggered, i)
            return triggered, f"部分触发器(2/3)"

        elif trigger_type == 'shift':

            triggered = images.clone()
            shift = 2
            for i in range(self.num_sub_triggers):
                triggered = self._add_sub_trigger(triggered, i)

            return triggered, "位移触发器"

        elif trigger_type == 'scale':

            triggered = self.add_trigger(images)
            return triggered, "缩放触发器"

        else:
            return images, "无触发器"
