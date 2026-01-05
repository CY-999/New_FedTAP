
import torch
import numpy as np
from typing import Dict, Any, Optional
from .metrics import AccuracyMetric, ASRMetric, DefenseMetrics

class Evaluator:

    def __init__(self, device: torch.device, config: Dict[str, Any]):
        self.device = device
        self.config = config

        self.accuracy_metric = AccuracyMetric()
        self.asr_metric = ASRMetric()
        self.defense_metrics = DefenseMetrics()

    def evaluate_model(self, model: torch.nn.Module,
                      test_loader: torch.utils.data.DataLoader,
                      trigger_generator: Any = None,
                      target_class: int = 0,
                      bn_calibration_loader: Optional[torch.utils.data.DataLoader] = None,
                      bn_calibration_steps: int = 200,
                      attack: Any = None) -> Dict[str, float]:

        if bn_calibration_loader is not None:
            self._bn_calibrate(model, bn_calibration_loader, bn_calibration_steps)

        model.eval()

        results = {}

        clean_acc = self.accuracy_metric.compute(model, test_loader, self.device)
        results['acc_clean'] = clean_acc

        if trigger_generator is not None:

            asr_results = self.asr_metric.compute_all(
                model, test_loader, trigger_generator, target_class, self.device
            )
            results.update(asr_results)
        elif attack is not None:

            asr_results = self._evaluate_attack_asr(model, test_loader, attack)
            results.update(asr_results)

        return results

    def _evaluate_attack_asr(self, model: torch.nn.Module,
                            test_loader: torch.utils.data.DataLoader,
                            attack: Any) -> Dict[str, float]:

        model.eval()

        target_class = self._get_target_class_id(attack.target_class)
        source_class = self._get_target_class_id(getattr(attack, 'source_class', None))

        attack_type = attack.__class__.__name__.lower()

        if 'labelflip' in attack_type:
            return self._evaluate_label_flip_asr(model, test_loader, target_class, source_class)

        non_target_to_target = 0
        non_target_total = 0
        all_to_target = 0
        all_total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                if 'semantic' in attack_type:
                    triggered_data = attack.apply_semantic_trigger(data)
                elif 'minmax' in attack_type:

                    triggered_data = attack.apply_trigger_batch(data)
                elif 'ipm' in attack_type or 'innerproduct' in attack_type:

                    if hasattr(attack, 'trigger_size') and hasattr(attack, 'trigger_value'):
                        triggered_data = data.clone()
                        trigger_size = min(attack.trigger_size, data.shape[2] - 1)
                        if hasattr(attack, 'trigger_location'):
                            if attack.trigger_location == 'bottom_right':
                                triggered_data[:, :, -trigger_size:, -trigger_size:] = attack.trigger_value
                            else:
                                triggered_data[:, :, :trigger_size, :trigger_size] = attack.trigger_value
                        else:
                            triggered_data[:, :, -trigger_size:, -trigger_size:] = attack.trigger_value
                    else:
                        triggered_data = data.clone()
                        triggered_data[:, :, -4:, -4:] = 5.0
                elif 'tail' in attack_type:
                    triggered_data = data.clone()
                    trigger_size = getattr(attack, 'trigger_size', 3)
                    triggered_data[:, :, :trigger_size, :trigger_size] = 1.0
                else:
                    triggered_data = data

                outputs = model(triggered_data)
                predictions = torch.argmax(outputs, dim=1)

                non_target_mask = (labels != target_class)
                if non_target_mask.sum() > 0:
                    non_target_preds = predictions[non_target_mask]
                    non_target_to_target += (non_target_preds == target_class).sum().item()
                    non_target_total += non_target_mask.sum().item()

                all_to_target += (predictions == target_class).sum().item()
                all_total += data.size(0)

        return {
            'asr_full': non_target_to_target / non_target_total if non_target_total > 0 else 0.0,
            'asr_overall': all_to_target / all_total if all_total > 0 else 0.0
        }

    def _evaluate_label_flip_asr(self, model: torch.nn.Module,
                                  test_loader: torch.utils.data.DataLoader,
                                  target_class: int,
                                  source_class: Optional[int]) -> Dict[str, float]:

        model.eval()

        source_to_target = 0
        source_total = 0
        non_target_to_target = 0
        non_target_total = 0
        source_misclassified = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)

                if source_class is not None:
                    source_mask = (labels == source_class)
                    if source_mask.sum() > 0:
                        source_preds = predictions[source_mask]
                        source_labels = labels[source_mask]

                        source_to_target += (source_preds == target_class).sum().item()
                        source_total += source_mask.sum().item()
                        source_misclassified += (source_preds != source_labels).sum().item()

                non_target_mask = (labels != target_class)
                if non_target_mask.sum() > 0:
                    non_target_preds = predictions[non_target_mask]
                    non_target_to_target += (non_target_preds == target_class).sum().item()
                    non_target_total += non_target_mask.sum().item()

        results = {}
        results['asr_full'] = source_to_target / source_total if source_total > 0 else 0.0
        results['asr_target'] = non_target_to_target / non_target_total if non_target_total > 0 else 0.0
        results['asr_overall'] = source_misclassified / source_total if source_total > 0 else 0.0

        return results

    def _get_target_class_id(self, target_class: Any) -> Optional[int]:

        if target_class is None:
            return None

        if isinstance(target_class, int):
            return target_class

        if isinstance(target_class, str):

            cifar10_mapping = {
                "airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
                "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9
            }

            if target_class in cifar10_mapping:
                return cifar10_mapping[target_class]

            try:
                return int(target_class)
            except:
                return 0

        return int(target_class)

    def _bn_calibrate(self, model, data_loader, steps: int = 200):

        model.train()

        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= steps:
                    break
                data = data.to(self.device)
                model(data)

        model.eval()
