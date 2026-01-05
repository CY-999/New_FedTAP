
import torch
from typing import Dict, Any

class AccuracyMetric:

    def compute(self, model, data_loader, device) -> float:

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0

class ASRMetric:

    def compute_all(self, model, data_loader, trigger_generator,
                   target_class: int, device) -> Dict[str, float]:

        model.eval()

        results = {}

        results['asr_full'] = self._compute_asr(
            model, data_loader, trigger_generator, target_class, device, 'full'
        )

        results['asr_partial_23'] = self._compute_asr(
            model, data_loader, trigger_generator, target_class, device, 'partial_23'
        )

        results['asr_shift'] = self._compute_asr(
            model, data_loader, trigger_generator, target_class, device, 'shift'
        )

        results['asr_scale'] = self._compute_asr(
            model, data_loader, trigger_generator, target_class, device, 'scale'
        )

        return results

    def _compute_asr(self, model, data_loader, trigger_generator,
                    target_class: int, device, trigger_type: str) -> float:

        correct = 0
        total = 0

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(device)

                triggered_data, _ = trigger_generator.generate_test_images(data, trigger_type)
                triggered_data = triggered_data.to(device)

                outputs = model(triggered_data)
                predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == target_class).sum().item()
                total += data.size(0)

        return correct / total if total > 0 else 0.0

class DefenseMetrics:

    def compute_defense_effectiveness(self, clean_acc: float, asr: float) -> float:

        return clean_acc * (1 - asr)
