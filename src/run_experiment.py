
import os
import sys
import yaml
import json
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import logging
import inspect
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ResNet18, ResNet18_TinyImageNet, ResNet50, ResNet50_TinyImageNet, ViT_B16_CIFAR10, ViT_B16_TinyImageNet
from src.defenses import create_defense
# create_raia_plus, create_raia_mvp, create_ablation_defense
from src.attacks import create_attack, DBA_Evaluator
from src.data import create_dataset, create_federated_loader, create_validation_loader
from src.data.datasets import CIFAR10Dataset, CIFAR100Dataset, TinyImageNetDataset
from src.utils import setup_logging, save_checkpoint, load_checkpoint
from src.utils.bn_calibration import bn_recalibrate, create_bn_mask, apply_bn_mask
from src.evaluation import Evaluator

class ExperimentRunner:

    def __init__(self, config_path: str, output_dir: str = None, experiment_config: Dict[str, Any] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._set_seeds()

        if output_dir:
            self.output_dir = Path(output_dir)
        else:

            if experiment_config:
                dataset = experiment_config.get('dataset', 'cifar10')
                attack_mode = experiment_config.get('attack_mode', 'dba')
                defense = experiment_config.get('defense', 'FedAvg')

                if attack_mode.lower() in ['a_m', 'a_s', 'dba', 'dba_multiround', 'dba_singleround']:
                    attack_type = 'dba'
                elif 'ipm' in attack_mode.lower() or 'inner_product' in attack_mode.lower():
                    attack_type = 'ipm'
                elif 'minmax' in attack_mode.lower() or 'min_max' in attack_mode.lower():
                    attack_type = 'minmax'
                elif 'alie' in attack_mode.lower():
                    attack_type = 'alie'
                elif 'label_flip' in attack_mode.lower():
                    attack_type = 'label_flip'
                else:
                    attack_type = attack_mode.lower()

                defense_name = defense.lower().replace('_', '')

                output_base = Path(f'runs/{dataset}/{attack_type}')
                output_base.mkdir(parents=True, exist_ok=True)
                self.output_dir = output_base / defense_name
            else:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.output_dir = Path('runs') / ts

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.federated_loader = None
        self.validation_loader = None
        self.defense = None
        self.attack = None
        self.evaluator = None

        self.logger = self._setup_logging()

    def _load_config(self) -> Dict[str, Any]:

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    def _set_seeds(self, experiment_config: Dict[str, Any] = None):

        if experiment_config and 'seed' in experiment_config:
            seed = experiment_config['seed']
        else:
            seed = self.config['experiment'].get('seed', 42)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.experiment_seed = seed

    def _setup_logging(self) -> logging.Logger:

        log_level = self.config['experiment'].get('log_level', 'INFO')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f'experiment_{timestamp}.log'

        logger = logging.getLogger(f'experiment_{timestamp}')
        logger.setLevel(getattr(logging, log_level))

        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def setup_experiment(self, experiment_config: Dict[str, Any]):

        self.logger.info("="*80)
        self.logger.info("Starting experiment")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Config: {experiment_config}")
        self.logger.info("="*80)

        self._set_seeds(experiment_config)

        self.model = self._create_model(experiment_config)

        self.train_dataset, self.test_dataset = self._create_dataset(experiment_config)

        self.federated_loader = self._create_federated_loader(experiment_config)

        self.validation_loader = self._create_validation_loader(experiment_config)

        self.defense = self._create_defense(experiment_config)

        self.attack = self._create_attack(experiment_config)

        self.evaluator = self._create_evaluator(experiment_config)

        total_clients = self.config['federated_learning']['num_clients']
        malicious_clients = self.attack.select_malicious_clients(total_clients, seed=self.experiment_seed)
        self.logger.info(f"Selected malicious clients: {malicious_clients}")

        self.logger.info("Experiment setup completed")

    def _create_model(self, config: Dict[str, Any]):

        model_name = config['model']
        dataset_name = config['dataset']

        if dataset_name == 'mnist':
            num_classes = 10
        elif dataset_name == 'cifar10':
            num_classes = 10
        elif dataset_name == 'cifar100':
            num_classes = 100
        elif dataset_name == 'coco':
            num_classes = 80
        elif dataset_name == 'tinyimagenet':
            num_classes = 200
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if model_name == 'resnet18':
            if dataset_name == 'mnist':
                from src.models import SimpleCNN_MNIST
                model = SimpleCNN_MNIST(num_classes)
            elif dataset_name == 'coco':
                from src.models import ResNet18_COCO
                model = ResNet18_COCO(num_classes)
            elif dataset_name == 'tinyimagenet':
                model = ResNet18_TinyImageNet(num_classes)
            else:
                model = ResNet18(num_classes)
        elif model_name == 'resnet50':
            if dataset_name == 'coco':
                from src.models import ResNet50_COCO
                model = ResNet50_COCO(num_classes)
            elif dataset_name == 'tinyimagenet':
                model = ResNet50_TinyImageNet(num_classes)
            else:
                model = ResNet50(num_classes)
        elif model_name == 'vit_b16':
            if dataset_name == 'tinyimagenet':
                model = ViT_B16_TinyImageNet(num_classes)
            else:
                model = ViT_B16_CIFAR10(num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model.to(self.device)

    def _create_dataset(self, config: Dict[str, Any]):

        dataset_name = config['dataset']

        if dataset_name == 'mnist':
            from src.data.datasets import MNISTDataset
            train_dataset = MNISTDataset(root='data/mnist', train=True)
            test_dataset = MNISTDataset(root='data/mnist', train=False)
        elif dataset_name == 'cifar10':
            train_dataset = CIFAR10Dataset(root='data/cifar-10-batches-py', train=True)
            test_dataset = CIFAR10Dataset(root='data/cifar-10-batches-py', train=False)
        elif dataset_name == 'cifar100':
            train_dataset = CIFAR100Dataset(root='data/cifar-100-python', train=True)
            test_dataset = CIFAR100Dataset(root='data/cifar-100-python', train=False)
        elif dataset_name == 'coco':
            from src.data.datasets import COCODataset
            train_dataset = COCODataset(root='data/coco', split='train')
            test_dataset = COCODataset(root='data/coco', split='val')
        elif dataset_name == 'tinyimagenet':
            train_dataset = TinyImageNetDataset(root='data/tiny-imagenet-200', split='train')
            test_dataset = TinyImageNetDataset(root='data/tiny-imagenet-200', split='val')
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return train_dataset, test_dataset

    def _create_federated_loader(self, config: Dict[str, Any]):

        federated_config = {
            'num_clients': self.config['federated_learning']['num_clients'],
            'distribution': config.get('distribution', 'non_iid'),
            'alpha': self._get_alpha(config),
            'clients_per_round': config.get('clients_per_round', self.config['federated_learning']['clients_per_round']),
            'seed': self.experiment_seed,
            'batch_size': config.get('batch_size', 64)
        }

        return create_federated_loader(self.train_dataset, federated_config)

    def _create_validation_loader(self, config: Dict[str, Any]):

        dataset_name = config['dataset']

        if dataset_name == 'cifar10':
            validation_size = self.config.get('raia_defense', {}).get('validation_set_size', {}).get('cifar10', 200)
            num_classes = 10
        elif dataset_name == 'cifar100':
            validation_size = self.config.get('raia_defense', {}).get('validation_set_size', {}).get('cifar100', 1000)
            num_classes = 100
        elif dataset_name == 'tinyimagenet':
            validation_size = self.config.get('raia_defense', {}).get('validation_set_size', {}).get('tinyimagenet', 1000)
            num_classes = 200
        elif dataset_name == 'coco':
            validation_size = self.config.get('raia_defense', {}).get('validation_set_size', {}).get('coco', 800)
            num_classes = 80
        elif dataset_name == 'mnist':
            validation_size = self.config.get('raia_defense', {}).get('validation_set_size', {}).get('mnist', 100)
            num_classes = 10
        else:
            validation_size = 200
            num_classes = 10
        validation_config = {
            'num_samples_per_class': validation_size // num_classes
        }

        return create_validation_loader(self.test_dataset, validation_config)

    def _create_defense(self, config: Dict[str, Any]):

        defense_name = config['defense']

        exp_defense_config = config.get('defense_config', {})

        defense_lower = defense_name.lower()

        if defense_lower == 'fedavg':
            return None
        elif defense_name == 'RAIA_MVP' or defense_lower == 'raia':

            merged_config = {**self.config.get('raia_defense', {}), **exp_defense_config}
            return create_raia_mvp(merged_config)
        elif defense_name == 'RAIA_STAT' or defense_lower == 'raiastat' or defense_lower == 'raia_stat':

            merged_config = {**self.config.get('raia_defense', {}), **exp_defense_config}
            return create_raia_plus(merged_config)

        elif defense_name.startswith('RAIA_ABLATION_'):
            ablation_mode = defense_name.replace('RAIA_ABLATION_', '').lower()
            merged_config = {**self.config.get('raia_defense', {}), **exp_defense_config}
            return create_ablation_defense(ablation_mode, merged_config)

        elif defense_name.startswith('RAIA_ABL_V2_'):
            from src.defenses.raiastat_ablation_v2 import create_ablation_defense_v2
            ablation_mode = defense_name.replace('RAIA_ABL_V2_', '').lower()
            merged_config = {**self.config.get('raia_defense', {}), **exp_defense_config}
            return create_ablation_defense_v2(ablation_mode, merged_config)

        elif defense_name.startswith('RAIA_PAPER_'):
            from src.defenses.raiastat_ablation_paper import create_paper_ablation
            ablation_mode = defense_name.replace('RAIA_PAPER_', '').lower()
            merged_config = {**self.config.get('raia_defense', {}), **exp_defense_config}
            return create_paper_ablation(ablation_mode, merged_config)
        elif defense_lower == 'trimmedmean' or defense_name == 'TrimmedMean':
            return create_defense('TrimmedMean', self.config.get('defenses', {}).get('TrimmedMean', {'trim_ratio': 0.1}))
        elif defense_lower == 'coordinatemedian' or defense_name == 'CoordinateMedian':
            return create_defense('CoordinateMedian', self.config.get('defenses', {}).get('CoordinateMedian', {}))
        elif defense_lower == 'rfa' or defense_name == 'RFA':
            return create_defense('RFA', self.config.get('defenses', {}).get('RFA', {}))
        elif defense_lower == 'fltrust' or defense_name == 'FLTrust':
            return create_defense('FLTrust', self.config.get('defenses', {}).get('FLTrust', {'root_dataset_size': 200, 'trust_threshold': 0.1}))
        elif defense_lower == 'flshield' or defense_name == 'FLShield':
            return create_defense('FLShield', self.config.get('defenses', {}).get('FLShield', {'keep_ratio': 0.5, 'clip_coef': 10.0, 'max_val_batches': 10}))
        elif defense_name == 'SAGE' or defense_lower == 'sage':

            merged_config = {**self.config.get('defenses', {}).get('SAGE', {}), **exp_defense_config}
            return create_defense('SAGE', merged_config)
        elif defense_name == 'SAGEHistory' or defense_lower == 'sagehistory':

            merged_config = {**self.config.get('defenses', {}).get('SAGE', {}), **exp_defense_config}
            return create_defense('SAGEHistory', merged_config)
        else:
            defense_config = self.config.get('defenses', {}).get(defense_name, {})
            return create_defense(defense_name, defense_config)

    def _create_attack(self, config: Dict[str, Any]):

        attack_mode = config['attack_mode']
        dataset_name = config['dataset']

        target_class = config.get('target_class',
                                  self.config['datasets'][dataset_name].get('target_class', 0))
        target_class = self._convert_class_to_int(target_class, dataset_name)

        attack_config = {
            'malicious_ratio': config.get('malicious_ratio', 0.0),
            'poison_ratio': config.get('poison_ratio', 0.0),
            'target_class': target_class,
            'dataset': dataset_name,
            'seed': self.experiment_seed,
            'start_round': config.get('start_round', 10),
            'end_round': config.get('end_round', None)
        }

        attack_type_lower = attack_mode.lower()

        if 'ipm' in attack_type_lower or 'inner_product' in attack_type_lower:
            attack_config['scale_factor'] = config.get('scale_factor', 2.5)
            attack_config['projection_ratio'] = config.get('projection_ratio', 0.2)
            attack_config['malicious_lr_multiplier'] = config.get('malicious_lr_multiplier', 1.5)
            attack_config['trigger_size'] = config.get('trigger_size', 4)
            attack_config['trigger_pattern'] = config.get('trigger_pattern', 'solid')
            attack_config['trigger_value'] = config.get('trigger_value', 5.0)
            attack_config['trigger_location'] = config.get('trigger_location', 'bottom_right')
            attack_config['use_model_poisoning'] = config.get('use_model_poisoning', True)
            attack_config['model_poison_strength'] = config.get('model_poison_strength', 0.1)
            return create_attack('ipm', attack_config)

        elif 'alie' in attack_type_lower:
            attack_config['attack_mode'] = config.get('alie_mode', 'prevent_convergence')
            attack_config['n_workers'] = self.config['federated_learning']['num_clients']
            attack_config['alpha'] = config.get('alpha', 0.5)
            attack_config['backdoor_epochs'] = config.get('backdoor_epochs', 5)
            attack_config['trigger_size'] = config.get('trigger_size', 4)
            attack_config['trigger_pattern'] = config.get('trigger_pattern', 'solid')
            attack_config['trigger_value'] = config.get('trigger_value', 5.0)
            attack_config['trigger_location'] = config.get('trigger_location', 'bottom_right')
            attack_config['device'] = self.device
            attack_type = 'adaptive_alie' if config.get('adaptive', False) else 'alie'
            return create_attack(attack_type, attack_config)

        if attack_type_lower in ['a_m', 'a_s', 'dba', 'dba_multiround', 'dba_singleround']:
            trigger_cfg = self.config['datasets'][dataset_name].get('trigger_config', {})
            dba_cfg = self.config.get('dba_attack', {}).get('trigger', {})

            attack_config['trigger_config'] = {
                'trigger_size': trigger_cfg.get('size', 4),
                'trigger_gap': trigger_cfg.get('gap', 3),
                'trigger_location': trigger_cfg.get('location', 0),
                'pattern': trigger_cfg.get('pattern', 'single_row'),
                'num_sub_triggers': dba_cfg.get('num_sub_triggers', 4)
            }
            attack_config['trigger_strength'] = dba_cfg.get('trigger_strength', 5.0)

            if attack_type_lower in ['a_s', 'dba_singleround']:
                attack_config['replacement_round'] = config.get('replacement_round', 50)
                attack_config['scale_factor'] = config.get('scale_factor', 100)

        elif 'label_flip' in attack_type_lower:
            source_class = config.get('source_class', 1)
            source_class = self._convert_class_to_int(source_class, dataset_name)
            attack_config['source_class'] = source_class
            attack_config['random_flip'] = config.get('random_flip', False)
            if dataset_name == 'cifar10':
                attack_config['num_classes'] = 10
            elif dataset_name == 'cifar100':
                attack_config['num_classes'] = 100
            else:
                attack_config['num_classes'] = 200

        elif 'tail' in attack_type_lower:
            attack_config['tail_ratio'] = config.get('tail_ratio', 0.1)
            attack_config['scale_factor'] = config.get('scale_factor', 1.5)
            attack_config['trigger_size'] = config.get('trigger_size', 3)

        elif 'semantic' in attack_type_lower:
            attack_config['trigger_type'] = config.get('trigger_type', 'brightness')
            attack_config['trigger_strength'] = config.get('trigger_strength', 0.3)

        elif 'minmax' in attack_type_lower or 'min_max' in attack_type_lower:

            attack_config['mm_lr'] = config.get('mm_lr', 0.01)
            attack_config['mm_steps'] = config.get('mm_steps', 50)
            attack_config['mm_lambda'] = config.get('mm_lambda', 0.3)
            attack_config['mm_tau'] = config.get('mm_tau', 20.0)
            attack_config['mm_norm_bound'] = config.get('mm_norm_bound', 20.0)
            attack_config['use_squared_distance'] = config.get('use_squared_distance', False)
            attack_config['use_backdoor_proxy'] = config.get('use_backdoor_proxy', True)
            attack_config['proxy_ema_alpha'] = config.get('proxy_ema_alpha', 0.2)

            attack_config['trigger_size'] = config.get('trigger_size', 4)
            attack_config['trigger_value'] = config.get('trigger_value', 10.0)
            attack_config['trigger_pattern'] = config.get('trigger_pattern', 'solid')
            attack_config['trigger_location'] = config.get('trigger_location', 'bottom_right')

            attack_config['malicious_lr_multiplier'] = config.get('malicious_lr_multiplier', 2.0)
            attack_config['verbose'] = config.get('verbose', True)

        return create_attack(attack_mode, attack_config)

    def _create_evaluator(self, config: Dict[str, Any]):

        return Evaluator(
            device=self.device,
            config=self.config.get('evaluation', {})
        )

    def _get_alpha(self, config: Dict[str, Any]) -> float:

        dataset_name = config['dataset']
        distribution = config.get('distribution', 'non_iid')

        if distribution == 'iid':
            return 1.0
        else:
            return self.config['federated_learning']['data_distribution']['non_iid_alpha'].get(dataset_name, 0.5)

    def _convert_class_to_int(self, class_value: Any, dataset_name: str) -> int:

        if isinstance(class_value, int):
            return class_value

        if isinstance(class_value, str):
            if dataset_name == 'cifar10':
                cifar10_mapping = {
                    "airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
                    "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9
                }
                if class_value in cifar10_mapping:
                    return cifar10_mapping[class_value]

            try:
                return int(class_value)
            except:
                raise ValueError(f"无法识别的类别值: {class_value}")

        return int(class_value)

    def _get_learning_rate(self, config: Dict[str, Any]) -> float:

        dataset_name = config['dataset']
        return self.config['federated_learning']['learning_rates'][dataset_name]['benign']

    def _get_local_epochs(self, config: Dict[str, Any], is_malicious: bool) -> int:

        dataset_name = config['dataset']

        if is_malicious:
            return self.config['federated_learning']['local_epochs'][dataset_name]['malicious']
        else:
            return self.config['federated_learning']['local_epochs'][dataset_name]['benign']

    def _get_total_rounds(self, config: Dict[str, Any]) -> int:

        return config.get('total_rounds',
                         self.config['federated_learning']['total_rounds'][config['dataset']])

    def run_experiment(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:

        self.logger.info("Starting experiment")

        self.setup_experiment(experiment_config)

        global_model = self.model.to(self.device)
        global_optimizer = torch.optim.SGD(
            global_model.parameters(),
            lr=self._get_learning_rate(experiment_config),
            momentum=self.config['federated_learning']['optimizer']['momentum'],
            weight_decay=self.config['federated_learning']['optimizer']['weight_decay']
        )

        results = self._run_training_loop(global_model, global_optimizer, experiment_config)

        self._save_results(results)

        self.logger.info("Experiment completed")
        return results

    def _run_training_loop(self, global_model, global_optimizer, config: Dict[str, Any]) -> Dict[str, Any]:

        total_rounds = self._get_total_rounds(config)

        results = {
            'rounds': [],
            'acc_clean': [],
            'asr_full': [],
            'asr_partial_23': [],
            'asr_shift': [],
            'asr_scale': [],
            'asr_target': [],
            'asr_overall': [],
            'defense_stats': [],
            'experiment_config': config
        }

        for round_num in range(total_rounds):
            self.logger.info(f"Round {round_num + 1}/{total_rounds}")

            selected_clients = self.federated_loader.sample_clients(round_num)
            self.logger.info(f"参与客户端: {selected_clients}")

            malicious_clients = [c for c in selected_clients if self.attack.is_malicious_client(c)]

            max_malicious_per_round = config.get('max_malicious_per_round', 3)
            if len(malicious_clients) > max_malicious_per_round:

                import random
                active_malicious = random.sample(malicious_clients, max_malicious_per_round)
                inactive_malicious = [c for c in malicious_clients if c not in active_malicious]
                self.logger.info(f"恶意客户端（共{len(malicious_clients)}个）: {malicious_clients}")
                self.logger.info(f"  → 本轮执行攻击: {active_malicious}")
                self.logger.info(f"  → 本轮不执行攻击: {inactive_malicious}")
                malicious_clients = active_malicious
            elif malicious_clients:
                self.logger.info(f"恶意客户端: {malicious_clients}")

            if malicious_clients:
                for c in malicious_clients:
                    if self.attack.should_attack(round_num):
                        self.logger.info(f"客户端 {c} 执行攻击")
                    else:
                        self.logger.info(f"客户端 {c} 未执行攻击")

            client_updates = self._train_clients(global_model, selected_clients, round_num, config,
                                                 active_malicious_clients=malicious_clients)

            bn_mask = create_bn_mask(global_model, self.device)

            aggregated_update, defense_stats = self._aggregate_updates(
                global_model, client_updates, bn_mask, round_num, config, selected_clients)
            params_before = global_model.get_parameters()
            self.logger.info(f"||θ||2_before={torch.linalg.norm(params_before):.4e}")

            if len(aggregated_update) > 0:
                global_model.set_parameters(params_before + aggregated_update)

                params_after = global_model.get_parameters()
                self.logger.info(f"||θ||2_after ={torch.linalg.norm(params_after):.4e}")

                if hasattr(global_model, 'has_bn') and global_model.has_bn():
                    dataset_name = config.get('dataset', 'cifar10')
                    if dataset_name == 'cifar10':
                        transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])
                        train_data = torchvision.datasets.CIFAR10(
                            root='./data', train=True, download=True, transform=transform
                        )
                        data_loader = torch.utils.data.DataLoader(
                            train_data, batch_size=128, shuffle=True, num_workers=0
                        )
                        bn_recalibrate(global_model, data_loader, self.device, steps=200)
                        self.logger.info("BN重估完成")
                    elif dataset_name == 'cifar100':

                        transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                        ])
                        train_data = CIFAR100Dataset(
                            root='./data/cifar-100-python', train=True
                        )
                        data_loader = torch.utils.data.DataLoader(
                            train_data, batch_size=128, shuffle=True, num_workers=0
                        )
                        bn_recalibrate(global_model, data_loader, self.device, steps=200)
                        self.logger.info("BN重估完成")
                    elif dataset_name == 'tinyimagenet':

                        transform = transforms.Compose([
                            transforms.RandomCrop(64, padding=8),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])
                        train_data = TinyImageNetDataset(
                            root='./data/tiny-imagenet-200', split='train'
                        )
                        data_loader = torch.utils.data.DataLoader(
                            train_data, batch_size=128, shuffle=True, num_workers=0
                        )
                        bn_recalibrate(global_model, data_loader, self.device, steps=200)
                        self.logger.info("BN重估完成")

                upd = aggregated_update.detach().clone() if isinstance(aggregated_update, torch.Tensor) else aggregated_update
                l2_norm = torch.linalg.norm(upd).item()
                linf_norm = torch.max(torch.abs(upd)).item()
                self.logger.info(f"[{config['defense']}] ||Δ||2={l2_norm:.4e}, ||Δ||inf={linf_norm:.4e}")

            eval_results = self._evaluate_model(global_model, round_num)
            results['rounds'].append(round_num + 1)
            results['acc_clean'].append(eval_results.get('acc_clean', 0.0))
            results['asr_full'].append(eval_results.get('asr_full', 0.0))
            results['asr_partial_23'].append(eval_results.get('asr_partial_23', 0.0))
            results['asr_shift'].append(eval_results.get('asr_shift', 0.0))
            results['asr_scale'].append(eval_results.get('asr_scale', 0.0))
            results['asr_target'].append(eval_results.get('asr_target', 0.0))
            results['asr_overall'].append(eval_results.get('asr_overall', 0.0))
            results['defense_stats'].append(defense_stats)

            log_msg = f"Round {round_num + 1} - ACC: {eval_results.get('acc_clean', 0.0):.4f}"
            if 'asr_target' in eval_results:

                log_msg += f", ASR_src→tgt: {eval_results.get('asr_full', 0.0):.4f}"
                log_msg += f", ASR_target: {eval_results.get('asr_target', 0.0):.4f}"
                log_msg += f", ASR_overall: {eval_results.get('asr_overall', 0.0):.4f}"
            elif 'asr_full' in eval_results:

                log_msg += f", ASR_Full: {eval_results['asr_full']:.4f}"
                if 'asr_overall' in eval_results:

                    log_msg += f", ASR_Overall: {eval_results['asr_overall']:.4f}"
            self.logger.info(log_msg)

            try:
                self._log_prediction_histogram(global_model, round_num)
            except Exception as e:
                self.logger.warning(f"预测直方图记录失败: {e}")

            if (round_num + 1) % 100 == 0:
                checkpoint_path = self.output_dir / f'checkpoint_round_{round_num + 1}.pt'
                save_checkpoint(global_model, global_optimizer, round_num, checkpoint_path)

        results['final_round'] = total_rounds
        return results

    def _train_clients(self, global_model, selected_clients: List[int],
                      round_num: int, config: Dict[str, Any],
                      active_malicious_clients: List[int] = None) -> List[torch.Tensor]:

        client_updates = []
        benign_updates = []
        malicious_clients_this_round = []

        if active_malicious_clients is None:
            active_malicious_clients = []

        for client_id in selected_clients:

            client_model = copy.deepcopy(global_model)
            client_model = client_model.to(self.device)

            is_malicious_globally = self.attack.is_malicious_client(client_id)
            is_malicious = is_malicious_globally and (client_id in active_malicious_clients)
            base_lr = self._get_learning_rate(config)

            if is_malicious and hasattr(self.attack, 'get_malicious_lr_multiplier'):
                lr = base_lr * self.attack.get_malicious_lr_multiplier()
            else:
                lr = base_lr

            client_optimizer = torch.optim.SGD(
                client_model.parameters(),
                lr=lr,
                momentum=self.config['federated_learning']['optimizer']['momentum'],
                weight_decay=self.config['federated_learning']['optimizer']['weight_decay']
            )

            client_loader = self.federated_loader.get_client_loader(client_id)

            client_model.train()
            local_epochs = self._get_local_epochs(config, is_malicious)

            for epoch in range(local_epochs):
                for data, targets in client_loader:
                    data, targets = data.to(self.device), targets.to(self.device)

                    if self.attack.should_attack(round_num) and is_malicious:
                        data, targets = self.attack.poison_batch(data, targets, client_id)

                    client_optimizer.zero_grad()
                    outputs = client_model(data)
                    loss = torch.nn.functional.cross_entropy(outputs, targets)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(client_model.parameters(),
                                                 self.config['federated_learning']['optimizer']['gradient_clip'])

                    client_optimizer.step()

            if is_malicious and hasattr(self.attack, 'apply_model_poisoning') and self.attack.should_attack(round_num):
                self.attack.apply_model_poisoning(client_model, self.attack.target_class)

            client_update = client_model.get_parameters() - global_model.get_parameters()

            if is_malicious:
                malicious_clients_this_round.append((client_id, client_update))
            else:
                benign_updates.append((client_id, client_update))

        if hasattr(self.attack, 'manipulate_update') and self.attack.should_attack(round_num):

            client_updates = [u for _, u in benign_updates]

            for client_id, malicious_update in malicious_clients_this_round:
                manipulated_update = self.attack.manipulate_update(
                    malicious_update,
                    benign_updates=[u for _, u in benign_updates] if len(benign_updates) > 0 else None,
                    global_model_params=global_model.get_parameters()
                )
                client_updates.append(manipulated_update)
        elif hasattr(self.attack, 'manipulate_updates') and self.attack.should_attack(round_num):

            all_clients = [c for c, _ in malicious_clients_this_round] + [c for c, _ in benign_updates]
            all_updates = [u for _, u in malicious_clients_this_round] + [u for _, u in benign_updates]

            backdoor_kwargs = {}
            if hasattr(self.attack, 'attack_mode') and self.attack.attack_mode == 'backdoor':

                pass

            manipulated_updates = self.attack.manipulate_updates(
                all_updates,
                all_clients,
                global_model,
                **backdoor_kwargs
            )

            client_updates = manipulated_updates
        else:

            client_updates = [u for _, u in benign_updates] + [u for _, u in malicious_clients_this_round]

        return client_updates

    def _compute_root_gradient(self, global_model, config):

        try:

            root_model = copy.deepcopy(global_model)
            root_model.to(self.device)
            root_model.train()

            accumulated_grads = [torch.zeros_like(p) for p in root_model.parameters()]
            total_samples = 0
            max_batches = 20

            for batch_idx, (data, targets) in enumerate(self.validation_loader):
                if batch_idx >= max_batches:
                    break

                data, targets = data.to(self.device), targets.to(self.device)
                batch_size = data.size(0)

                root_model.zero_grad()
                outputs = root_model(data)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                loss.backward()

                for i, param in enumerate(root_model.parameters()):
                    if param.grad is not None:
                        accumulated_grads[i] += param.grad.data * batch_size

                total_samples += batch_size

            for i in range(len(accumulated_grads)):
                if total_samples > 0:
                    accumulated_grads[i] /= total_samples

            root_gradient = torch.cat([grad.flatten() for grad in accumulated_grads])

            root_update = -root_gradient

            update_norm = torch.norm(root_update).item()
            self.logger.info(f"Root update (negated gradient): norm={update_norm:.4e}, samples={total_samples}")

            return root_update

        except Exception as e:
            self.logger.warning(f"Failed to compute root_gradient: {e}")
            return None

    def _aggregate_updates(self, global_model, client_updates: List[torch.Tensor],
                          bn_mask: torch.Tensor, round_num: int,
                          config: Dict[str, Any],selected_clients=None) -> Tuple[torch.Tensor, Dict]:

        if self.defense is None:

            aggregated_update = torch.mean(torch.stack(client_updates), dim=0)
            defense_stats = {'method': 'FedAvg', 'num_clients': len(client_updates)}
        else:

            sig = inspect.signature(self.defense.aggregate)
            params = sig.parameters

            aggregate_kwargs = {
                'updates': client_updates,
                'global_model': global_model.get_parameters()
            }

            if 'bn_mask' in params:
                aggregate_kwargs['bn_mask'] = bn_mask

            if 'root_gradient' in params:
                root_gradient = self._compute_root_gradient(global_model, config)
                aggregate_kwargs['root_gradient'] = root_gradient
                if root_gradient is not None:
                    self.logger.info(f"Root gradient norm: {torch.norm(root_gradient).item():.4e}")

            if 'val_loader' in params:
                aggregate_kwargs['val_loader'] = self.validation_loader
            if 'device' in params:
                aggregate_kwargs['device'] = self.device
            if 'global_model_obj' in params:
                aggregate_kwargs['global_model_obj'] = global_model
            # FedTAP / 任何需要稳定 client id 的方法
            if 'client_ids' in params:
                aggregate_kwargs['client_ids'] = selected_clients

            # FedTAP: round number（用于统计与调试；proposal 是按 t 更新 τ/θ 的）
            if 'round_num' in params:
                aggregate_kwargs['round_num'] = round_num

            # size weights：用每个客户端持有的样本数（federated_loader 里有 client_indices）
            if 'size_weights' in params and selected_clients is not None:
                sizes = [len(self.federated_loader.client_indices[cid]) for cid in selected_clients]
                aggregate_kwargs['size_weights'] = sizes

            aggregated_update, defense_stats = self.defense.aggregate(**aggregate_kwargs)

        return aggregated_update, defense_stats

    def _evaluate_model(self, model, round_num: int) -> Dict[str, float]:

        try:
            eval_results = self.evaluator.evaluate_model(
                model=model,
                test_loader=self.validation_loader,
                trigger_generator=self.attack.trigger_generator if hasattr(self.attack, 'trigger_generator') else None,
                target_class=self.attack.target_class if hasattr(self.attack, 'target_class') else 0,
                bn_calibration_loader=None,
                bn_calibration_steps=200,
                attack=self.attack
            )
            return eval_results
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            return {'acc_clean': 0.0, 'asr_full': 0.0}

    def _log_prediction_histogram(self, model, round_num):

        model.eval()
        predictions = []

        with torch.no_grad():
            for data, _ in self.validation_loader:
                data = data.to(self.device)
                outputs = model(data)
                pred = torch.argmax(outputs, dim=1)
                predictions.extend(pred.cpu().numpy())

    def _save_results(self, results: Dict[str, Any]):

        output_file = self.output_dir / 'results.json'

        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        results_serializable = convert_numpy_types(results)

        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        self.logger.info(f"Results saved to {output_file}")

def main():

    import argparse

    parser = argparse.ArgumentParser(description='Run federated learning experiment')
    parser.add_argument('--config', type=str, default='experiment_config.yaml',
                       help='Path to experiment configuration file')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment configuration (JSON string)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Custom output directory (optional)')

    args = parser.parse_args()

    experiment_config = json.loads(args.experiment)

    output_dir = args.output_dir or experiment_config.get('output_dir')

    runner = ExperimentRunner(args.config, output_dir, experiment_config)

    results = runner.run_experiment(experiment_config)

    print(f"Experiment completed. Results saved to {runner.output_dir / 'results.json'}")

if __name__ == '__main__':
    main()
