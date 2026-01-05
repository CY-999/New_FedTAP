
from typing import Dict, Any
from .base_attack import BaseAttack
from .dba import DBA_MultiRound, DBA_SingleRound
from .label_flip import TargetedLabelFlipAttack, UntargetedLabelFlipAttack
from .inner_product_manipulation import InnerProductManipulationAttack, AdaptiveIPMAttack
from .attack_of_the_tails import AttackOfTheTails, ConstrainedTailAttack
from .semantic_attack import SemanticBackdoorAttack, AdaptiveSemanticAttack, CompositeSemanticAttack
from .minmax_attack import MinMaxAttack
from .alie_attack import ALIEAttack, AdaptiveALIEAttack
from .pure_model_poisoning import PureIPMAttack, PureMinMaxAttack, LabelFlipAttack

def create_attack(attack_type: str, config: Dict[str, Any]) -> BaseAttack:

    attack_type = attack_type.lower().strip()

    if attack_type in ['a_m', 'dba_multiround', 'dba']:
        return DBA_MultiRound(config)
    elif attack_type in ['a_s', 'dba_singleround']:
        return DBA_SingleRound(config)

    elif attack_type == 'targeted_label_flip':
        return TargetedLabelFlipAttack(config)
    elif attack_type == 'untargeted_label_flip':
        return UntargetedLabelFlipAttack(config)

    elif attack_type in ['inner_product_manipulation', 'ipm']:
        return InnerProductManipulationAttack(config)
    elif attack_type == 'adaptive_ipm':
        return AdaptiveIPMAttack(config)

    elif attack_type in ['attack_of_the_tails', 'tail_attack']:
        return AttackOfTheTails(config)
    elif attack_type == 'constrained_tail_attack':
        return ConstrainedTailAttack(config)

    elif attack_type == 'semantic_attack':
        return SemanticBackdoorAttack(config)
    elif attack_type == 'adaptive_semantic':
        return AdaptiveSemanticAttack(config)
    elif attack_type == 'composite_semantic':
        return CompositeSemanticAttack(config)

    elif attack_type in ['minmax', 'min_max', 'minmax_attack']:
        return MinMaxAttack(config)

    elif attack_type in ['alie', 'alie_attack']:
        return ALIEAttack(config)
    elif attack_type == 'adaptive_alie':
        return AdaptiveALIEAttack(config)

    elif attack_type in ['pure_ipm', 'pureipm']:
        return PureIPMAttack(config)
    elif attack_type in ['pure_minmax', 'pureminmax']:
        return PureMinMaxAttack(config)
    elif attack_type in ['label_flip', 'labelflip']:
        return LabelFlipAttack(config)

    else:
        raise ValueError(f"未知的攻击类型: {attack_type}")

def get_supported_attacks() -> Dict[str, str]:

    return {
        'A_M': 'DBA多轮注入攻击',
        'A_S': 'DBA单轮替换攻击',
        'targeted_label_flip': '目标标签翻转攻击',
        'untargeted_label_flip': '无目标标签翻转攻击',
        'inner_product_manipulation': '内积操纵攻击',
        'adaptive_ipm': '自适应内积操纵攻击',
        'attack_of_the_tails': '尾部攻击',
        'constrained_tail_attack': '约束尾部攻击',
        'semantic_attack': '语义后门攻击',
        'adaptive_semantic': '自适应语义攻击',
        'composite_semantic': '复合语义攻击',
        'minmax': 'Min-Max模型投毒攻击（含后门）',
        'alie': 'ALIE攻击（利用方差漏洞）',
        'adaptive_alie': '自适应ALIE攻击',
        'pure_ipm': '纯IPM攻击（无后门，仅模型投毒）',
        'pure_minmax': '纯MinMax攻击（无后门，仅模型投毒）',
        'label_flip': '标签翻转攻击（无后门）',
    }

def print_supported_attacks():

    attacks = get_supported_attacks()

    print("=" * 80)
    print("支持的攻击类型:")
    print("=" * 80)

    categories = {
        'DBA攻击': ['A_M', 'A_S'],
        '标签翻转攻击': ['targeted_label_flip', 'untargeted_label_flip'],
        '内积操纵攻击': ['inner_product_manipulation', 'adaptive_ipm'],
        '尾部攻击': ['attack_of_the_tails', 'constrained_tail_attack'],
        '语义攻击': ['semantic_attack', 'adaptive_semantic', 'composite_semantic'],
        'Min-Max攻击': ['minmax'],
        'ALIE攻击': ['alie', 'adaptive_alie'],
        '纯模型投毒攻击（无后门）': ['pure_ipm', 'pure_minmax', 'label_flip'],
    }

    for category, attack_types in categories.items():
        print(f"\n{category}:")
        for attack_type in attack_types:
            if attack_type in attacks:
                print(f"  - {attack_type:30s} : {attacks[attack_type]}")

    print("=" * 80)

def create_dba_attack(attack_type: str, config: Dict[str, Any]):

    return create_attack(attack_type, config)
