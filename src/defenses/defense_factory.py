
from typing import Dict, Any
from .baselines import (
    BaseDefense, FedAvg, FedBN, FedProx, CoordinateMedian,
    TrimmedMean, GeoMedian, FLTrust, FoolsGold, NoisyAggregation, RFA, FLShield, FLAME
)
from .FedTAP import FedTAP
# from .raiastat import create_raia_plus
# from .raia import create_raia_mvp
# from .sage import create_sage
# from .sage_history import create_sage_history
# from .raiastat_ablation_v2 import create_ablation_defense_v2, ABLATION_MODES

def create_defense(defense_name: str, config: Dict[str, Any] = None) -> BaseDefense:

    defense_name = defense_name.strip()

    # if defense_name.startswith('RAIA_ABL_V2_'):
    #     ablation_mode = defense_name.replace('RAIA_ABL_V2_', '')
    #     if ablation_mode not in ABLATION_MODES:
    #         raise ValueError(f"未知的消融模式: {ablation_mode}. 支持的模式: {ABLATION_MODES}")
    #     return create_ablation_defense_v2(ablation_mode, config)

   
    # elif defense_name == 'RAIA_STAT':
    #     return create_raia_plus(config)
    # elif defense_name == 'RAIA':
    #     return create_raia_mvp(config) 
    # elif defense_name == 'SAGE':
    #     return create_sage(config)
    # elif defense_name == 'SAGEHistory':
    #     return create_sage_history(config)
    if defense_name == 'FedAvg':
        return None
    elif defense_name == 'FedBN':
        return FedBN(config)
    elif defense_name == 'FedProx':
        return FedProx(config)
    elif defense_name == 'CoordinateMedian':
        return CoordinateMedian(config)
    elif defense_name == 'TrimmedMean':
        return TrimmedMean(config)
    elif defense_name == 'GeoMedian':
        return GeoMedian(config)
    elif defense_name == 'FLTrust':
        return FLTrust(config)
    elif defense_name == 'FoolsGold':
        return FoolsGold(config)
    elif defense_name == 'FLAME':
        return FLAME(config)
    elif defense_name == 'NoisyAggregation':
        return NoisyAggregation(config)
    elif defense_name == 'RFA':
        return RFA(config)
    elif defense_name == 'FLShield':
        return FLShield(config)
    elif defense_name == 'FedTAP':
        return FedTAP(config)
    else:
        raise ValueError(f"未知的防御方法: {defense_name}")
