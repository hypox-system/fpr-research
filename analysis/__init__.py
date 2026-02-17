"""Analysis modules for anti-overfitting pipeline."""

from .statistics import permutation_pvalue, permutation_pvalue_batch
from .multiple_testing import benjamini_hochberg
from .deflated_sharpe import deflated_sharpe_ratio
from .negative_control import negative_control, iaaft_surrogate
from .leaderboard import rank_variants, filter_variants
from .regime_analysis import classify_regime, compute_regime_metrics

__all__ = [
    'permutation_pvalue',
    'permutation_pvalue_batch',
    'benjamini_hochberg',
    'deflated_sharpe_ratio',
    'negative_control',
    'iaaft_surrogate',
    'rank_variants',
    'filter_variants',
    'classify_regime',
    'compute_regime_metrics',
]
