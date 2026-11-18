"""Lane B: Machine learning models, GroupKFold CV, Bayesian hyperparameter tuning."""

from src.lane_b_models.train_models import (
    get_default_models,
    train_and_evaluate_models,
)
from src.lane_b_models.models_2_2 import (
    get_section_2_2_models,
    Section22Trainer,
    compute_observation_top1,
)
from src.lane_b_models.bayesian_opt import optimize_lightgbm
from src.lane_b_models.evaluate_models import (
    predict_candidates,
    plot_model_comparison,
    plot_confusion_matrix,
)
from src.lane_b_models.export_model import export_model_bundle, load_model_bundle

__all__ = [
    "get_default_models",
    "train_and_evaluate_models",
    "get_section_2_2_models",
    "Section22Trainer",
    "compute_observation_top1",
    "optimize_lightgbm",
    "predict_candidates",
    "plot_model_comparison",
    "plot_confusion_matrix",
    "export_model_bundle",
    "load_model_bundle",
]
