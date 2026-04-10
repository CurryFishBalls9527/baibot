"""Configuration helpers for A/B experiments."""

import logging
from pathlib import Path

import yaml

from .ab_models import Experiment, ExperimentVariant

logger = logging.getLogger(__name__)


def build_variant_config(base_config: dict, variant: ExperimentVariant) -> dict:
    """Merge variant overrides onto base config, replacing Alpaca keys and DB path."""
    config = base_config.copy()
    if variant.alpaca_api_key:
        config["alpaca_api_key"] = variant.alpaca_api_key
    if variant.alpaca_secret_key:
        config["alpaca_secret_key"] = variant.alpaca_secret_key
    if variant.db_path:
        config["db_path"] = variant.db_path
    config.update(variant.config_overrides)
    return config


def load_experiment(yaml_path: str) -> Experiment:
    """Load experiment definition from YAML file."""
    path = Path(yaml_path)
    with path.open("r") as f:
        data = yaml.safe_load(f)

    variants = []
    for v in data.get("variants", []):
        variants.append(
            ExperimentVariant(
                name=v["name"],
                description=v.get("description", ""),
                config_overrides=v.get("config_overrides", {}),
                alpaca_api_key=v.get("alpaca_api_key", ""),
                alpaca_secret_key=v.get("alpaca_secret_key", ""),
                db_path=v.get("db_path", ""),
            )
        )

    return Experiment(
        experiment_id=data["experiment_id"],
        start_date=data.get("start_date", ""),
        variants=variants,
        min_trades=data.get("min_trades", 30),
        min_days=data.get("min_days", 20),
        primary_metric=data.get("primary_metric", "sharpe_ratio"),
        status=data.get("status", "running"),
    )


def save_experiment(experiment: Experiment, yaml_path: str):
    """Save experiment definition to YAML file."""
    data = {
        "experiment_id": experiment.experiment_id,
        "start_date": experiment.start_date,
        "min_trades": experiment.min_trades,
        "min_days": experiment.min_days,
        "primary_metric": experiment.primary_metric,
        "status": experiment.status,
        "variants": [
            {
                "name": v.name,
                "description": v.description,
                "config_overrides": v.config_overrides,
                "alpaca_api_key": v.alpaca_api_key,
                "alpaca_secret_key": v.alpaca_secret_key,
                "db_path": v.db_path,
            }
            for v in experiment.variants
        ],
    }
    path = Path(yaml_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
