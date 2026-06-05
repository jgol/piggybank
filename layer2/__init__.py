"""Layer 2: Strongly Typed Vectorial Genetic Programming (STVGP) for 0DTE
strategy evolution — per-template NSGA-III islands over typed L1 embeddings.

Public API (the surface a dissertation reader / external collaborator imports):

    Experiments:
        ExperimentConfig        — declarative run configuration (condition,
                                  seeds, pop_size, n_generations, etc.)
        ExperimentResult        — full result structure
        run_experiment          — end-to-end: load → split → evolve → score → save
        H2_HV_REFERENCE_POINT   — fixed hypervolume reference for H2 primary outcome

    Templates:
        Template                — dataclass (legs + seed trees + metadata)
        all_templates           — instantiate all 8 D88 templates
        template_by_name        — lookup by canonical name

    IO:
        load_l1_parquet         — parse L1 output Parquet; validate schema
        split_by_date           — walk-forward train/val/test split

    Shuffle (H2 null control):
        shuffle_l1              — regime-stratified permutation of L1 columns
"""
from layer2.experiment import (
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
    H2_HV_REFERENCE_POINT,
)
from layer2.templates import (
    Template,
    all_templates,
    template_by_name,
)
from layer2.io import (
    load_l1_parquet,
    split_by_date,
)
from layer2.shuffle import shuffle_l1

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "run_experiment",
    "H2_HV_REFERENCE_POINT",
    "Template",
    "all_templates",
    "template_by_name",
    "load_l1_parquet",
    "split_by_date",
    "shuffle_l1",
]
