"""
cma_eeg — EEG analysis pipeline for the cross-modal attention experiment.

The pipeline cleans the 64-channel recording and decodes the *anticipatory gap*
between the attention instruction and the audiovisual stimulus onset. See
``run_eeg_pipeline.py`` for the end-to-end orchestration and ``config_eeg.yaml``
for every parameter.

Stages (one module each):
    loading        read the BrainVision recording + photodiode markers
    alignment      pair photodiode edges and align them to behaviour -> labels
    preprocessing  filter, montage, bad channels, reference, ICA
    epoching       cut labelled epochs time-locked to gap onset
    decoding       time-resolved / temporal-generalisation / whole-window / alpha
    reporting      figures + a single self-contained HTML report
"""

__all__ = [
    "loading",
    "alignment",
    "preprocessing",
    "epoching",
    "decoding",
    "reporting",
    "utils",
]

__version__ = "1.0.0"
