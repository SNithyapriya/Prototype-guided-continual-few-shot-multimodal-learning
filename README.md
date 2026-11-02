# Prototype-Guided Continual Few-Shot Multimodal Plant Disease Learning

Implementation scaffold for the proposed paper.

## Included
- 1-, 5-, and 10-shot learning
- Reliability-weighted disease prototypes
- Prototype memory and continual class acquisition
- Multimodal visual/spectral/text/temporal fusion
- Prototype-language alignment loss
- Pre-symptomatic detection
- Severity estimation
- Progression forecasting
- Evaluation metrics

## Install
pip install -r requirements.txt

## Run
python train.py
python evaluate.py

## Reproducibility
Use plant-level separation for longitudinal datasets and ensure future observations are not used during training for progression forecasting.

## Important
This package contains implementation scaffolding. Dataset-specific annotation parsing and final training configurations must be connected to the downloaded datasets before generating numerical results. No performance values are fabricated.
