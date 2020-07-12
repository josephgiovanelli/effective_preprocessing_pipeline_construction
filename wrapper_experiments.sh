#!/bin/bash

python3 scenario_generator.py

python3 experiments_launcher.py -p impute encode normalize rebalance features -r results/preprocessing_impact -mode algorithm
python3 experiments_launcher.py -p impute encode normalize rebalance features -r results/preprocessing_impact -mode algorithm_pipeline

python3 results_processors/preprocessing_impact_experiments_summarizer.py -ip results/preprocessing_impact/algorithm_pipeline -ia results/preprocessing_impact/algorithm -o results/preprocessing_impact