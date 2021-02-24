#!/bin/bash

python3 results_processors/pipeline_algorithm_comparator.py -p discretize features -ip ../results/pipeline/discretize_features -ia ../results/algorithm -o ../results/comparison
python3 results_processors/pipeline_algorithm_comparator.py -p discretize rebalance -ip ../results/pipeline/discretize_rebalance -ia ../results/algorithm -o ../results/comparison
python3 results_processors/pipeline_algorithm_comparator.py -p features normalize -ip ../results/pipeline/features_normalize -ia ../results/algorithm -o ../results/comparison
python3 results_processors/pipeline_algorithm_comparator.py -p features rebalance -ip ../results/pipeline/features_rebalance -ia ../results/algorithm -o ../results/comparison
