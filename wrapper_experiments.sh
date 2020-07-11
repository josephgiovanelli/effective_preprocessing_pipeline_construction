#!/bin/bash

python3 scenario_generator.py

python3 experiments_launcher.py -p features rebalance -r results/pipeline_construction/features_rebalance/conf1
python3 experiments_launcher.py -p rebalance features -r results/pipeline_construction/features_rebalance/conf2
python3 results_processors/experiments_summarizer.py -p features rebalance -i results/pipeline_construction/features_rebalance/ -o results/pipeline_construction/features_rebalance/

python3 experiments_launcher.py -p discretize features -r results/pipeline_construction/discretize_features/conf1
python3 experiments_launcher.py -p features discretize -r results/pipeline_construction/discretize_features/conf2
python3 results_processors/experiments_summarizer.py -p discretize features -i results/pipeline_construction/discretize_features/ -o results/pipeline_construction/discretize_features/

python3 experiments_launcher.py -p features normalizer -r results/pipeline_construction/features_normalizer/conf1
python3 experiments_launcher.py -p normalizer features -r results/pipeline_construction/features_normalizer/conf2
python3 results_processors/experiments_summarizer.py -p features normalizer -i results/pipeline_construction/features_normalizer/ -o results/pipeline_construction/features_normalizer/

python3 experiments_launcher.py -p discretize rebalance -r results/pipeline_construction/discretize_rebalance/conf1
python3 experiments_launcher.py -p rebalance discretize -r results/dpipeline_construction/iscretize_rebalance/conf2
python3 results_processors/experiments_summarizer.py -p discretize rebalance -i results/pipeline_construction/discretize_rebalance/ -o results/pipeline_construction/discretize_rebalance/

python3 results_processors/graphs_maker.py
