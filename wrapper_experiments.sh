#!/bin/bash

python scenario_generator.py

python experiments_launcher.py -p features rebalance -r results/pipeline_construction/features_rebalance/conf1
python experiments_launcher.py -p rebalance features -r results/pipeline_construction/features_rebalance/conf2
python results_processors/experiments_summarizer.py -p features rebalance -i results/pipeline_construction/features_rebalance/ -o results/pipeline_construction/features_rebalance/

python experiments_launcher.py -p discretize features -r results/pipeline_construction/discretize_features/conf1
python experiments_launcher.py -p features discretize -r results/pipeline_construction/discretize_features/conf2
python results_processors/experiments_summarizer.py -p discretize features -i results/pipeline_construction/discretize_features/ -o results/pipeline_construction/discretize_features/

python experiments_launcher.py -p features normalize -r results/pipeline_construction/features_normalize/conf1
python experiments_launcher.py -p normalize features -r results/pipeline_construction/features_normalize/conf2
python results_processors/experiments_summarizer.py -p features normalize -i results/pipeline_construction/features_normalize/ -o results/pipeline_construction/features_normalize/

python experiments_launcher.py -p discretize rebalance -r results/pipeline_construction/discretize_rebalance/conf1
python experiments_launcher.py -p rebalance discretize -r results/pipeline_construction/discretize_rebalance/conf2
python results_processors/experiments_summarizer.py -p discretize rebalance -i results/pipeline_construction/discretize_rebalance/ -o results/pipeline_construction/discretize_rebalance/

python results_processors/graphs_maker.py
