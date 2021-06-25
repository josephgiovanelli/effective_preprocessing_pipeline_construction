#!/bin/bash

python results_processors/experiments_summarizer_10x4cv.py -p features rebalance -i results/pipeline_construction/features_rebalance/ -o results/pipeline_construction/features_rebalance/
python results_processors/experiments_summarizer_10x4cv.py -p discretize features -i results/pipeline_construction/discretize_features/ -o results/pipeline_construction/discretize_features/
python results_processors/experiments_summarizer_10x4cv.py -p features normalize -i results/pipeline_construction/features_normalize/ -o results/pipeline_construction/features_normalize/
python results_processors/experiments_summarizer_10x4cv.py -p discretize rebalance -i results/pipeline_construction/discretize_rebalance/ -o results/pipeline_construction/discretize_rebalance/

