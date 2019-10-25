#!/bin/bash

python3 experiments_launcher.py -p features rebalance -r results/features_rebalance/conf1
python3 experiments_launcher.py -p rebalance features -r results/features_rebalance/conf2
python3 results_collector.py -p features rebalance -i results/features_rebalance/conf1 -ii results/features_rebalance/conf2 -o results/features_rebalance