#!/bin/bash

python3 experiments_launcher.py -p features rebalance -r results/features_rebalance/run1/conf1
python3 experiments_launcher.py -p rebalance features -r results/features_rebalance/run1/conf2
python3 experiments_launcher.py -p features rebalance -r results/features_rebalance/run2/conf1
python3 experiments_launcher.py -p rebalance features -r results/features_rebalance/run2/conf2
python3 experiments_launcher.py -p features rebalance -r results/features_rebalance/run3/conf1
python3 experiments_launcher.py -p rebalance features -r results/features_rebalance/run3/conf2
