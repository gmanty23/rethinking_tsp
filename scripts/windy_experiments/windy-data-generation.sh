#!/bin/bash

# Create directories if they don't exist
mkdir -p data/windy_tsp

# 1. Validation Data
# Updated --num_samples to 5120 to match 'VAL_SIZE=5120' in Script 1
echo "Generating Validation Data (5,120 samples)..."
python data/windy_tsp/generate_windy_tsp.py \
    --min_nodes 20 \
    --max_nodes 20 \
    --num_samples 5120 \
    --filename data/windy_tsp/windy_tsp20_val.pkl \
    --alpha 5.0 \
    --max_wind 0.5 \
    --seed 1234

# # 2. Training Data
# # Updated --num_samples to 1280000 to match '--epoch_size 1280000' in Script 1
# echo "Generating Training Data (1,280,000 samples)..."
# python data/windy_tsp/generate_windy_tsp.py \
#     --min_nodes 20 \
#     --max_nodes 20 \
#     --num_samples 1280000 \
#     --filename data/windy_tsp/windy_tsp20_train.pkl \
#     --alpha 5.0 \
#     --max_wind 0.5 \
#     --seed 4321

# echo "Data generation complete."