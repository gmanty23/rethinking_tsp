#!/bin/bash

# Create directories if they don't exist
mkdir -p data/windy_tsp

echo "Generating Validation Data (Fixed for all experiments)..."
python data/windy_tsp/generate_windy_tsp.py \
    --min_nodes 20 \
    --max_nodes 20 \
    --num_samples 10000 \
    --filename data/windy_tsp/windy_tsp20_val.pkl \
    --alpha 5.0 \
    --max_wind 0.5 \
    --seed 1234

echo "Generating Training Data (Fixed environment for reproducibility)..."
python data/windy_tsp/generate_windy_tsp.py \
    --min_nodes 20 \
    --max_nodes 20 \
    --num_samples 128000 \
    --filename data/windy_tsp/windy_tsp20_train.pkl \
    --alpha 5.0 \
    --max_wind 0.5 \
    --seed 4321

echo "Data generation complete."