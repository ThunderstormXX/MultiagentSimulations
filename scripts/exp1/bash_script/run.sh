#!/bin/bash

echo "Starting Financial Market Simulation Experiment 1..."

# Navigate to script directory
cd "$(dirname "$0")"

echo "Phase 1: Running baseline simulation..."
python ../python_script/run_experiment.py

echo "Phase 2: Training LSTM on multiple simulations..."
python ../python_script/train_lstm.py

echo "Phase 3: Running control simulation with trained model..."
python ../python_script/control_simulation.py

echo "Full experiment completed! Check results/ directory for all outputs."