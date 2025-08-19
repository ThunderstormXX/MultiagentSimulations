#!/bin/bash

echo "Starting Multi-Agent Financial Market Simulation Experiment 2..."
echo "5 Random Agents vs 5 RL Agents"

# Navigate to script directory
cd "$(dirname "$0")"

echo "Phase 1: Running baseline simulation with 10 agents..."
python ../python_script/run_experiment.py

echo "Phase 2: Training LSTM on multiple simulations..."
python ../python_script/train_lstm.py

echo "Phase 3: Running control simulation with trained RL agents..."
python ../python_script/control_simulation.py

echo "Multi-agent experiment completed! Check results/ directory for:"
echo "- multi_agent_comparison.png (individual RL agent comparisons)"
echo "- agent_type_comparison.png (random vs RL performance)"
echo "- All CSV data and individual plots"