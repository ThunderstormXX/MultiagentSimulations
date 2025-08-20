#!/bin/bash

echo "Starting Large-Scale Multi-Agent Experiment 3..."
echo "15 Random Agents vs 15 RL Agents (30 total)"

# Navigate to script directory
cd "$(dirname "$0")"

echo "Phase 1: Running baseline simulation with 30 agents..."
python ../python_script/run_experiment.py

echo "Phase 2: Training enhanced LSTM on multiple simulations..."
python ../python_script/train_lstm.py

echo "Phase 3: Running control simulation with trained RL agents..."
python ../python_script/control_simulation.py

echo ""
echo "Large-scale experiment completed! Results available:"
echo "- large_scale_comparison.png (15 individual RL agent comparisons + summary)"
echo "- population_analysis.png (statistical analysis across agent populations)"
echo "- Detailed performance statistics in terminal output"
echo "- All CSV data for further analysis"