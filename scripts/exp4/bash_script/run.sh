#!/bin/bash

echo "Starting Competitive Multi-Agent Experiment 4..."
echo "Limited liquidity, execution delays, spread constraints"
echo "15 Random vs 15 RL agents in competitive environment"

# Navigate to script directory
cd "$(dirname "$0")"

echo "Phase 1: Running competitive baseline simulation..."
python ../python_script/run_experiment.py

echo "Phase 2: Training competitive LSTM model..."
python ../python_script/train_lstm.py

echo "Phase 3: Running competitive control simulation..."
python ../python_script/control_simulation.py

echo ""
echo "Competitive experiment completed! Results:"
echo "- competitive_analysis.png (trading activity, spreads, performance)"
echo "- competitive_comparison.png (individual agent comparisons)"
echo "- Detailed competitive statistics in terminal output"
echo "- Market efficiency and trading activity analysis"