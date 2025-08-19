import sys
import os
import json
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.agents import RandomAgent, RLAgent
from src.simulations.market_simulation import MarketSimulation
from src.utils.visualization import plot_pnl, plot_market_dynamics, plot_comparison

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def create_agent(agent_config, model_path=None):
    agent_type = agent_config['type']
    agent_id = agent_config['id']
    initial_cash = agent_config['initial_cash']
    
    if agent_type == 'RandomAgent':
        return RandomAgent(agent_id, initial_cash)
    elif agent_type == 'RLAgent':
        return RLAgent(agent_id, initial_cash, model_path=model_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def compare_with_baseline(control_results, baseline_results):
    """Compare control simulation with baseline simulations"""
    print("\n=== PERFORMANCE COMPARISON ===")
    
    # Final PnL comparison
    control_final = control_results.iloc[-1]
    baseline_final = baseline_results.iloc[-1]
    
    print(f"Control RL Agent PnL: {control_final.get('rl_trained_pnl', 0):.2f}")
    print(f"Baseline RL Agent PnL: {baseline_final.get('rl_1_pnl', 0):.2f}")
    
    # Calculate improvement
    improvement = control_final.get('rl_trained_pnl', 0) - baseline_final.get('rl_1_pnl', 0)
    print(f"Improvement: {improvement:.2f}")
    
    # Volatility comparison
    control_vol = control_results['rl_trained_pnl'].std()
    baseline_vol = baseline_results['rl_1_pnl'].std()
    print(f"Control volatility: {control_vol:.2f}")
    print(f"Baseline volatility: {baseline_vol:.2f}")

def main():
    print("Starting control simulation with trained LSTM...")
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    config = load_config(config_path)
    
    # Path to trained model
    model_path = os.path.join(os.path.dirname(__file__), '../results/trained_lstm.pth')
    
    # Create agents - one with trained model
    agents = [
        RandomAgent("random_baseline", 10000),
        RLAgent("rl_trained", 10000, model_path=model_path if os.path.exists(model_path) else None),
        RandomAgent("random_control", 10000)
    ]
    
    # Run control simulation
    sim = MarketSimulation(agents)
    sim.run(config['simulation']['num_steps'])
    
    # Get results
    control_results = sim.get_results()
    
    # Save control results
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    control_results.to_csv(os.path.join(results_dir, 'control_simulation.csv'), index=False)
    
    # Load baseline results for comparison
    baseline_path = os.path.join(results_dir, 'simulation_results.csv')
    if os.path.exists(baseline_path):
        baseline_results = pd.read_csv(baseline_path)
        compare_with_baseline(control_results, baseline_results)
        
        # Generate comparison visualization
        plot_comparison(baseline_results, control_results, 
                       os.path.join(results_dir, 'training_comparison.png'))
    
    # Generate individual plots
    agent_ids = [agent.agent_id for agent in agents]
    plot_pnl(control_results, agent_ids, os.path.join(results_dir, 'control_pnl_plot.png'))
    plot_market_dynamics(control_results, os.path.join(results_dir, 'control_market_dynamics.png'))
    
    # Print final results
    print("\nControl simulation completed!")
    print("\nFinal PnL:")
    for agent in agents:
        print(f"{agent.agent_id}: {agent.pnl:.2f}")

if __name__ == "__main__":
    main()