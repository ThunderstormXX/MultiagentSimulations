import sys
import os
import json
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.agents import RandomAgent, RLAgent
from src.simulations.market_simulation import MarketSimulation
from src.utils.visualization import plot_pnl, plot_market_dynamics
from src.utils.large_scale_visualization import plot_large_scale_comparison, plot_agent_population_analysis

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def create_control_agents(config, model_path=None):
    agents = []
    
    # Create random agents (same as baseline)
    for i in range(config['agents']['random_agents']):
        agents.append(RandomAgent(f"random_{i+1}", config['agents']['initial_cash']))
    
    # Create trained RL agents with diversity
    for i in range(config['agents']['rl_agents']):
        epsilon = 0.02 + (i * 0.01)  # Different exploration for diversity
        agents.append(RLAgent(f"rl_trained_{i+1}", config['agents']['initial_cash'], 
                             model_path=model_path if os.path.exists(model_path or '') else None,
                             epsilon=epsilon))
    
    return agents

def analyze_large_scale_performance(control_results, baseline_results, config):
    """Detailed analysis for large-scale experiment"""
    print("\n=== LARGE-SCALE PERFORMANCE ANALYSIS ===")
    
    num_rl = config['agents']['rl_agents']
    num_random = config['agents']['random_agents']
    
    # Collect all improvements
    improvements = []
    baseline_finals = []
    control_finals = []
    
    for i in range(1, num_rl + 1):
        baseline_col = f'rl_{i}_pnl'
        control_col = f'rl_trained_{i}_pnl'
        
        if baseline_col in baseline_results.columns and control_col in control_results.columns:
            baseline_final = baseline_results.iloc[-1][baseline_col]
            control_final = control_results.iloc[-1][control_col]
            improvement = control_final - baseline_final
            
            improvements.append(improvement)
            baseline_finals.append(baseline_final)
            control_finals.append(control_final)
    
    # Statistics
    if improvements:
        print(f"Total RL agents analyzed: {len(improvements)}")
        print(f"Average improvement: {np.mean(improvements):+.2f}")
        print(f"Median improvement: {np.median(improvements):+.2f}")
        print(f"Best improvement: {max(improvements):+.2f}")
        print(f"Worst improvement: {min(improvements):+.2f}")
        print(f"Std deviation: {np.std(improvements):.2f}")
        
        # Success rate
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        success_rate = positive_improvements / len(improvements) * 100
        print(f"Success rate: {success_rate:.1f}% ({positive_improvements}/{len(improvements)} agents improved)")
        
        # Compare with random agents
        random_cols = [f'random_{i+1}_pnl' for i in range(num_random)]
        random_finals = [baseline_results.iloc[-1].get(col, 0) for col in random_cols if col in baseline_results.columns]
        
        if random_finals:
            random_avg = np.mean(random_finals)
            baseline_rl_avg = np.mean(baseline_finals)
            control_rl_avg = np.mean(control_finals)
            
            print(f"\nComparison with Random agents:")
            print(f"Random agents average: {random_avg:.2f}")
            print(f"RL before training: {baseline_rl_avg:.2f}")
            print(f"RL after training: {control_rl_avg:.2f}")
            print(f"RL improvement vs Random: {control_rl_avg - random_avg:+.2f}")

def main():
    print("Starting Large-Scale Control Simulation (30 agents)...")
    
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    config = load_config(config_path)
    
    model_path = os.path.join(os.path.dirname(__file__), '../results/trained_lstm.pth')
    
    agents = create_control_agents(config, model_path)
    print(f"Created {len(agents)} agents: {config['agents']['random_agents']} random + {config['agents']['rl_agents']} trained RL")
    
    # Run control simulation
    sim = MarketSimulation(agents)
    sim.run(config['simulation']['num_steps'])
    
    control_results = sim.get_results()
    
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    control_results.to_csv(os.path.join(results_dir, 'control_simulation.csv'), index=False)
    
    # Load baseline for comparison
    baseline_path = os.path.join(results_dir, 'baseline_simulation.csv')
    if os.path.exists(baseline_path):
        baseline_results = pd.read_csv(baseline_path)
        analyze_large_scale_performance(control_results, baseline_results, config)
        
        # Generate large-scale visualizations
        plot_large_scale_comparison(baseline_results, control_results, 
                                   config['agents']['rl_agents'],
                                   os.path.join(results_dir, 'large_scale_comparison.png'))
        
        plot_agent_population_analysis(baseline_results, control_results,
                                     config['agents']['random_agents'], config['agents']['rl_agents'],
                                     os.path.join(results_dir, 'population_analysis.png'))
    
    # Generate sample plots (subset for readability)
    sample_agents = [f"random_{i}" for i in range(1, 4)] + [f"rl_trained_{i}" for i in range(1, 4)]
    plot_pnl(control_results, sample_agents, os.path.join(results_dir, 'control_sample_pnl.png'))
    plot_market_dynamics(control_results, os.path.join(results_dir, 'control_market_dynamics.png'))
    
    # Final summary
    print("\nLarge-scale control simulation completed!")
    
    random_agents = [agent for agent in agents if 'random' in agent.agent_id]
    rl_agents = [agent for agent in agents if 'rl_trained' in agent.agent_id]
    
    if random_agents and rl_agents:
        random_avg = sum(agent.pnl for agent in random_agents) / len(random_agents)
        rl_avg = sum(agent.pnl for agent in rl_agents) / len(rl_agents)
        
        print(f"Final averages - Random: {random_avg:.2f}, Trained RL: {rl_avg:.2f}")
        print(f"Overall improvement: {rl_avg - random_avg:+.2f}")

if __name__ == "__main__":
    main()