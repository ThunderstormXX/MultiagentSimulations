import sys
import os
import json
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.agents import RandomAgent, RLAgent
from src.simulations.market_simulation import MarketSimulation
from src.utils.visualization import plot_pnl, plot_market_dynamics
from src.utils.multi_agent_visualization import plot_multi_agent_comparison, plot_agent_type_comparison

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def create_control_agents(config, model_path=None):
    agents = []
    
    # Create random agents (same as baseline)
    for i in range(config['agents']['random_agents']):
        agents.append(RandomAgent(f"random_{i+1}", config['agents']['initial_cash']))
    
    # Create trained RL agents
    for i in range(config['agents']['rl_agents']):
        agents.append(RLAgent(f"rl_trained_{i+1}", config['agents']['initial_cash'], 
                             model_path=model_path if os.path.exists(model_path or '') else None))
    
    return agents

def compare_multi_agent_performance(control_results, baseline_results, config):
    """Compare performance across multiple RL agents"""
    print("\n=== MULTI-AGENT PERFORMANCE COMPARISON ===")
    
    num_rl = config['agents']['rl_agents']
    
    # Compare each RL agent
    improvements = []
    for i in range(1, num_rl + 1):
        baseline_col = f'rl_{i}_pnl'
        control_col = f'rl_trained_{i}_pnl'
        
        if baseline_col in baseline_results.columns and control_col in control_results.columns:
            baseline_final = baseline_results.iloc[-1][baseline_col]
            control_final = control_results.iloc[-1][control_col]
            improvement = control_final - baseline_final
            improvements.append(improvement)
            
            print(f"RL Agent {i}: {baseline_final:.2f} → {control_final:.2f} (Δ{improvement:+.2f})")
    
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"\nAverage improvement: {avg_improvement:+.2f}")
        print(f"Best improvement: {max(improvements):+.2f}")
        print(f"Worst improvement: {min(improvements):+.2f}")

def main():
    print("Starting Multi-Agent Control Simulation...")
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    config = load_config(config_path)
    
    # Path to trained model
    model_path = os.path.join(os.path.dirname(__file__), '../results/trained_lstm.pth')
    
    # Create agents with trained models
    agents = create_control_agents(config, model_path)
    print(f"Created {len(agents)} agents with trained RL models")
    
    # Run control simulation
    sim = MarketSimulation(agents)
    sim.run(config['simulation']['num_steps'])
    
    # Get results
    control_results = sim.get_results()
    
    # Save control results
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    control_results.to_csv(os.path.join(results_dir, 'control_simulation.csv'), index=False)
    
    # Load baseline results for comparison
    baseline_path = os.path.join(results_dir, 'baseline_simulation.csv')
    if os.path.exists(baseline_path):
        baseline_results = pd.read_csv(baseline_path)
        compare_multi_agent_performance(control_results, baseline_results, config)
        
        # Generate multi-agent comparison visualizations
        plot_multi_agent_comparison(baseline_results, control_results, 
                                   config['agents']['rl_agents'],
                                   os.path.join(results_dir, 'multi_agent_comparison.png'))
        
        plot_agent_type_comparison(baseline_results, control_results,
                                 config['agents']['random_agents'], config['agents']['rl_agents'],
                                 os.path.join(results_dir, 'agent_type_comparison.png'))
    
    # Generate individual plots
    agent_ids = [agent.agent_id for agent in agents]
    plot_pnl(control_results, agent_ids, os.path.join(results_dir, 'control_pnl_plot.png'))
    plot_market_dynamics(control_results, os.path.join(results_dir, 'control_market_dynamics.png'))
    
    # Print final results by type
    print("\nControl simulation completed!")
    
    random_agents = [agent for agent in agents if 'random' in agent.agent_id]
    rl_agents = [agent for agent in agents if 'rl_trained' in agent.agent_id]
    
    print(f"\nRandom agents PnL: {[f'{agent.pnl:.2f}' for agent in random_agents]}")
    print(f"Trained RL agents PnL: {[f'{agent.pnl:.2f}' for agent in rl_agents]}")
    
    if random_agents and rl_agents:
        random_avg = sum(agent.pnl for agent in random_agents) / len(random_agents)
        rl_avg = sum(agent.pnl for agent in rl_agents) / len(rl_agents)
        print(f"\nAverage PnL - Random: {random_avg:.2f}, Trained RL: {rl_avg:.2f}")

if __name__ == "__main__":
    main()