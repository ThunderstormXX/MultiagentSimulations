import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.agents import RandomAgent, RLAgent
from src.simulations.market_simulation import MarketSimulation
from src.utils.visualization import plot_pnl, plot_market_dynamics

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def create_agents(config):
    agents = []
    
    # Create random agents
    for i in range(config['agents']['random_agents']):
        agents.append(RandomAgent(f"random_{i+1}", config['agents']['initial_cash']))
    
    # Create RL agents with different epsilons for diversity
    for i in range(config['agents']['rl_agents']):
        epsilon = 0.05 + (i * 0.02)  # Different exploration rates
        agents.append(RLAgent(f"rl_{i+1}", config['agents']['initial_cash'], epsilon=epsilon))
    
    return agents

def main():
    print("Starting Large-Scale Multi-Agent Experiment 3...")
    
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    config = load_config(config_path)
    
    agents = create_agents(config)
    print(f"Created {len(agents)} agents: {config['agents']['random_agents']} random + {config['agents']['rl_agents']} RL")
    
    # Run simulation with more steps for larger population
    sim = MarketSimulation(agents)
    sim.run(config['simulation']['num_steps'])
    
    results_df = sim.get_results()
    
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)
    
    if config['output']['save_csv']:
        results_df.to_csv(os.path.join(results_dir, 'baseline_simulation.csv'), index=False)
    
    if config['output']['save_plots']:
        # Plot only subset for readability
        sample_agents = [f"random_{i}" for i in range(1, 6)] + [f"rl_{i}" for i in range(1, 6)]
        plot_pnl(results_df, sample_agents, os.path.join(results_dir, 'baseline_pnl_sample.png'))
        plot_market_dynamics(results_df, os.path.join(results_dir, 'baseline_market_dynamics.png'))
    
    # Print summary statistics
    print("Baseline simulation completed!")
    
    random_agents = [agent for agent in agents if 'random' in agent.agent_id]
    rl_agents = [agent for agent in agents if 'rl' in agent.agent_id]
    
    random_pnls = [agent.pnl for agent in random_agents]
    rl_pnls = [agent.pnl for agent in rl_agents]
    
    print(f"\nRandom agents - Avg: {sum(random_pnls)/len(random_pnls):.2f}, "
          f"Min: {min(random_pnls):.2f}, Max: {max(random_pnls):.2f}")
    print(f"RL agents - Avg: {sum(rl_pnls)/len(rl_pnls):.2f}, "
          f"Min: {min(rl_pnls):.2f}, Max: {max(rl_pnls):.2f}")

if __name__ == "__main__":
    main()