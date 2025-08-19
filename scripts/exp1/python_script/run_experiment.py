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

def create_agent(agent_config):
    agent_type = agent_config['type']
    agent_id = agent_config['id']
    initial_cash = agent_config['initial_cash']
    
    if agent_type == 'RandomAgent':
        return RandomAgent(agent_id, initial_cash)
    elif agent_type == 'RLAgent':
        return RLAgent(agent_id, initial_cash)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    config = load_config(config_path)
    
    # Create agents
    agents = [create_agent(agent_config) for agent_config in config['agents']]
    
    # Run simulation
    sim = MarketSimulation(agents)
    sim.run(config['simulation']['num_steps'])
    
    # Get results
    results_df = sim.get_results()
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)
    
    if config['output']['save_csv']:
        results_df.to_csv(os.path.join(results_dir, 'simulation_results.csv'), index=False)
    
    if config['output']['save_plots']:
        agent_ids = [agent.agent_id for agent in agents]
        plot_pnl(results_df, agent_ids, os.path.join(results_dir, 'pnl_plot.png'))
        plot_market_dynamics(results_df, os.path.join(results_dir, 'market_dynamics.png'))
    
    # Print final results
    print("Simulation completed!")
    print("\nFinal PnL:")
    for agent in agents:
        print(f"{agent.agent_id}: {agent.pnl:.2f}")

if __name__ == "__main__":
    main()