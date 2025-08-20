import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.agents import RandomAgent, RLAgent
from src.simulations.competitive_market import CompetitiveMarketSimulation
from src.utils.visualization import plot_pnl, plot_market_dynamics

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def create_competitive_agents(config):
    agents = []
    
    # Create random agents with faster reaction times
    for i in range(config['agents']['random_agents']):
        agents.append(RandomAgent(f"random_{i+1}", config['agents']['initial_cash']))
    
    # Create RL agents with competitive parameters
    for i in range(config['agents']['rl_agents']):
        epsilon = 0.03 + (i * 0.015)  # Lower epsilon for more exploitation
        agents.append(RLAgent(f"rl_{i+1}", config['agents']['initial_cash'], epsilon=epsilon))
    
    return agents

def main():
    print("Starting Competitive Multi-Agent Experiment 4...")
    print("Limited liquidity, execution delays, spread constraints")
    
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    config = load_config(config_path)
    
    agents = create_competitive_agents(config)
    print(f"Created {len(agents)} competitive agents")
    
    # Run competitive simulation
    sim = CompetitiveMarketSimulation(
        agents,
        max_spread=config['simulation']['max_spread'],
        liquidity_factor=config['simulation']['liquidity_factor'],
        execution_delay=config['simulation']['execution_delay']
    )
    
    sim.run(config['simulation']['num_steps'])
    
    results_df = sim.get_results()
    competition_stats = sim.get_competition_stats()
    
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)
    
    if config['output']['save_csv']:
        results_df.to_csv(os.path.join(results_dir, 'competitive_baseline.csv'), index=False)
    
    if config['output']['save_plots']:
        sample_agents = [f"random_{i}" for i in range(1, 4)] + [f"rl_{i}" for i in range(1, 4)]
        plot_pnl(results_df, sample_agents, os.path.join(results_dir, 'competitive_baseline_pnl.png'))
        plot_market_dynamics(results_df, os.path.join(results_dir, 'competitive_market_dynamics.png'))
    
    # Print competitive statistics
    print("Competitive baseline simulation completed!")
    print(f"\nCompetition Statistics:")
    print(f"Total trades executed: {competition_stats['total_trades']}")
    print(f"Average trades per agent: {competition_stats['avg_trades_per_agent']:.1f}")
    if competition_stats['most_active_agent']:
        print(f"Most active agent: {competition_stats['most_active_agent'][0]} ({competition_stats['most_active_agent'][1]} trades)")
    
    # Performance by type
    random_agents = [agent for agent in agents if 'random' in agent.agent_id]
    rl_agents = [agent for agent in agents if 'rl' in agent.agent_id]
    
    random_pnls = [agent.pnl for agent in random_agents]
    rl_pnls = [agent.pnl for agent in rl_agents]
    
    print(f"\nPerformance in competitive environment:")
    print(f"Random agents - Avg: {sum(random_pnls)/len(random_pnls):.2f}, Range: [{min(random_pnls):.2f}, {max(random_pnls):.2f}]")
    print(f"RL agents - Avg: {sum(rl_pnls)/len(rl_pnls):.2f}, Range: [{min(rl_pnls):.2f}, {max(rl_pnls):.2f}]")

if __name__ == "__main__":
    main()