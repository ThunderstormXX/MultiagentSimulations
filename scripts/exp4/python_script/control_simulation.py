import sys
import os
import json
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.agents import RandomAgent, RLAgent
from src.simulations.competitive_market import CompetitiveMarketSimulation
from src.utils.visualization import plot_pnl, plot_market_dynamics
from src.utils.large_scale_visualization import plot_large_scale_comparison, plot_agent_population_analysis

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def create_competitive_control_agents(config, model_path=None):
    agents = []
    
    # Create random agents
    for i in range(config['agents']['random_agents']):
        agents.append(RandomAgent(f"random_{i+1}", config['agents']['initial_cash']))
    
    # Create competitive trained RL agents
    for i in range(config['agents']['rl_agents']):
        epsilon = 0.01 + (i * 0.005)  # Very low epsilon for competitive environment
        agents.append(RLAgent(f"rl_trained_{i+1}", config['agents']['initial_cash'], 
                             model_path=model_path if os.path.exists(model_path or '') else None,
                             epsilon=epsilon))
    
    return agents

def analyze_competitive_performance(control_results, baseline_results, config):
    """Analyze performance in competitive environment"""
    print("\n=== COMPETITIVE MARKET ANALYSIS ===")
    
    num_rl = config['agents']['rl_agents']
    
    # Trading activity analysis
    baseline_trades = baseline_results['total_trades'].iloc[-1] if 'total_trades' in baseline_results.columns else 0
    control_trades = control_results['total_trades'].iloc[-1] if 'total_trades' in control_results.columns else 0
    
    print(f"Total trades - Baseline: {baseline_trades}, Control: {control_trades}")
    print(f"Trade increase: {control_trades - baseline_trades:+}")
    
    # Performance improvements
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
    
    if improvements:
        print(f"\nCompetitive Performance:")
        print(f"Average improvement: {np.mean(improvements):+.2f}")
        print(f"Median improvement: {np.median(improvements):+.2f}")
        print(f"Best performer: {max(improvements):+.2f}")
        print(f"Worst performer: {min(improvements):+.2f}")
        
        # Competition winners
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        success_rate = positive_improvements / len(improvements) * 100
        print(f"Competitive success rate: {success_rate:.1f}% ({positive_improvements}/{len(improvements)})")
        
        # Market efficiency analysis
        baseline_spreads = baseline_results['spread'].mean()
        control_spreads = control_results['spread'].mean()
        print(f"\nMarket efficiency:")
        print(f"Average spread - Baseline: {baseline_spreads:.3f}, Control: {control_spreads:.3f}")
        print(f"Spread improvement: {baseline_spreads - control_spreads:+.3f}")

def plot_competitive_analysis(baseline_df, control_df, save_path=None):
    """Plot competitive market analysis"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Trading activity
    axes[0,0].plot(baseline_df['step'], baseline_df.get('total_trades', 0), label='Baseline', color='red')
    axes[0,0].plot(control_df['step'], control_df.get('total_trades', 0), label='Trained', color='green')
    axes[0,0].set_title('Trading Activity Over Time')
    axes[0,0].set_ylabel('Cumulative Trades')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Spread dynamics
    axes[0,1].plot(baseline_df['step'], baseline_df.get('spread', 1), label='Baseline', color='red', alpha=0.7)
    axes[0,1].plot(control_df['step'], control_df.get('spread', 1), label='Trained', color='green', alpha=0.7)
    axes[0,1].set_title('Bid-Ask Spread Evolution')
    axes[0,1].set_ylabel('Spread')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Performance comparison
    rl_cols_baseline = [col for col in baseline_df.columns if col.startswith('rl_') and col.endswith('_pnl')]
    rl_cols_control = [col for col in control_df.columns if col.startswith('rl_trained_') and col.endswith('_pnl')]
    
    if rl_cols_baseline and rl_cols_control:
        baseline_avg = baseline_df[rl_cols_baseline].mean(axis=1)
        control_avg = control_df[rl_cols_control].mean(axis=1)
        
        axes[1,0].plot(baseline_df['step'], baseline_avg, label='Before Training', color='red', linewidth=2)
        axes[1,0].plot(control_df['step'], control_avg, label='After Training', color='green', linewidth=2)
        axes[1,0].set_title('Average RL Performance in Competition')
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Average PnL')
        axes[1,0].legend()
        axes[1,0].grid(True)
    
    # Final performance distribution
    if rl_cols_baseline and rl_cols_control:
        baseline_finals = [baseline_df[col].iloc[-1] for col in rl_cols_baseline]
        control_finals = [control_df[col].iloc[-1] for col in rl_cols_control]
        
        axes[1,1].hist(baseline_finals, alpha=0.7, label='Before Training', color='red', bins=10)
        axes[1,1].hist(control_finals, alpha=0.7, label='After Training', color='green', bins=10)
        axes[1,1].set_title('Final PnL Distribution')
        axes[1,1].set_xlabel('Final PnL')
        axes[1,1].set_ylabel('Count')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Starting Competitive Control Simulation...")
    
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    config = load_config(config_path)
    
    model_path = os.path.join(os.path.dirname(__file__), '../results/competitive_lstm.pth')
    
    agents = create_competitive_control_agents(config, model_path)
    print(f"Created {len(agents)} agents for competitive environment")
    
    # Run competitive control simulation
    sim = CompetitiveMarketSimulation(
        agents,
        max_spread=config['simulation']['max_spread'],
        liquidity_factor=config['simulation']['liquidity_factor'],
        execution_delay=config['simulation']['execution_delay']
    )
    
    sim.run(config['simulation']['num_steps'])
    
    control_results = sim.get_results()
    competition_stats = sim.get_competition_stats()
    
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    control_results.to_csv(os.path.join(results_dir, 'competitive_control.csv'), index=False)
    
    # Load baseline for comparison
    baseline_path = os.path.join(results_dir, 'competitive_baseline.csv')
    if os.path.exists(baseline_path):
        baseline_results = pd.read_csv(baseline_path)
        analyze_competitive_performance(control_results, baseline_results, config)
        
        # Generate competitive visualizations
        plot_competitive_analysis(baseline_results, control_results,
                                os.path.join(results_dir, 'competitive_analysis.png'))
        
        plot_large_scale_comparison(baseline_results, control_results, 
                                   config['agents']['rl_agents'],
                                   os.path.join(results_dir, 'competitive_comparison.png'))
    
    # Competition statistics
    print(f"\nCompetitive Market Statistics:")
    print(f"Total trades: {competition_stats['total_trades']}")
    print(f"Average trades per agent: {competition_stats['avg_trades_per_agent']:.1f}")
    if competition_stats['most_active_agent']:
        print(f"Most active: {competition_stats['most_active_agent'][0]} ({competition_stats['most_active_agent'][1]} trades)")
    
    # Final summary
    random_agents = [agent for agent in agents if 'random' in agent.agent_id]
    rl_agents = [agent for agent in agents if 'rl_trained' in agent.agent_id]
    
    if random_agents and rl_agents:
        random_avg = sum(agent.pnl for agent in random_agents) / len(random_agents)
        rl_avg = sum(agent.pnl for agent in rl_agents) / len(rl_agents)
        
        print(f"\nFinal competitive results:")
        print(f"Random agents average: {random_avg:.2f}")
        print(f"Trained RL agents average: {rl_avg:.2f}")
        print(f"Competitive advantage: {rl_avg - random_avg:+.2f}")

if __name__ == "__main__":
    main()