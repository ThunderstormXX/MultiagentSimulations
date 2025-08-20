import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

def plot_large_scale_comparison(baseline_df: pd.DataFrame, control_df: pd.DataFrame, 
                               num_rl_agents: int = 15, save_path: str = None):
    """Plot comparison for large number of RL agents"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid: 4x4 for first 15 agents + summary plots
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Individual RL agent comparisons (first 15 in 4x4 grid)
    for i in range(min(15, num_rl_agents)):
        row = i // 4
        col = i % 4
        
        ax = fig.add_subplot(gs[row, col])
        
        baseline_col = f'rl_{i+1}_pnl'
        control_col = f'rl_trained_{i+1}_pnl'
        
        if baseline_col in baseline_df.columns and control_col in control_df.columns:
            ax.plot(baseline_df['step'], baseline_df[baseline_col], 
                   label='Before', color='red', alpha=0.6, linewidth=1)
            ax.plot(control_df['step'], control_df[control_col], 
                   label='After', color='green', linewidth=1.5)
            
            # Calculate improvement
            baseline_final = baseline_df[baseline_col].iloc[-1]
            control_final = control_df[control_col].iloc[-1]
            improvement = control_final - baseline_final
            
            ax.set_title(f'RL-{i+1} (Δ{improvement:+.0f})', fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            if i == 0:  # Legend only on first plot
                ax.legend(fontsize=8)
    
    # Summary statistics in bottom row
    # Average performance
    ax_avg = fig.add_subplot(gs[3, 2])
    baseline_rl_cols = [f'rl_{i+1}_pnl' for i in range(num_rl_agents)]
    control_rl_cols = [f'rl_trained_{i+1}_pnl' for i in range(num_rl_agents)]
    
    baseline_avg = baseline_df[[col for col in baseline_rl_cols if col in baseline_df.columns]].mean(axis=1)
    control_avg = control_df[[col for col in control_rl_cols if col in control_df.columns]].mean(axis=1)
    
    ax_avg.plot(baseline_df['step'], baseline_avg, label='Before Training', color='red', linewidth=2)
    ax_avg.plot(control_df['step'], control_avg, label='After Training', color='green', linewidth=2)
    ax_avg.set_title('Average RL Performance', fontsize=12)
    ax_avg.legend()
    ax_avg.grid(True)
    
    # Improvement distribution
    ax_dist = fig.add_subplot(gs[3, 3])
    improvements = []
    for i in range(num_rl_agents):
        baseline_col = f'rl_{i+1}_pnl'
        control_col = f'rl_trained_{i+1}_pnl'
        if baseline_col in baseline_df.columns and control_col in control_df.columns:
            baseline_final = baseline_df[baseline_col].iloc[-1]
            control_final = control_df[control_col].iloc[-1]
            improvements.append(control_final - baseline_final)
    
    ax_dist.hist(improvements, bins=10, alpha=0.7, color='blue', edgecolor='black')
    ax_dist.axvline(np.mean(improvements), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(improvements):.1f}')
    ax_dist.set_title('Improvement Distribution')
    ax_dist.set_xlabel('PnL Improvement')
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    
    plt.suptitle(f'Large-Scale RL Training Results ({num_rl_agents} Agents)', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_agent_population_analysis(baseline_df: pd.DataFrame, control_df: pd.DataFrame,
                                 num_random: int = 15, num_rl: int = 15, save_path: str = None):
    """Analyze performance across large agent populations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Final PnL distributions
    random_cols = [f'random_{i+1}_pnl' for i in range(num_random)]
    baseline_rl_cols = [f'rl_{i+1}_pnl' for i in range(num_rl)]
    control_rl_cols = [f'rl_trained_{i+1}_pnl' for i in range(num_rl)]
    
    random_final = [baseline_df.iloc[-1].get(col, 0) for col in random_cols if col in baseline_df.columns]
    baseline_rl_final = [baseline_df.iloc[-1].get(col, 0) for col in baseline_rl_cols if col in baseline_df.columns]
    control_rl_final = [control_df.iloc[-1].get(col, 0) for col in control_rl_cols if col in control_df.columns]
    
    axes[0,0].boxplot([random_final, baseline_rl_final, control_rl_final], 
                     labels=['Random', 'RL Before', 'RL After'])
    axes[0,0].set_title('Final PnL Distribution by Agent Type')
    axes[0,0].set_ylabel('Final PnL')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Performance over time
    random_avg = baseline_df[[col for col in random_cols if col in baseline_df.columns]].mean(axis=1)
    baseline_rl_avg = baseline_df[[col for col in baseline_rl_cols if col in baseline_df.columns]].mean(axis=1)
    control_rl_avg = control_df[[col for col in control_rl_cols if col in control_df.columns]].mean(axis=1)
    
    axes[0,1].plot(baseline_df['step'], random_avg, label='Random Agents', color='blue', linewidth=2)
    axes[0,1].plot(baseline_df['step'], baseline_rl_avg, label='RL Before', color='red', linewidth=2)
    axes[0,1].plot(control_df['step'], control_rl_avg, label='RL After', color='green', linewidth=2)
    axes[0,1].set_title('Average Performance Over Time')
    axes[0,1].set_xlabel('Step')
    axes[0,1].set_ylabel('Average PnL')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # 3. Volatility comparison
    random_vol = baseline_df[[col for col in random_cols if col in baseline_df.columns]].std(axis=1)
    baseline_rl_vol = baseline_df[[col for col in baseline_rl_cols if col in baseline_df.columns]].std(axis=1)
    control_rl_vol = control_df[[col for col in control_rl_cols if col in control_df.columns]].std(axis=1)
    
    axes[1,0].plot(baseline_df['step'], random_vol, label='Random', color='blue', alpha=0.7)
    axes[1,0].plot(baseline_df['step'], baseline_rl_vol, label='RL Before', color='red', alpha=0.7)
    axes[1,0].plot(control_df['step'], control_rl_vol, label='RL After', color='green', alpha=0.7)
    axes[1,0].set_title('PnL Volatility Across Agent Types')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('PnL Standard Deviation')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # 4. Winner analysis
    winners_random = sum(1 for pnl in random_final if pnl > 0)
    winners_baseline = sum(1 for pnl in baseline_rl_final if pnl > 0)
    winners_control = sum(1 for pnl in control_rl_final if pnl > 0)
    
    axes[1,1].bar(['Random', 'RL Before', 'RL After'], 
                 [winners_random/len(random_final)*100, 
                  winners_baseline/len(baseline_rl_final)*100,
                  winners_control/len(control_rl_final)*100],
                 color=['blue', 'red', 'green'], alpha=0.7)
    axes[1,1].set_title('Percentage of Profitable Agents')
    axes[1,1].set_ylabel('% Profitable')
    axes[1,1].grid(True, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()