import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

def plot_multi_agent_comparison(baseline_df: pd.DataFrame, control_df: pd.DataFrame, 
                               num_rl_agents: int = 5, save_path: str = None):
    """Plot comparison for multiple RL agents before and after training"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Individual RL agent comparisons (2x3 grid for 5 agents + summary)
    for i in range(num_rl_agents):
        row = i // 3
        col = i % 3
        
        baseline_col = f'rl_{i+1}_pnl'
        control_col = f'rl_trained_{i+1}_pnl'
        
        if baseline_col in baseline_df.columns and control_col in control_df.columns:
            axes[row, col].plot(baseline_df['step'], baseline_df[baseline_col], 
                              label=f'Before Training', color='red', alpha=0.7)
            axes[row, col].plot(control_df['step'], control_df[control_col], 
                              label=f'After Training', color='green', linewidth=2)
            
            axes[row, col].set_title(f'RL Agent {i+1}')
            axes[row, col].set_xlabel('Step')
            axes[row, col].set_ylabel('PnL')
            axes[row, col].legend()
            axes[row, col].grid(True)
    
    # Summary comparison in the last subplot
    axes[1, 2].clear()
    
    # Calculate average performance
    baseline_rl_cols = [f'rl_{i+1}_pnl' for i in range(num_rl_agents)]
    control_rl_cols = [f'rl_trained_{i+1}_pnl' for i in range(num_rl_agents)]
    
    baseline_avg = baseline_df[[col for col in baseline_rl_cols if col in baseline_df.columns]].mean(axis=1)
    control_avg = control_df[[col for col in control_rl_cols if col in control_df.columns]].mean(axis=1)
    
    axes[1, 2].plot(baseline_df['step'], baseline_avg, 
                   label='Average Before Training', color='red', linewidth=3)
    axes[1, 2].plot(control_df['step'], control_avg, 
                   label='Average After Training', color='green', linewidth=3)
    
    axes[1, 2].set_title('Average RL Performance')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Average PnL')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_agent_type_comparison(baseline_df: pd.DataFrame, control_df: pd.DataFrame, 
                             num_random: int = 5, num_rl: int = 5, save_path: str = None):
    """Compare performance between random and RL agents"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Final PnL comparison
    random_cols = [f'random_{i+1}_pnl' for i in range(num_random)]
    baseline_rl_cols = [f'rl_{i+1}_pnl' for i in range(num_rl)]
    control_rl_cols = [f'rl_trained_{i+1}_pnl' for i in range(num_rl)]
    
    # Get final values
    random_final = [baseline_df.iloc[-1].get(col, 0) for col in random_cols if col in baseline_df.columns]
    baseline_rl_final = [baseline_df.iloc[-1].get(col, 0) for col in baseline_rl_cols if col in baseline_df.columns]
    control_rl_final = [control_df.iloc[-1].get(col, 0) for col in control_rl_cols if col in control_df.columns]
    
    # Box plot comparison
    data_to_plot = [random_final, baseline_rl_final, control_rl_final]
    labels = ['Random Agents', 'RL Before Training', 'RL After Training']
    
    ax1.boxplot(data_to_plot, labels=labels)
    ax1.set_ylabel('Final PnL')
    ax1.set_title('Final PnL Distribution by Agent Type')
    ax1.grid(True, axis='y')
    
    # Average performance over time
    random_avg = baseline_df[[col for col in random_cols if col in baseline_df.columns]].mean(axis=1)
    baseline_rl_avg = baseline_df[[col for col in baseline_rl_cols if col in baseline_df.columns]].mean(axis=1)
    control_rl_avg = control_df[[col for col in control_rl_cols if col in control_df.columns]].mean(axis=1)
    
    ax2.plot(baseline_df['step'], random_avg, label='Random Agents', color='blue', linewidth=2)
    ax2.plot(baseline_df['step'], baseline_rl_avg, label='RL Before Training', color='red', linewidth=2)
    ax2.plot(control_df['step'], control_rl_avg, label='RL After Training', color='green', linewidth=2)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Average PnL')
    ax2.set_title('Average Performance Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()