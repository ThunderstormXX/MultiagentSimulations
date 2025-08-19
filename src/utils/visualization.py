import matplotlib.pyplot as plt
import pandas as pd
from typing import List

def plot_pnl(df: pd.DataFrame, agent_ids: List[str], save_path: str = None):
    plt.figure(figsize=(10, 6))
    for agent_id in agent_ids:
        plt.plot(df['step'], df[f'{agent_id}_pnl'], label=f'{agent_id} PnL')
    plt.xlabel('Step')
    plt.ylabel('PnL')
    plt.title('Agent PnL Over Time')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_comparison(baseline_df: pd.DataFrame, control_df: pd.DataFrame, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PnL comparison
    ax1.plot(baseline_df['step'], baseline_df.get('rl_1_pnl', [0]*len(baseline_df)), 
             label='RL Agent (Before Training)', color='red', alpha=0.7)
    ax1.plot(control_df['step'], control_df.get('rl_trained_pnl', [0]*len(control_df)), 
             label='RL Agent (After Training)', color='green', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('PnL')
    ax1.set_title('RL Agent Performance: Before vs After Training')
    ax1.legend()
    ax1.grid(True)
    
    # Final PnL bar chart
    baseline_final = baseline_df.iloc[-1].get('rl_1_pnl', 0) if len(baseline_df) > 0 else 0
    control_final = control_df.iloc[-1].get('rl_trained_pnl', 0) if len(control_df) > 0 else 0
    
    ax2.bar(['Before Training', 'After Training'], [baseline_final, control_final], 
            color=['red', 'green'], alpha=0.7)
    ax2.set_ylabel('Final PnL')
    ax2.set_title('Final PnL Comparison')
    ax2.grid(True, axis='y')
    
    # Add improvement text
    improvement = control_final - baseline_final
    ax2.text(0.5, max(baseline_final, control_final) * 0.8, 
             f'Improvement: {improvement:.2f}', ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_market_dynamics(df: pd.DataFrame, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Price dynamics
    ax1.plot(df['step'], df['best_bid'], label='Best Bid', alpha=0.7)
    ax1.plot(df['step'], df['best_ask'], label='Best Ask', alpha=0.7)
    ax1.plot(df['step'], df['mid_price'], label='Mid Price', linewidth=2)
    ax1.set_ylabel('Price')
    ax1.set_title('Market Price Dynamics')
    ax1.legend()
    ax1.grid(True)
    
    # Spread
    ax2.plot(df['step'], df['spread'], color='red')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Spread')
    ax2.set_title('Bid-Ask Spread')
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()