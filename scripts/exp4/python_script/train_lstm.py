import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.agents import RandomAgent
from src.simulations.competitive_market import CompetitiveMarketSimulation

class CompetitiveLSTMTrainer:
    def __init__(self, input_size=12, hidden_size=256, output_size=4):
        self.model = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.fc.parameters()), lr=0.0005)
        self.criterion = nn.MSELoss()
        
    def prepare_competitive_data(self, simulation_results):
        """Prepare training data with competitive market features"""
        X, y = [], []
        
        for i in range(20, len(simulation_results)):  # Longer history for competitive environment
            features = []
            for j in range(i-20, i):
                row = simulation_results.iloc[j]
                
                # Basic market data
                bid = row.get('best_bid') or 100.0
                ask = row.get('best_ask') or 101.0
                spread = row.get('spread') or 1.0
                mid = row.get('mid_price', (bid + ask) / 2)
                
                # Competitive features
                total_trades = row.get('total_trades', 0)
                pending_orders = row.get('pending_orders', 0)
                
                # Agent activity
                total_agent_orders = sum([row.get(f'random_{k}_orders', 0) for k in range(1, 11)])
                avg_agent_pnl = np.mean([row.get(f'random_{k}_pnl', 0.0) for k in range(1, 11)])
                
                # Market dynamics
                price_momentum = (mid - 100.0) / 100.0
                trade_intensity = total_trades / max(j, 1)  # Trades per step
                market_stress = spread / mid if mid > 0 else 0
                step_normalized = j / 1000.0
                
                features.append([
                    bid / 100.0, ask / 100.0, spread / 10.0, mid / 100.0,
                    avg_agent_pnl / 1000.0, total_agent_orders / 50.0,
                    price_momentum, trade_intensity / 10.0,
                    market_stress, step_normalized,
                    total_trades / 100.0, pending_orders / 10.0
                ])
            
            # Predict competitive outcomes
            current_price = simulation_results.iloc[i].get('mid_price', 100.0)
            prev_price = simulation_results.iloc[i-1].get('mid_price', 100.0)
            
            if prev_price > 0:
                price_change = (current_price - prev_price) / prev_price
            else:
                price_change = 0.0
                
            price_change = max(-0.03, min(0.03, price_change))  # Tighter bounds for competitive market
            
            # Additional targets for competitive environment
            current_spread = simulation_results.iloc[i].get('spread', 1.0) / 10.0
            trade_opportunity = min(simulation_results.iloc[i].get('total_trades', 0) / 10.0, 1.0)
            market_efficiency = 1.0 / (1.0 + current_spread)  # Higher when spread is tight
            
            X.append(features)
            y.append([price_change, current_price / 100.0, current_spread, market_efficiency])
            
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def train(self, X, y, epochs=50):
        if torch.isnan(X).any() or torch.isnan(y).any():
            X = torch.nan_to_num(X, 0.0)
            y = torch.nan_to_num(y, 0.0)
            
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize weights
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                lstm_out, _ = self.model(batch_X)
                predictions = self.fc(lstm_out[:, -1, :])
                
                loss = self.criterion(predictions, batch_y)
                
                if torch.isnan(loss):
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.fc.parameters()), 0.3)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def save_model(self, path):
        torch.save({
            'lstm_state': self.model.state_dict(),
            'fc_state': self.fc.state_dict(),
            'input_size': 12,
            'hidden_size': 256,
            'output_size': 4
        }, path)

def run_competitive_training_simulations(num_simulations=20):
    """Run competitive training simulations"""
    all_results = []
    
    for i in range(num_simulations):
        print(f"Running competitive training simulation {i+1}/{num_simulations}")
        
        # Create agents for training
        agents = [RandomAgent(f"random_{j}", 10000) for j in range(1, 11)]
        
        # Competitive simulation with varying parameters
        liquidity_factor = 0.3 + (i % 3) * 0.1  # Vary liquidity
        max_spread = 2.0 + (i % 4) * 0.5  # Vary spread constraints
        
        sim = CompetitiveMarketSimulation(
            agents, 
            max_spread=max_spread,
            liquidity_factor=liquidity_factor,
            execution_delay=1
        )
        
        sim.run(1000)
        
        results_df = sim.get_results()
        all_results.append(results_df)
    
    return all_results

def main():
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Starting competitive LSTM training...")
    
    training_data = run_competitive_training_simulations(config['training']['num_simulations'])
    
    trainer = CompetitiveLSTMTrainer()
    all_X, all_y = [], []
    
    for sim_data in training_data:
        if len(sim_data) > 20:
            X, y = trainer.prepare_competitive_data(sim_data)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
    
    if not all_X:
        print("No valid training data generated")
        return
        
    combined_X = torch.cat(all_X, dim=0)
    combined_y = torch.cat(all_y, dim=0)
    
    print(f"Training competitive model on {len(combined_X)} samples...")
    
    trainer.train(combined_X, combined_y, config['training']['epochs'])
    
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)
    trainer.save_model(os.path.join(results_dir, 'competitive_lstm.pth'))
    
    print("Competitive LSTM training completed!")

if __name__ == "__main__":
    main()