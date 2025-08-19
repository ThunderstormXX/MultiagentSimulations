import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.agents import RandomAgent
from src.simulations.market_simulation import MarketSimulation

class LSTMTrainer:
    def __init__(self, input_size=6, hidden_size=64, output_size=3):
        self.model = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.fc.parameters()))
        self.criterion = nn.MSELoss()
        
    def prepare_data(self, simulation_results):
        """Convert simulation data to LSTM training format"""
        X, y = [], []
        for i in range(10, len(simulation_results)):
            # Use last 10 steps as input
            features = []
            for j in range(i-10, i):
                row = simulation_results.iloc[j]
                bid = row.get('best_bid') or 100.0
                ask = row.get('best_ask') or 101.0
                spread = row.get('spread') or 1.0
                mid = row.get('mid_price', (bid + ask) / 2)
                pnl1 = row.get('random_1_pnl', 0.0)
                pnl2 = row.get('random_2_pnl', 0.0)
                
                # Normalize features
                features.append([
                    bid / 100.0, ask / 100.0, spread / 10.0,
                    mid / 100.0, pnl1 / 1000.0, pnl2 / 1000.0
                ])
            
            # Predict next price movement
            current_price = simulation_results.iloc[i].get('mid_price', 100.0)
            prev_price = simulation_results.iloc[i-1].get('mid_price', 100.0)
            
            if prev_price > 0:
                price_change = (current_price - prev_price) / prev_price
            else:
                price_change = 0.0
                
            # Clamp extreme values
            price_change = max(-0.1, min(0.1, price_change))
            
            X.append(features)
            y.append([price_change, current_price / 100.0, (simulation_results.iloc[i].get('spread', 1.0)) / 10.0])
            
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def train(self, X, y, epochs=50):
        # Check for NaN
        if torch.isnan(X).any() or torch.isnan(y).any():
            print("Warning: NaN detected in training data")
            X = torch.nan_to_num(X, 0.0)
            y = torch.nan_to_num(y, 0.0)
            
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize weights
        for param in self.model.parameters():
            if param.dim() > 1:
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
                    print(f"NaN loss at epoch {epoch}, skipping batch")
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.fc.parameters()), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def save_model(self, path):
        torch.save({
            'lstm_state': self.model.state_dict(),
            'fc_state': self.fc.state_dict()
        }, path)

def run_training_simulations(num_simulations=20):
    """Run multiple simulations with random agents for training data"""
    all_results = []
    
    for i in range(num_simulations):
        print(f"Running training simulation {i+1}/{num_simulations}")
        
        # Create random agents
        agents = [
            RandomAgent(f"random_{j}", 10000) 
            for j in range(3)
        ]
        
        # Run simulation
        sim = MarketSimulation(agents)
        sim.run(500)  # Shorter simulations for training
        
        results_df = sim.get_results()
        all_results.append(results_df)
    
    return all_results

def main():
    print("Starting LSTM training phase...")
    
    # Run training simulations
    training_data = run_training_simulations(5)  # Reduce to 5 for faster training
    
    # Prepare training data
    trainer = LSTMTrainer()
    all_X, all_y = [], []
    
    for sim_data in training_data:
        if len(sim_data) > 10:  # Ensure enough data
            X, y = trainer.prepare_data(sim_data)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
    
    if not all_X:
        print("No valid training data generated")
        return
        
    # Combine all training data
    combined_X = torch.cat(all_X, dim=0)
    combined_y = torch.cat(all_y, dim=0)
    
    print(f"Training on {len(combined_X)} samples...")
    
    # Train model
    trainer.train(combined_X, combined_y, epochs=20)  # Reduce epochs
    
    # Save trained model
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)
    trainer.save_model(os.path.join(results_dir, 'trained_lstm.pth'))
    
    print("LSTM training completed!")

if __name__ == "__main__":
    main()