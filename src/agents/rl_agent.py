import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Any, Optional
from .base_agent import BaseAgent


class LSTMNetwork(nn.Module):
    """
    Simple LSTM network for market feature processing.
    Produces logits over possible actions.
    """
    def __init__(self, input_size: int = 10, hidden_size: int = 64, output_size: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # outputs logits for actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # take last hidden state


class RLAgent(BaseAgent):
    """
    Reinforcement Learning agent with LSTM policy network.
    Can load pre-trained model for better performance.
    """
    ACTIONS = ["buy", "sell", "cancel"]

    def __init__(self, agent_id: str, initial_cash: float = 10_000.0, 
                 model_path: str = None, epsilon: float = 0.1):
        super().__init__(agent_id, initial_cash)
        
        self.device = "cpu"
        self.model = None
        self.fc = None
        self.epsilon = epsilon
        self.history = []
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def act(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an action using ε-greedy strategy:
        - with prob ε → random action
        - otherwise → argmax from network (placeholder: heuristic)
        """
        if np.random.random() < self.epsilon:
            return self._random_action(market_state)
        else:
            return self._heuristic_action(market_state)
            # позже здесь будет: return self._network_action(market_state)

    def load_model(self, model_path: str):
        """Load trained LSTM model"""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model = nn.LSTM(6, 64, batch_first=True)
        self.fc = nn.Linear(64, 3)
        self.model.load_state_dict(checkpoint['lstm_state'])
        self.fc.load_state_dict(checkpoint['fc_state'])
        self.model.eval()
        self.fc.eval()
    
    def _heuristic_action(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """LSTM-based action if model available, else simple heuristic"""
        best_bid = market_state.get("best_bid", 100.0)
        best_ask = market_state.get("best_ask", 101.0)
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        # Store market state
        self.history.append([best_bid, best_ask, spread, mid_price, self.pnl, self.inventory])
        
        # Use LSTM if available and enough history
        if self.model and len(self.history) >= 10:
            with torch.no_grad():
                input_data = torch.FloatTensor([self.history[-10:]])
                lstm_out, _ = self.model(input_data)
                prediction = self.fc(lstm_out[:, -1, :])
                
                price_change_pred = prediction[0][0].item()
                
                # Make decision based on prediction
                if price_change_pred > 0.001:  # Expect price to rise
                    action = {"type": "buy", "quantity": 3, "price": round(best_bid + 0.01, 2)}
                elif price_change_pred < -0.001:  # Expect price to fall
                    action = {"type": "sell", "quantity": 3, "price": round(best_ask - 0.01, 2)}
                else:
                    action = {"type": "cancel"}
        else:
            # Fallback heuristic
            if spread > 2:  
                if np.random.random() > 0.5:
                    action = {"type": "buy", "quantity": 5, "price": round(best_bid + 0.01, 2)}
                else:
                    action = {"type": "sell", "quantity": 5, "price": round(best_ask - 0.01, 2)}
            else:
                action = {"type": "cancel"}

        self.orders_count += 1
        return action

    def _random_action(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """Exploration: pick random action from ACTIONS."""
        action_type = np.random.choice(self.ACTIONS)
        return {"type": action_type, "quantity": 1, "price": market_state.get("best_bid", 100.0)}  

    def _network_action(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placeholder: convert market_state to tensor and query LSTM.
        """
        features = self._extract_features(market_state)
        with torch.no_grad():
            logits = self.network(features.to(self.device))
            action_idx = torch.argmax(logits, dim=-1).item()
        
        action_type = self.ACTIONS[action_idx]
        return {"type": action_type, "quantity": 1, "price": market_state.get("best_bid", 100.0)}

    def _extract_features(self, market_state: Dict[str, Any]) -> torch.Tensor:
        """
        Convert market_state dict into feature tensor for the network.
        Currently dummy: best_bid, best_ask, spread, step → padded to input_size.
        """
        best_bid = market_state.get("best_bid", 100.0)
        best_ask = market_state.get("best_ask", 101.0)
        spread = market_state.get("spread", best_ask - best_bid)
        step = market_state.get("step", 0)

        features = np.array([best_bid, best_ask, spread, step] + [0.0]*6, dtype=np.float32)
        features = features.reshape(1, 1, -1)  # (batch=1, seq=1, input_size)
        return torch.tensor(features)

    def save_model(self, path: str):
        if self.model and self.fc:
            torch.save({
                'lstm_state': self.model.state_dict(),
                'fc_state': self.fc.state_dict()
            }, path)
