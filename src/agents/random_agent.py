import random
from typing import Dict, Any, Optional
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    A simple agent that takes random actions: buy, sell, cancel, or hold.
    
    - Buy: posts a bid around the best bid.
    - Sell: posts an ask around the best ask.
    - Cancel: cancels an order (placeholder for now).
    - Hold: does nothing.
    """
    
    def __init__(self, agent_id: str, initial_cash: float = 10_000.0, seed: Optional[int] = None):
        super().__init__(agent_id, initial_cash)
        if seed is not None:
            random.seed(seed)  # for reproducibility
    
    def act(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        action_type = random.choice(["buy", "sell", "cancel", "hold"])
        
        # Cancel → no specific order targeted (placeholder)
        if action_type == "cancel":
            return {"type": "cancel", "order_id": None}
        
        # Hold → no action
        if action_type == "hold":
            return {"type": "hold"}
        
        # Extract market info
        best_bid = market_state.get("best_bid", 100.0)
        best_ask = market_state.get("best_ask", 101.0)
        
        # Generate order parameters
        price = self._generate_price(action_type, best_bid, best_ask)
        quantity = self._generate_quantity()
        
        self.orders_count += 1
        
        return {
            "type": action_type,
            "quantity": quantity,
            "price": round(price, 2)
        }
    
    def _generate_price(self, action_type: str, best_bid: float, best_ask: float) -> float:
        """Generate a random price depending on the action type."""
        if action_type == "buy":
            return random.uniform(best_bid * 0.95, best_bid * 1.02)
        elif action_type == "sell":
            return random.uniform(best_ask * 0.98, best_ask * 1.05)
        else:
            raise ValueError(f"Unsupported action type: {action_type}")
    
    def _generate_quantity(self) -> int:
        """Generate a random trade quantity."""
        return random.randint(1, 10)
