from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseAgent(ABC):
    """
    Abstract base class for trading agents.
    
    Each agent can take actions (buy/sell/cancel) based on the market state,
    manages its cash and inventory, and tracks its PnL.
    """
    
    def __init__(self, agent_id: str, initial_cash: float = 10_000.0):
        self.agent_id: str = agent_id
        self.initial_cash: float = initial_cash
        
        # Current state
        self.cash: float = initial_cash
        self.inventory: int = 0
        self.pnl: float = 0.0
        self.orders_count: int = 0
        
        # Optional tracking
        self.history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def act(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide the next action based on the current market state.
        
        Should return a dictionary with fields:
        {
            'type': 'buy' | 'sell' | 'cancel' | 'hold',
            'quantity': int,
            'price': float (ignored for cancel/hold)
        }
        """
        pass
    
    def update_position(self, cash_change: float, inventory_change: int):
        """Update cash and inventory after a trade execution."""
        self.cash += cash_change
        self.inventory += inventory_change
    
    def calculate_pnl(self, current_price: float) -> float:
        """
        Compute Profit and Loss (PnL) as current cash + inventory value - initial cash.
        """
        self.pnl = self.cash + self.inventory * current_price - self.initial_cash
        return self.pnl
    
    def record_state(self, step: int, current_price: Optional[float] = None):
        """
        Save agent's state for debugging and analysis.
        """
        record = {
            "step": step,
            "cash": self.cash,
            "inventory": self.inventory,
            "orders": self.orders_count,
            "pnl": self.pnl,
        }
        if current_price is not None:
            record["market_price"] = current_price
        self.history.append(record)
    
    def __repr__(self) -> str:
        return (f"<Agent {self.agent_id} | "
                f"Cash: {self.cash:.2f}, Inv: {self.inventory}, "
                f"PnL: {self.pnl:.2f}>")
