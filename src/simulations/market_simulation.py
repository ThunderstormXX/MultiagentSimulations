import pandas as pd
from typing import List, Dict, Any, Optional
from ..agents.base_agent import BaseAgent
from .order_book import OrderBook

import numpy as np

class MarketSimulation:
    def __init__(self, agents: List[BaseAgent], record_lob: bool = False, seed: Optional[int] = None,
                 exog_mu: float = 0.0, exog_sigma: float = 0.5):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.order_book = OrderBook()
        self.step = 0
        self.history: List[Dict[str, Any]] = []
        self.trades_history: List[Dict[str, Any]] = []
        self.lob_history: List[Dict[str, Any]] = []
        self.record_lob = record_lob

        # экзогенный фактор
        self.exog_price = 100.0
        self.exog_mu = exog_mu
        self.exog_sigma = exog_sigma

        if seed is not None:
            import numpy as np, random, torch
            random.seed(seed)
            np.random.seed(seed)
            try:
                torch.manual_seed(seed)
            except ImportError:
                pass

    def _update_exog_price(self):
        """Простое стохастическое движение экзогенной цены."""
        dt = 1
        shock = np.random.randn() * self.exog_sigma * np.sqrt(dt)
        self.exog_price *= np.exp(self.exog_mu * dt + shock)

    def run_step(self):
        """Run one simulation step: agents act, trades execute, stats are recorded."""
        # обновляем экзогенную цену
        self._update_exog_price()

        # market_state как раньше, просто добавляем exog_price
        market_state = self._get_market_state()
        market_state['exog_price'] = self.exog_price

        for agent in self.agents.values():
            action = agent.act(market_state)
            if action["type"] in ["buy", "sell"]:
                trades = self.order_book.add_order(
                    action["type"], action["quantity"], action["price"], agent.agent_id
                )
                self._apply_trades(agent, trades)

        self._record_step()
        self.step += 1

    def run(self, num_steps: int):
        """Run simulation for a fixed number of steps."""
        for _ in range(num_steps):
            self.run_step()

    def get_results(self) -> pd.DataFrame:
        """Return per-step statistics as DataFrame."""
        return pd.DataFrame(self.history)

    def get_trades(self) -> pd.DataFrame:
        """Return trade history as DataFrame."""
        return pd.DataFrame(self.trades_history)

    def get_lob_history(self) -> List[Dict[str, Any]]:
        """Return full LOB snapshots (if record_lob=True)."""
        return self.lob_history
    
    def _get_market_state(self) -> Dict[str, Any]:
        """Get current market state for agents"""
        best_bid, best_ask = self.order_book.get_best_bid_ask()
        return {
            'best_bid': best_bid or 100,
            'best_ask': best_ask or 101,
            'spread': self.order_book.get_spread() or 1,
            'step': self.step
        }
    
    def _apply_trades(self, agent: BaseAgent, trades: List[Dict]):
        """Apply trades to agent positions"""
        for trade in trades:
            if trade['buyer'] == agent.agent_id:
                agent.update_position(-trade['price'] * trade['quantity'], trade['quantity'])
            elif trade['seller'] == agent.agent_id:
                agent.update_position(trade['price'] * trade['quantity'], -trade['quantity'])
    
    def _record_step(self):
        """Record current step statistics"""
        best_bid, best_ask = self.order_book.get_best_bid_ask()
        mid_price = ((best_bid or 100) + (best_ask or 101)) / 2
        
        step_data = {
            'step': self.step,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': self.order_book.get_spread(),
            'mid_price': mid_price
        }
        
        # Add agent data
        for agent in self.agents.values():
            agent.calculate_pnl(mid_price)
            step_data[f'{agent.agent_id}_pnl'] = agent.pnl
            step_data[f'{agent.agent_id}_cash'] = agent.cash
            step_data[f'{agent.agent_id}_inventory'] = agent.inventory
            step_data[f'{agent.agent_id}_orders'] = agent.orders_count
            
        self.history.append(step_data)
