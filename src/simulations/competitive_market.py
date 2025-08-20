import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from ..agents.base_agent import BaseAgent
from .order_book import OrderBook

class CompetitiveMarketSimulation:
    """More competitive market with limited liquidity and faster execution"""
    
    def __init__(self, agents: List[BaseAgent], max_spread: float = 5.0, 
                 liquidity_factor: float = 0.3, execution_delay: int = 0):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.order_book = OrderBook()
        self.step = 0
        self.history = []
        
        # Competitive parameters
        self.max_spread = max_spread  # Maximum allowed spread
        self.liquidity_factor = liquidity_factor  # Reduces available liquidity
        self.execution_delay = execution_delay  # Steps before order becomes active
        self.pending_orders = []  # Orders waiting to be activated
        
        # Market maker to provide base liquidity
        self._seed_market()
    
    def _seed_market(self):
        """Add initial market maker orders for base liquidity"""
        base_price = 100.0
        for i in range(5):
            # Bid side
            self.order_book.add_order('buy', 10, base_price - (i+1) * 0.5, 'market_maker')
            # Ask side  
            self.order_book.add_order('sell', 10, base_price + (i+1) * 0.5, 'market_maker')
    
    def _get_competitive_market_state(self) -> Dict[str, Any]:
        """Enhanced market state with competitive information"""
        best_bid, best_ask = self.order_book.get_best_bid_ask()
        spread = self.order_book.get_spread() or 1.0
        
        # Calculate market pressure (order imbalance)
        bid_volume = sum(qty for _, qty, _, _ in self.order_book.bids[:5])  # Top 5 levels
        ask_volume = sum(qty for _, qty, _, _ in self.order_book.asks[:5])
        
        total_volume = bid_volume + ask_volume
        market_pressure = (bid_volume - ask_volume) / max(total_volume, 1) if total_volume > 0 else 0
        
        # Recent trade activity
        recent_trades = len([t for t in self.order_book.trades[-10:]])
        
        return {
            'best_bid': best_bid or 100,
            'best_ask': best_ask or 101,
            'spread': spread,
            'step': self.step,
            'market_pressure': market_pressure,  # -1 (sell pressure) to +1 (buy pressure)
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'recent_trades': recent_trades,
            'liquidity_available': min(bid_volume, ask_volume) * self.liquidity_factor
        }
    
    def _process_pending_orders(self):
        """Activate orders after delay period"""
        if self.execution_delay == 0:
            return
            
        ready_orders = [order for order in self.pending_orders if order['ready_step'] <= self.step]
        self.pending_orders = [order for order in self.pending_orders if order['ready_step'] > self.step]
        
        for order in ready_orders:
            trades = self.order_book.add_order(
                order['type'], order['quantity'], order['price'], order['agent_id']
            )
            self._apply_trades_to_agent(order['agent_id'], trades)
    
    def _enforce_spread_limit(self):
        """Remove orders that create excessive spread"""
        best_bid, best_ask = self.order_book.get_best_bid_ask()
        if best_bid and best_ask and (best_ask - best_bid) > self.max_spread:
            # Remove extreme orders
            while self.order_book.bids and -self.order_book.bids[0][0] < best_ask - self.max_spread:
                self.order_book.bids.pop(0)
            while self.order_book.asks and self.order_book.asks[0][0] > best_bid + self.max_spread:
                self.order_book.asks.pop(0)
    
    def run_step(self):
        """Competitive simulation step with order processing"""
        # Process pending orders first
        self._process_pending_orders()
        
        # Get market state
        market_state = self._get_competitive_market_state()
        
        # Shuffle agents for fair competition
        agent_list = list(self.agents.values())
        np.random.shuffle(agent_list)
        
        # Agents act in random order
        for agent in agent_list:
            action = agent.act(market_state)
            
            if action['type'] in ['buy', 'sell']:
                # Apply liquidity constraints
                max_quantity = int(market_state['liquidity_available'])
                actual_quantity = min(action.get('quantity', 1), max_quantity)
                
                if actual_quantity > 0:
                    if self.execution_delay > 0:
                        # Add to pending orders
                        self.pending_orders.append({
                            'type': action['type'],
                            'quantity': actual_quantity,
                            'price': action['price'],
                            'agent_id': agent.agent_id,
                            'ready_step': self.step + self.execution_delay
                        })
                    else:
                        # Immediate execution
                        trades = self.order_book.add_order(
                            action['type'], actual_quantity, action['price'], agent.agent_id
                        )
                        self._apply_trades_to_agent(agent.agent_id, trades)
        
        # Enforce market rules
        self._enforce_spread_limit()
        
        # Add market maker activity to maintain liquidity
        self._market_maker_activity()
        
        # Record step
        self._record_step()
        self.step += 1
    
    def _market_maker_activity(self):
        """Market maker adds liquidity when spread is too wide"""
        best_bid, best_ask = self.order_book.get_best_bid_ask()
        if best_bid and best_ask:
            spread = best_ask - best_bid
            if spread > 2.0:  # Add liquidity when spread > 2
                mid_price = (best_bid + best_ask) / 2
                # Add tight quotes
                self.order_book.add_order('buy', 5, mid_price - 0.25, 'market_maker')
                self.order_book.add_order('sell', 5, mid_price + 0.25, 'market_maker')
    
    def _apply_trades_to_agent(self, agent_id: str, trades: List[Dict]):
        """Apply trades to specific agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            for trade in trades:
                if trade['buyer'] == agent_id:
                    agent.update_position(-trade['price'] * trade['quantity'], trade['quantity'])
                elif trade['seller'] == agent_id:
                    agent.update_position(trade['price'] * trade['quantity'], -trade['quantity'])
    
    def _record_step(self):
        """Record competitive market statistics"""
        best_bid, best_ask = self.order_book.get_best_bid_ask()
        mid_price = ((best_bid or 100) + (best_ask or 101)) / 2
        
        step_data = {
            'step': self.step,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': self.order_book.get_spread(),
            'mid_price': mid_price,
            'total_trades': len(self.order_book.trades),
            'pending_orders': len(self.pending_orders)
        }
        
        # Agent data
        for agent in self.agents.values():
            agent.calculate_pnl(mid_price)
            step_data[f'{agent.agent_id}_pnl'] = agent.pnl
            step_data[f'{agent.agent_id}_cash'] = agent.cash
            step_data[f'{agent.agent_id}_inventory'] = agent.inventory
            step_data[f'{agent.agent_id}_orders'] = agent.orders_count
            
        self.history.append(step_data)
    
    def run(self, num_steps: int):
        """Run competitive simulation"""
        for _ in range(num_steps):
            self.run_step()
    
    def get_results(self) -> pd.DataFrame:
        """Get simulation results"""
        return pd.DataFrame(self.history)
    
    def get_competition_stats(self) -> Dict[str, Any]:
        """Get competitive market statistics"""
        total_trades = len(self.order_book.trades)
        agent_trades = {}
        
        for trade in self.order_book.trades:
            buyer = trade['buyer']
            seller = trade['seller']
            if buyer != 'market_maker':
                agent_trades[buyer] = agent_trades.get(buyer, 0) + 1
            if seller != 'market_maker':
                agent_trades[seller] = agent_trades.get(seller, 0) + 1
        
        return {
            'total_trades': total_trades,
            'agent_trades': agent_trades,
            'avg_trades_per_agent': np.mean(list(agent_trades.values())) if agent_trades else 0,
            'most_active_agent': max(agent_trades.items(), key=lambda x: x[1]) if agent_trades else None
        }