from typing import Dict, List, Tuple, Optional, Any
import heapq
import itertools


class OrderBook:
    """
    Simple limit order book implementation with bids/asks stored in heaps.
    Supports matching, spread calculation, and cancellations.
    """

    def __init__(self):
        self.bids: List[Tuple[float, int, str, int]] = []  # (-price, qty, agent, order_id)
        self.asks: List[Tuple[float, int, str, int]] = []  # (price, qty, agent, order_id)
        self.trades: List[Dict[str, Any]] = []
        self._order_id_counter = itertools.count(1)
        self._active_orders: Dict[int, bool] = {}  # track cancellations

    def add_order(self, order_type: str, quantity: int, price: float, agent_id: str) -> List[Dict[str, Any]]:
        """
        Add a buy/sell order and perform matching.
        Returns list of executed trades.
        """
        trades = []
        order_id = next(self._order_id_counter)
        self._active_orders[order_id] = True

        if order_type == "buy":
            # Match against asks
            while self.asks and quantity > 0 and self.asks[0][0] <= price:
                ask_price, ask_qty, ask_agent, ask_id = heapq.heappop(self.asks)
                if not self._active_orders.get(ask_id, True):
                    continue  # skip canceled

                trade_qty = min(quantity, ask_qty)
                trades.append({
                    "buyer": agent_id, "seller": ask_agent,
                    "quantity": trade_qty, "price": ask_price
                })

                quantity -= trade_qty
                if ask_qty > trade_qty:
                    heapq.heappush(self.asks, (ask_price, ask_qty - trade_qty, ask_agent, ask_id))

            if quantity > 0:
                heapq.heappush(self.bids, (-price, quantity, agent_id, order_id))

        elif order_type == "sell":
            # Match against bids
            while self.bids and quantity > 0 and -self.bids[0][0] >= price:
                neg_bid_price, bid_qty, bid_agent, bid_id = heapq.heappop(self.bids)
                if not self._active_orders.get(bid_id, True):
                    continue  # skip canceled

                bid_price = -neg_bid_price
                trade_qty = min(quantity, bid_qty)
                trades.append({
                    "buyer": bid_agent, "seller": agent_id,
                    "quantity": trade_qty, "price": bid_price
                })

                quantity -= trade_qty
                if bid_qty > trade_qty:
                    heapq.heappush(self.bids, (neg_bid_price, bid_qty - trade_qty, bid_agent, bid_id))

            if quantity > 0:
                heapq.heappush(self.asks, (price, quantity, agent_id, order_id))

        self.trades.extend(trades)
        return trades

    def cancel_order(self, order_id: int):
        """Mark order as inactive. Actual removal happens lazily during pop."""
        self._active_orders[order_id] = False

    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """Return current best bid and ask prices."""
        best_bid = -self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return best_bid, best_ask

    def get_spread(self) -> Optional[float]:
        """Return spread (ask - bid)."""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None

    def snapshot(self) -> Dict[str, Any]:
        """Return full order book snapshot (for logging/debugging)."""
        return {
            "bids": [(-p, q, a, oid) for p, q, a, oid in self.bids],
            "asks": [(p, q, a, oid) for p, q, a, oid in self.asks],
            "best_bid": self.get_best_bid_ask()[0],
            "best_ask": self.get_best_bid_ask()[1],
            "spread": self.get_spread(),
            "trades": list(self.trades[-10:])  # last N trades
        }
