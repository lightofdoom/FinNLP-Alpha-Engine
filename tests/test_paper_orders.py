import unittest

import pathsetup  # noqa: F401

from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from broker.alpaca_trading import (
    build_exit_order_request,
    build_market_order_request,
    limit_price_from_quote,
    qty_from_notional,
)
from live.paper_trader import check_kill_switch, plan_paper_order


class OrderPolicyTests(unittest.TestCase):
    def test_buy_limit_uses_ask_plus_offset(self):
        quote = {'bid': 100.0, 'ask': 100.2, 'last': 100.1}
        price = limit_price_from_quote('buy', quote, offset_bps=5)
        self.assertEqual(price, round(100.2 * 1.0005, 2))

    def test_sell_limit_uses_bid_minus_offset(self):
        quote = {'bid': 100.0, 'ask': 100.2, 'last': 100.1}
        price = limit_price_from_quote('sell', quote, offset_bps=5)
        self.assertEqual(price, round(100.0 * 0.9995, 2))

    def test_qty_from_notional(self):
        self.assertEqual(qty_from_notional(1000, 233.0), 4)

    def test_extended_hours_flag(self):
        quote = {'bid': 100.0, 'ask': 100.2, 'last': 100.1}
        plan = plan_paper_order('AMZN', 'buy', quote, 'premarket', notional=1000)
        self.assertTrue(plan['request'].extended_hours)
        plan_rth = plan_paper_order('AMZN', 'buy', quote, 'regular', notional=1000)
        self.assertFalse(plan_rth['request'].extended_hours)

    def test_closed_refuses(self):
        quote = {'bid': 100.0, 'ask': 100.2, 'last': 100.1}
        with self.assertRaises(RuntimeError):
            plan_paper_order('AMZN', 'buy', quote, 'closed', notional=1000)

    def test_open_order_refuses(self):
        quote = {'bid': 100.0, 'ask': 100.2, 'last': 100.1}
        with self.assertRaises(RuntimeError):
            plan_paper_order('AMZN', 'buy', quote, 'regular', notional=1000, open_order_count=1)

    def test_position_refuses_without_flatten(self):
        quote = {'bid': 100.0, 'ask': 100.2, 'last': 100.1}
        with self.assertRaises(RuntimeError):
            plan_paper_order('AMZN', 'buy', quote, 'regular', notional=1000, has_position=True)


class ExitOrderTests(unittest.TestCase):
    def test_urgent_rth_is_market(self):
        quote = {'bid': 100.0, 'ask': 100.2, 'last': 100.1}
        request = build_exit_order_request('AMZN', 'sell', 4, 'regular', quote, urgent=True)
        self.assertIsInstance(request, MarketOrderRequest)
        self.assertFalse(request.extended_hours)

    def test_urgent_extended_is_ioc_limit(self):
        quote = {'bid': 100.0, 'ask': 100.2, 'last': 100.1}
        request = build_exit_order_request('AMZN', 'sell', 4, 'afterhours', quote, urgent=True)
        self.assertEqual(request.time_in_force, TimeInForce.IOC)
        self.assertTrue(request.extended_hours)
        self.assertIsNotNone(request.limit_price)

    def test_non_urgent_is_day_limit(self):
        quote = {'bid': 100.0, 'ask': 100.2, 'last': 100.1}
        request = build_exit_order_request('AMZN', 'sell', 4, 'regular', quote, urgent=False)
        self.assertEqual(request.time_in_force, TimeInForce.DAY)
        self.assertIsNotNone(request.limit_price)

    def test_market_refuses_extended_hours(self):
        with self.assertRaises(ValueError):
            build_market_order_request('AMZN', 'sell', 4, 'afterhours')


class KillSwitchTests(unittest.TestCase):
    def test_max_orders(self):
        with self.assertRaises(RuntimeError):
            check_kill_switch({'orders_submitted': 10, 'starting_equity': 100000}, 100000, max_orders=10, max_loss=2000)

    def test_max_loss(self):
        with self.assertRaises(RuntimeError):
            check_kill_switch({'orders_submitted': 1, 'starting_equity': 100000}, 97000, max_orders=10, max_loss=2000)

    def test_ok(self):
        self.assertTrue(
            check_kill_switch({'orders_submitted': 1, 'starting_equity': 100000}, 99500, max_orders=10, max_loss=2000)
        )


if __name__ == '__main__':
    unittest.main()
