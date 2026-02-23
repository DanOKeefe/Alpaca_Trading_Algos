from unittest.mock import MagicMock

from alpaca_trade_api.rest import APIError

from gmv_algo import submit_order


class TestSubmitOrder:
    def test_buy_order_calls_api(self):
        api = MagicMock()
        submit_order(api, 10, 'AAPL', 'buy')
        api.submit_order.assert_called_once_with('AAPL', 10, 'buy', 'market', 'day')

    def test_sell_order_calls_api(self):
        api = MagicMock()
        submit_order(api, 5, 'MSFT', 'sell')
        api.submit_order.assert_called_once_with('MSFT', 5, 'sell', 'market', 'day')

    def test_zero_quantity_skips_order(self):
        api = MagicMock()
        submit_order(api, 0, 'AAPL', 'buy')
        api.submit_order.assert_not_called()

    def test_negative_quantity_skips_order(self):
        api = MagicMock()
        submit_order(api, -5, 'AAPL', 'buy')
        api.submit_order.assert_not_called()

    def test_api_error_does_not_raise(self):
        api = MagicMock()
        api.submit_order.side_effect = APIError({'message': 'insufficient funds'})
        # Should not propagate the exception
        submit_order(api, 10, 'AAPL', 'buy')

    def test_returns_none(self):
        api = MagicMock()
        result = submit_order(api, 10, 'AAPL', 'buy')
        assert result is None
