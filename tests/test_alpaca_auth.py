import os
import unittest

import pathsetup  # noqa: F401

from broker.alpaca_auth import assert_paper_or_allowed, is_live_base_url


class LiveGuardTests(unittest.TestCase):
    def test_paper_url_is_not_live(self):
        self.assertFalse(is_live_base_url('https://paper-api.alpaca.markets'))
        self.assertFalse(is_live_base_url('https://paper-api.alpaca.markets/v2'))

    def test_live_url_is_live(self):
        self.assertTrue(is_live_base_url('https://api.alpaca.markets'))
        self.assertTrue(is_live_base_url('https://api.alpaca.markets/v2'))

    def test_empty_is_not_live(self):
        self.assertFalse(is_live_base_url(''))
        self.assertFalse(is_live_base_url(None))

    def test_assert_rejects_live_without_flag(self):
        os.environ['ALPACA_ALLOW_LIVE'] = 'false'
        os.environ['APCA_API_KEY_ID'] = 'key'
        os.environ['APCA_API_SECRET_KEY'] = 'secret'
        os.environ['APCA_API_BASE_URL'] = 'https://api.alpaca.markets'
        with self.assertRaises(RuntimeError):
            assert_paper_or_allowed()

    def test_assert_allows_paper(self):
        os.environ['ALPACA_ALLOW_LIVE'] = 'false'
        os.environ['APCA_API_KEY_ID'] = 'key'
        os.environ['APCA_API_SECRET_KEY'] = 'secret'
        os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
        creds = assert_paper_or_allowed()
        self.assertTrue(creds['paper'])

    def test_assert_allows_live_with_flag(self):
        os.environ['ALPACA_ALLOW_LIVE'] = 'true'
        os.environ['APCA_API_KEY_ID'] = 'key'
        os.environ['APCA_API_SECRET_KEY'] = 'secret'
        os.environ['APCA_API_BASE_URL'] = 'https://api.alpaca.markets'
        creds = assert_paper_or_allowed()
        self.assertFalse(creds['paper'])
        self.assertTrue(creds['allow_live'])


if __name__ == '__main__':
    unittest.main()
