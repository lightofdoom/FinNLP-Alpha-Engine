import time
import unittest

import pathsetup  # noqa: F401

from broker.rate_limit import TokenBucket


class RateLimitTests(unittest.TestCase):
    def test_acquire_then_block(self):
        bucket = TokenBucket(rate_per_min=60)
        bucket.tokens = 1
        self.assertTrue(bucket.acquire(block=False))
        self.assertFalse(bucket.acquire(block=False))

    def test_refill(self):
        bucket = TokenBucket(rate_per_min=60)
        bucket.tokens = 0
        bucket.updated = time.monotonic() - 2
        self.assertTrue(bucket.acquire(block=False))


if __name__ == '__main__':
    unittest.main()
