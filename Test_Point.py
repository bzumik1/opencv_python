import unittest

from SoftwareVision import ImagePoint


class MyTestCase(unittest.TestCase):
    def test_returns_correct_distance(self):
        p1 = ImagePoint(0, 0)
        p2 = ImagePoint(3, 4)
        self.assertEqual(5.0, p1.distance(p2))


if __name__ == '__main__':
    unittest.main()
