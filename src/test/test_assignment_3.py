import unittest

class TestAssignment3B(unittest.TestCase):

    def test_gaussian_elimination(self):
        matrix = [
            [2, -1, 1, 6],
            [1, 3, 1, 0],
            [-1, 5, 4, -3]
        ]
        expected = [1.2446380979332121, 1.251316587879806, 1.0]
        result = gaussian_elimination(matrix)
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=10)

    def test_diagonal_dominance(self):
        matrix = [
            [9, 0, 5, 2, 1],
            [3, 9, 1, 2, 1],
            [0, 1, 7, 2, 3],
            [4, 2, 3, 12, 2],
            [3, 2, 4, 0, 8]
        ]
        self.assertTrue(is_diagonally_dominant(matrix))

    def test_positive_definite(self):
        matrix = [
            [2, 2, 1],
            [2, 3, 0],
            [1, 0, 2]
        ]
        self.assertTrue(is_positive_definite(matrix))

unittest.main(argv=['first-arg-is-ignored'], exit=False)
