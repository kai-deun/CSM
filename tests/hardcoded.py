import unittest
import core.core as gj


class MyTestCase(unittest.TestCase):
    def test_multiple_matrices(self):
        test_matrix = [
            {
                'a': [[0, 2, 0, 1],
                      [2, 2, 3, 2],
                      [4, -3, 0, 1],
                      [6, 1, -6, -5]],

                'b': [0, -2, -7, 6],
                'expected': [-0.5, 1, 0.33333333, -2],
            },
            {
                'a': [[12, 2, 0, 1],
                      [2, 0, 3, 2],
                      [4, -3, 0, 1],
                      [6, 1, -6, -5]],

                'b': [0, -2, -7, 6],
                'expected': [-0.12790698, 1.60465116, 0.53488372, -1.6744186],
            },
        ]

        # running the multiple matrix test
        for case in test_matrix:
            with self.subTest(case=case):
                solution, _ = gj.gaussjordan(case['a'], case['b'])
                for sol, ex in zip(solution, case['expected']):
                    self.assertAlmostEqual(sol, ex, places=6)

    def test_matrix_letters(self):
        test_matrix = [
            {
                'a': [[0, 'q', 0, 1],
                      [2, 2, 3, 2],
                      [4, -3, 0, 1],
                      [6, 1, -6, -5]],

                'b': [0, -2, -7, 6],
                'expected': [-0.5, 1, 0.33333333, -2],
            },
            {
                'a': [[12, 2, 0, 1],
                      [2, 0, 3, '@'],
                      [4, -3, 0, 1],
                      [6, 1, -6, -5]],

                'b': [0, -2, -7, 6],
                'expected': [-0.12790698, 1.60465116, 0.53488372, -1.6744186],
            },
        ]

        for case in test_matrix:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    gj.gaussjordan(case['a'], case['b'])


if __name__ == '__main__':
    unittest.main()
