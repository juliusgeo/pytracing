from main import Point, Vector, Tuple, Projectile, Environment, Color, Canvas, Matrix

import unittest
import math


class TupleTests(unittest.TestCase):
    def test_add(self):
        a = Point(3, -2, 5)
        b = Vector(-2, 3, 1)
        self.assertEqual(a + b, Tuple(1, 1, 6, 1))

    def test_sub(self):
        p1 = Point(3, 2, 1)
        p2 = Point(5, 6, 7)
        self.assertEqual(p1 - p2, Vector(-2, -4, -6))
        p1 = Point(3, 2, 1)
        p2 = Vector(5, 6, 7)
        self.assertEqual(p1 - p2, Point(-2, -4, -6))
        p1 = Vector(3, 2, 1)
        p2 = Vector(5, 6, 7)
        self.assertEqual(p1 - p2, Vector(-2, -4, -6))

    def test_negation(self):
        self.assertEqual(-Tuple(1, -2, 3, -4), Tuple(-1, 2, -3, 4))

    def test_dot(self):
        a = Vector(1, 2, 3)
        b = Vector(2, 3, 4)
        self.assertEqual(a.dot(b), 20)

    def test_cross(self):
        a = Vector(1, 2, 3)
        b = Vector(2, 3, 4)
        self.assertEqual(a * b, Vector(-1, 2, -1))
        self.assertEqual(b * a, Vector(1, -2, 1))

    def test_scalar_mul(self):
        self.assertEqual(Tuple(1, -2, 3, -4) * 3.5, Tuple(3.5, -7, 10.5, -14))
        self.assertEqual(Tuple(1, -2, 3, -4) * .5, Tuple(0.5, -1, 1.5, -2))

    def test_scalar_div(self):
        self.assertEqual(Tuple(1, -2, 3, -4) / 2, Tuple(0.5, -1, 1.5, -2))

    def test_magnitude(self):
        combos = [(Vector(0, 1, 0), 1),
                  (Vector(0, 0, 1), 1),
                  (Vector(1, 2, 3), math.sqrt(14)),
                  (Vector(-1, -2, -3), math.sqrt(14))]

        for v, m in combos:
            self.assertEqual(abs(v), m)

    def test_normalize(self):
        self.assertEqual(Vector(4, 0, 0).normalize(), Vector(1, 0, 0))
        self.assertEqual(Vector(1, 2, 3).normalize(), Vector(0.26726, 0.53452, 0.80178))

    def test_projectiles(self):
        p = Projectile(Point(0, 1, 0), Vector(1, 1, 0).normalize())
        e = Environment(Vector(0, -0.1, 0), Vector(-0.01, 0, 0))
        while p.position.y > 0:
            print(p.position)
            e.tick(p)

    def test_colors(self):
        self.assertEqual(Color(0.9, 0.6, 0.75) + Color(0.7, 0.1, 0.25), Color(1.6, 0.7, 1.0))
        self.assertEqual(Color(0.9, 0.6, 0.75) - Color(0.7, 0.1, 0.25), Color(0.2, 0.5, 0.5))
        self.assertEqual(Color(0.2, 0.3, 0.4) * 2, Color(0.4, 0.6, 0.8))
        self.assertEqual(Color(1, 0.2, 0.4) * Color(0.9, 1, 0.1), Color(0.9, 0.2, 0.04))

    def test_canvas(self):
        c = Canvas(10, 20)
        red = Color(1, 0, 0)
        c.write_pixel(2, 3, red)
        self.assertEqual(c[3 * 10 + 2], red)

    def test_ppm(self):
        c = Canvas(5, 3)
        c.write_pixel(0, 0, Color(1.5, 0, 0))
        c.write_pixel(2, 1, Color(0, 0.5, 0))
        c.write_pixel(4, 2, Color(-0.5, 0, 1))
        self.assertEqual(c.ppm_file(),
                         """P3
5 3
255
255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 255""")

        c = Canvas(10, 2)
        for i in range(20):
            c[i] = Color(1, 0.8, 0.6)
        self.assertEqual(c.ppm_file(),
                         """P3
10 2
255
255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 
255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 
255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 
255 204 153 255 204 153""")
        c.ppm_file("out.ppm")

    def test_projectile_canvas(self):
        p = Projectile(Point(0, 1, 0), Vector(1, 1.8, 0).normalize() * 11.25)
        e = Environment(Vector(0, -0.1, 0), Vector(-0.01, 0, 0))
        c = Canvas(900, 550)
        while p.position.y > 0:
            c.write_pixel(round(p.position.x), c.height - round(p.position.y), Color(1, 1, 1))
            e.tick(p)
        c.ppm_file("out.ppm")

    def test_mat_mult(self):
        m1 = Matrix([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 8, 7, 6],
                     [5, 4, 3, 2]])
        m2 = Matrix([[-2, 1, 2, 3],
                     [3, 2, 1, -1],
                     [4, 3, 6, 5],
                     [1, 2, 7, 8]])

        self.assertEqual(m1*m2, Matrix([[20, 22, 50, 48], [44, 54, 114, 108], [40, 58, 110, 102], [16, 26, 46, 42]]))

    def test_mat_tuple_mult(self):
        m = Matrix([[1, 2, 3, 4],
        [2, 4, 4, 2],
        [8, 6, 4, 1],
        [0, 0, 0, 1]])
        b = Tuple(1, 2, 3, 1)
        self.assertEqual(m*b, Tuple(18, 24, 33, 1))

    def test_mat_transpose(self):
        self.assertEqual(Matrix([[0, 9, 3, 0],
        [9, 8, 0, 8],
        [1, 8, 5, 3],
        [0, 0, 5, 8]]).transpose(), Matrix([[0, 9, 1, 0], [9, 8, 8, 0], [3, 0, 5, 5], [0, 8, 3, 8]]))

    def test_mat_minor(self):
        m = Matrix([[3, 5, 0], [2, -1, -7], [6, -1, 5]])
        b = m.submatrix(1, 0)
        self.assertEqual(b.determinant(), 25)
        self.assertEqual(m.minor(1, 0), 25)

    def test_mat_determinant(self):
        self.assertEqual(Matrix([[1, 5], [-3, 2]]).determinant(), 17)

    def test_submatrices(self):
        self.assertEqual(Matrix([[1, 5, 0], [-3, 2, 7], [0, 6, -3]]).submatrix(0, 2),
                         Matrix([[-3, 2], [0, 6]]))
        self.assertEqual(Matrix([[-6, 1, 1, 6], [-8, 5, 8, 6], [-1, 0, 8, 2], [-7, 1, -1, 1]]).submatrix(2, 1),
                         Matrix([[-6, 1, 6], [-8, 8, 6], [-7, -1, 1]]))

    def test_larger_mat_determinant(self):
        self.assertEqual(Matrix([[1, 2, 6], [-5, 8, -4], [2, 6, 4]]).determinant(), -196)
        m = Matrix([[-2, -8, 3, 5], [-3, 1, 7, 3], [1, 2, -9, 6], [-6, 7, 7, -9]])
        self.assertEqual(m.determinant(), -4071)
        self.assertEqual(m.cofactor(0, 1), 447)
    
    def test_inversion(self):
        m = Matrix([[8, -5, 9, 2],
                    [7, 5, 6, 1],
                    [-6, 0, 9, 6],
                    [-3, 0, -9, -4]])

        self.assertEqual(m.inverse(), Matrix([[-0.15385, -0.15385, -0.28205, -0.53846],
                                              [-0.07692, 0.12308, 0.02564, 0.03077],
                                              [0.35897, 0.35897, 0.43590, 0.92308],
                                              [-0.69231, -0.69231, -0.76923, -1.92308]]))
        m = Matrix([[9, 3, 0, 9],
             [-5, -2, -6, -3],
             [-4, 9, 6, 4], 
             [-7, 6, 6, 2]])
        self.assertEqual(m.inverse(), Matrix([[-0.04074, -0.07778, 0.14444, -0.22222],
                                              [-0.07778, 0.03333, 0.36667, -0.33333],
                                              [-0.02901, -0.14630, -0.10926, 0.12963],
                                              [0.17778, 0.06667, -0.26667, 0.33333]]))