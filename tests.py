from main import Point, Vector, Tuple, Projectile, Environment, Color, Canvas
import unittest
import math


class TupleTests(unittest.TestCase):
    def test_add(self):
        a = Point(3, -2, 5)
        b = Vector(-2, 3, 1)
        self.assertEqual(a+b, Tuple(1, 1, 6, 1))

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
        self.assertEqual(Color(0.9, 0.6, 0.75)+Color(0.7, 0.1, 0.25), Color(1.6, 0.7, 1.0))
        self.assertEqual(Color(0.9, 0.6, 0.75) - Color(0.7, 0.1, 0.25), Color(0.2, 0.5, 0.5))
        self.assertEqual(Color(0.2, 0.3, 0.4) * 2, Color(0.4, 0.6, 0.8))
        self.assertEqual(Color(1, 0.2, 0.4)*Color(0.9, 1, 0.1), Color(0.9, 0.2, 0.04))

    def test_canvas(self):
        c = Canvas(10, 20)
        red = Color(1, 0, 0)
        c.write_pixel(2, 3, red)
        self.assertEqual(c[2*20+3], red)

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
        p = Projectile(Point(0, 1, 0), Vector(1, 1.8, 0).normalize()*11.25)
        e = Environment(Vector(0, -0.1, 0), Vector(-0.01, 0, 0))
        c = Canvas(900, 550)
        while p.position.y > 0:
            c.write_pixel(round(p.position.x), c.height-round(p.position.y), Color(1, 1, 1))
            e.tick(p)
        c.ppm_file("out.ppm")



