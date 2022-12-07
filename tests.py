import os

from main import Point, Vector, Tuple, Projectile, Environment, Color, Canvas, Matrix, Ray, Sphere, Intersection, Scene

import unittest
import math
import numpy as np
import numpy.testing as npt
class TupleTests(unittest.TestCase):
    @staticmethod
    def show_image(canvas=None):
        if canvas:
            canvas.ppm_file("out.ppm")
        from PIL import Image
        i = Image.open("out.ppm")
        i.show()

    def test_add(self):
        a = Point(3, -2, 5)
        b = Vector(-2, 3, 1)
        self.assertAlmostEqual(a + b, Tuple(1, 1, 6, 1))

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
        self.assertEqual(Vector(1, 2, 3).normalize(),
                         Vector(0.26726, 0.53452, 0.80178))

    @unittest.skip
    def test_projectiles(self):
        p = Projectile(Point(0, 1, 0), Vector(1, 1, 0).normalize())
        e = Environment(Vector(0, -0.1, 0), Vector(-0.01, 0, 0))
        while p.position.y > 0:
            print(p.position)
            e.tick(p)

    def test_colors(self):
        self.assertEqual(Color(0.9, 0.6, 0.75) +
                         Color(0.7, 0.1, 0.25), Color(1.6, 0.7, 1.0))
        self.assertEqual(Color(0.9, 0.6, 0.75) -
                         Color(0.7, 0.1, 0.25), Color(0.2, 0.5, 0.5))
        self.assertEqual(Color(0.2, 0.3, 0.4) * 2, Color(0.4, 0.6, 0.8))
        self.assertEqual(Color(1, 0.2, 0.4) * Color(0.9,
                         1, 0.1), Color(0.9, 0.2, 0.04))

    def test_canvas(self):
        c = Canvas(10, 20)
        red = Color(1, 0, 0)
        c.write_pixel(2, 3, red)
        self.assertEqual(c[2][3], red)

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
        for i in range(10):
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

    @unittest.skip
    def test_projectile_canvas(self):
        p = Projectile(Point(0, 1, 0), Vector(1, 1.8, 0).normalize() * 11.25)
        e = Environment(Vector(0, -0.1, 0), Vector(-0.01, 0, 0))
        c = Canvas(900, 550)
        while p.position.y > 0:
            c.write_pixel(round(p.position.x), c.height -
                          round(p.position.y), Color(1, 1, 1))
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

        self.assertEqual(
            m1*m2, Matrix([[20, 22, 50, 48], [44, 54, 114, 108], [40, 58, 110, 102], [16, 26, 46, 42]]))

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
        self.assertAlmostEqual(b.determinant(), 25)
        self.assertAlmostEqual(m.minor(1, 0), 25)

    def test_mat_determinant(self):
        self.assertAlmostEqual(Matrix([[1, 5], [-3, 2]]).determinant(), 17)

    def test_submatrices(self):
        self.assertEqual(Matrix([[1, 5, 0], [-3, 2, 7], [0, 6, -3]]).submatrix(0, 2),
                         Matrix([[-3, 2], [0, 6]]))
        self.assertEqual(Matrix([[-6, 1, 1, 6], [-8, 5, 8, 6], [-1, 0, 8, 2], [-7, 1, -1, 1]]).submatrix(2, 1),
                         Matrix([[-6, 1, 6], [-8, 8, 6], [-7, -1, 1]]))

    def test_larger_mat_determinant(self):
        self.assertAlmostEqual(
            Matrix([[1, 2, 6], [-5, 8, -4], [2, 6, 4]]).determinant(), -196)
        m = Matrix([[-2, -8, 3, 5], [-3, 1, 7, 3],
                   [1, 2, -9, 6], [-6, 7, 7, -9]])
        self.assertAlmostEqual(m.determinant(), -4071)
        self.assertAlmostEqual(m.cofactor(0, 1), 447)

    def test_inversion(self):
        m = Matrix([[8, -5, 9, 2],
                    [7, 5, 6, 1],
                    [-6, 0, 9, 6],
                    [-3, 0, -9, -4]])

        self.assertEqual(m.inv, Matrix([[-0.15385, -0.15385, -0.28205, -0.53846],
                                              [-0.07692, 0.12308,
                                                  0.02564, 0.03077],
                                              [0.35897, 0.35897, 0.43590, 0.92308],
                                              [-0.69231, -0.69231, -0.76923, -1.92308]]))
        m = Matrix([[9, 3, 0, 9],
                    [-5, -2, -6, -3],
                    [-4, 9, 6, 4],
                    [-7, 6, 6, 2]])
        self.assertEqual(m.inverse(), Matrix([[-0.04074, -0.07778, 0.14444, -0.22222],
                                              [-0.07778, 0.03333,
                                                  0.36667, -0.33333],
                                              [-0.02901, -0.14630, -
                                                  0.10926, 0.12963],
                                              [0.17778, 0.06667, -0.26667, 0.33333]]))

    def test_transforms(self):
        m = Matrix([[9, 3, 0, 9],
                    [-5, -2, -6, -3],
                    [-4, 9, 6, 4],
                    [-7, 6, 6, 2]])
        transform = Matrix.translating(1, 2, 3)
        self.assertEqual(m, m*transform*transform.inverse())
        transform = Matrix.scaling(2, 3, 4)
        self.assertEqual(m, m * transform * transform.inverse())
        v = Vector(-4, 6, 8)
        self.assertEqual(v*transform, Vector(-8, 18, 32))

    def test_rotation(self):
        for axis in [0, 1, 2]:
            point = Point(1, 1, 1)
            print(Matrix.rotating(math.pi))
            self.assertEqual(Matrix.rotating(math.pi, axis=axis)
                             * Matrix.rotating(math.pi, axis=axis)*point, point)

    def test_shearing(self):
        start = [1, 0, 0, 0, 0, 0]
        combos = [("x", "y"), ("x", "z"), ("y", "x"),
                  ("y", "z"), ("z", "x"), ("z", "y")]
        while start[-1] == 0:
            p = Point(1, 2, 3)
            cur_comb = combos.pop(0)
            kwargs = {"x":1, "y":2, "z":3}
            kwargs.update({cur_comb[0]: getattr(
                p, cur_comb[0])+getattr(p, cur_comb[1])})
            p = Point(**kwargs)
            self.assertEqual(Matrix.shearing(*start)*Point(1, 2, 3), p)
            start = np.roll(start, shift=1, axis=0)

    def test_clock_face(self):
        hand = Vector(0, 1, 0).normalize()
        hands = 0
        center_point = (20, 20)
        c = Canvas(40, 40)
        white = Color(1, 1, 1)
        while hands < 12:
            c.write_pixel(np.round(center_point[0]+hand.x*(c.width*2/4)),
                          np.round(center_point[1]+hand.y*(c.height*2/4)), white)
            hand = (Matrix.rotating(math.pi/6, axis=2)*hand).normalize()
            hands = hands + 1
        c.ppm_file("out.ppm")
        # self.show_image()

    def test_ray_dist(self):
        r = Ray(Point(2, 3, 4), Vector(1, 0, 0))
        self.assertEqual(r.position(0), Point(2, 3, 4))
        self.assertEqual(r.position(1), Point(3, 3, 4))
        self.assertEqual(r.position(-1), Point(1, 3, 4))
        self.assertEqual(r.position(2.5), Point(4.5, 3, 4))

    def test_ray_sphere_intersect(self):
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s = Sphere(20)
        self.assertEqual(s & r, (4.0, 6.0))
        r = Ray(Point(0, 0, 5), Vector(0, 0, 1))
        self.assertEqual(s & r, (-6.0, -4.0))
        r = Ray(Point(0, 2, -5), Vector(0, 0, 1))
        self.assertEqual(s & r, None)
        r = Ray(Point(0, 1, -5), Vector(0, 0, 1))
        self.assertEqual(s & r, (5.0, 5.0))
        r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
        self.assertEqual(s & r, (-1.0, 1.0))

    def test_hit(self):
        s = Sphere(20)
        i1 = Intersection(1, s)
        i2 = Intersection(2, s)
        self.assertEqual(Intersection.hit(i1 + i2), i1)
        i1 = Intersection(-1, s)
        i2 = Intersection(1, s)
        self.assertEqual(Intersection.hit(i1 + i2), i2)
        i1 = Intersection(-2, s)
        i2 = Intersection(-1, s)
        self.assertEqual(Intersection.hit(i1 + i2), None)
        i1 = Intersection(5, s)
        i2 = Intersection(7, s)
        i3 = Intersection(-3, s)
        i4 = Intersection(2, s)
        self.assertEqual(Intersection.hit(i1 + i2 + i3 + i4), i4)

    def test_ray_transforms(self):
        r = Ray(Point(1, 2, 3), Vector(0, 1, 0))
        m = Matrix.translating(3, 4, 5)
        r2 = r.transform(m)
        self.assertEqual(r2.origin, Point(4, 6, 8))
        self.assertEqual(r2.direction, Vector(0, 1, 0))
        r = Ray(Point(1, 2, 3), Vector(0, 1, 0))
        m = Matrix.scaling(2, 3, 4)
        r2 = r.transform(m)
        self.assertEqual(r2.origin, Point(2, 6, 12))
        self.assertEqual(r2.direction, Vector(0, 3, 0))

    def test_transform_sphere_intersect(self):
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s = Sphere(240, transform=Matrix.translating(5, 0, 0))
        self.assertEqual(r & s, None)
        s.transform = Matrix.scaling(2, 2, 2)
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        npt.assert_almost_equal(s & r, (3, 7))

    @unittest.skipUnless(os.environ.get("TEST_RAY"), "this is too slow")
    def test_raycast_sphere(self):
        camera = Point(0, 0, -5)
        background = (10, 10)
        canvas = Canvas(200, 200)
        shape = Sphere(5)
        self.show_image(Scene(camera, background, canvas, [shape]).trace())
