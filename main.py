import weakref
from math import sqrt, cos, sin
from uuid import uuid4
from enum import Enum
from functools import cached_property, cache, partial
import math
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock
from collections.abc import Iterable
EPSILON = .0001
import numpy as np

def clamper(x, l, u=None):
    return l if x < l else u if x > u else x

class Tuple:
    __slots__ = ("x", "y", "z", "w", "m", "mat")

    def __init__(self, x=0, y=0, z=0, w=0, mat=None):
        if mat is None:
            mat = x, y, z, w
        self.mat = mat
        self.x, self.y, self.z, self.w = mat
        self.mat = np.asarray([x, y, z, w], dtype="float")
        self.m = np.linalg.norm(self.mat)

    def __iter__(self):
        return self.mat

    def __hash__(self):
        return hash(self.mat.tobytes())

    def __eq__(self, other):
        return all((math.isclose(i, j, abs_tol=EPSILON) for i, j in zip(self.mat.flatten(), other.mat.flatten())))

    def __add__(self, other):
        if not isinstance(other, Tuple):
            return type(self)(*[self.mat+other])
        return self.type_from_op(other)(*self.mat+other.mat)

    def __radd__(self, other):
        return self.__add__(other)

    @cache
    def __sub__(self, other):
        if not isinstance(other, Tuple):
            return type(self)(*[self.mat - other])
        return self.type_from_op(other)(*self.mat - other.mat)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __neg__(self):
        return Tuple(-self.x, -self.y, -self.z, -self.w)

    @cache
    def __mul__(self, other):
        if isinstance(other, Matrix):
            return Matrix.__mul__(other, self)
        elif isinstance(other, Color):
            return Color(*(i*j for i, j in zip(self, other)))
        elif not isinstance(other, Vector):
            return type(self)(*(i*other for i in self))
        else:
            return self.cross(other)

    @cache
    def __truediv__(self, other, inverse=False):
        if not isinstance(other, Tuple):
            return type(self)(*self.mat/other)
        else:
            raise TypeError

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __repr__(self):
        return f"{type(self).__name__}({self.x}, {self.y}, {self.z}, {self.w})"

    def __abs__(self):
        return self.m

    def __iter__(self):
        return iter(self.mat)

    @cache
    def normalize(self):
        if self.m == 0:
            return self
        return type(self)(*self.mat/self.m)

    @cache
    def dot(self, other):
        return np.dot(self.mat, other.mat)

    @cache
    def type_from_op(self, other):
        if type(self) == Color:
            return Color

        new_w = self.w + other.w
        if new_w == 1:
            return Point
        elif new_w == 0:
            return Vector
        else:
            return Tuple

    @cache
    def invert(self):
        return type(self)(*1/self.mat)

    def reflect(self, normal):
        return type(self)(*(self.mat - normal.mat * 2 * self.dot(normal)))


class Point(Tuple):
    w = 1
    def __init__(self, x=0, y=0, z=0, w=1):
        super().__init__(x, y, z, 1)


class Vector(Tuple):
    w = 0
    def __init__(self, x=0, y=0, z=0, w=0):
        super().__init__(x, y, z, 0)

    @cache
    def cross(self, other):
        return Vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)


class Projectile:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity


class Environment:
    def __init__(self, gravity, wind):
        self.gravity = gravity
        self.wind = wind

    def tick(self, projectile):
        projectile.position = projectile.position + projectile.velocity
        projectile.velocity = projectile.velocity+self.gravity+self.wind


class Color(Tuple):
    def __init__(self, r, g, b, w=0):
        super().__init__(r, g, b, w)

    @cache
    def __mul__(self, other):
        if not isinstance(other, Color):
            return Color(*(i*other for i in self.mat))
        return Color(self.x*other.x, self.y*other.y, self.z*other.z)

    def clamp(self):
        return Color(*(round(clamper(i, 0, 255)) for i in self))

    def __str__(self):
        clamped = (self*255).clamp()
        return f"{clamped.x} {clamped.y} {clamped.z}"

    def __repr__(self):
        return str(self)


class Canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canv_arr = np.empty((self.width, self.height), dtype="object")
        self.canv_arr.fill(Color(0, 0, 0))

    def write_pixel(self, x, y, color):
        try:
            self.canv_arr[int(x)][int(y)] = color
        except:
            pass

    def __getitem__(self, item):
        return self.canv_arr[item]

    def __setitem__(self, key, value):
        if key > len(self.canv_arr):
            return
        self.canv_arr[key] = value

    def ppm_header(self):
        return f"""P3
{self.width} {self.height}
255
"""

    def chunk_canvas(self):
        string = ' '.join((str(i) for i in self.canv_arr.flatten()))

        def chunker(s):
            chunks = []
            chunk = ""
            idx = 0
            for i in s:
                chunk += i
                idx = idx + 1
                if i.isspace() and idx >= 70:
                    chunks.append(chunk+"\n")
                    chunk = ""
                    idx = 0
            if chunk:
                chunks.append(chunk)
            return chunks
        return "".join(chunker(string))

    def ppm_body(self):
        return self.chunk_canvas()

    def ppm_file(self, path=None):
        if path:
            with open(path, "w+") as file:
                print(self.ppm_header()+self.ppm_body(), file=file)
        else:
            return self.ppm_header()+self.ppm_body()

class TransformType(Enum):
    SHEARING = "shearing"
    TRANSLATING = "translating"
    SCALING = "scaling"
    ROTATING = "rotating"
    IDENTITY = "identity"
    UNKNOWN = None

class Matrix:
    def __init__(self, mat, transform_type=TransformType.UNKNOWN):
        self.transform_type = transform_type
        self.size = (len(mat), len(mat[0])) if isinstance(mat, list) else mat.shape
        if not isinstance(mat, np.ndarray):
            self.mat = np.asarray(mat, dtype="float").reshape(self.size)
        else:
            self.mat = mat

    def __hash__(self):
        return hash(self.mat.tobytes()) + hash(self.transform_type)

    def __eq__(self, other):
        return all((math.isclose(i, j, abs_tol=EPSILON) for i, j in zip(self.mat.flatten(), other.mat.flatten())))

    def __getitem__(self, item):
        return self.mat[item]

    def __setitem__(self, key, value):
        self.mat[key] = value

    def __round__(self, precision):
        return np.round(self.mat, decimals=precision)

    def __iter__(self):
        return iter(self.mat)

    def __abs__(self):
        return Matrix(np.abs(self.mat))

    def pop(self, idx):
        return Matrix([sublist for i, sublist in enumerate(self.mat) if i != idx])

    @cache
    def __mul__(self, other):
        if isinstance(other, Tuple):
            return type(other)(*[sum([i*n for i, n in zip(row, other)]) for row in self.mat])
        if isinstance(other, Matrix):
            return Matrix([[sum([i*n for i, n in zip(row, col)]) for col in other.transpose().mat] for row in self.mat])
        return Matrix([[i*other for i in row] for row in self.mat])

    def transpose(self):
        return Matrix(self.mat.transpose())

    @cache
    def determinant(self):
        return np.linalg.det(self.mat)


    @cache
    def submatrix(self, row, column):
        return Matrix(self.pop(row).transpose().pop(column).transpose().mat)

    @cache
    def minor(self, row, column):
        return self.submatrix(row, column).determinant()

    @cache
    def cofactor(self, row, column):
        return self.minor(row, column)*(1 if (row+column) % 2 == 0 else -1)

    @cache
    def invertible(self):
        return not (self.determinant() == 0)

    @cached_property
    def inv(self):
        return self.inverse()
    @cache
    def inverse(self):
        return Matrix(np.linalg.inv(self.mat))

    @staticmethod
    def translating(x, y, z):
        return Matrix([[1, 0, 0, x],
                       [0, 1, 0, y],
                       [0, 0, 1, z],
                       [0, 0, 0, 1]], transform_type=TransformType.TRANSLATING)

    @staticmethod
    def scaling(x, y, z):
        return Matrix([[x, 0, 0, 0],
                       [0, y, 0, 0],
                       [0, 0, z, 0],
                       [0, 0, 0, 1]], transform_type=TransformType.SCALING)

    @staticmethod
    def rotating(r, axis=0):
        if axis == 0:
            return Matrix([[1,      0,       0, 0],
                           [0, cos(r), -sin(r), 0],
                           [0, sin(r),  cos(r), 0],
                           [0,      0,       0, 1]], transform_type=TransformType.ROTATING)
        elif axis == 1:
            return Matrix([[cos(r),  0, sin(r), 0],
                           [0,       1,      0, 0],
                           [-sin(r), 0, cos(r), 0],
                           [0,       0,      0, 1]], transform_type=TransformType.ROTATING)
        elif axis == 2:
            return Matrix([[cos(r), -sin(r), 0, 0],
                           [sin(r),  cos(r), 0, 0],
                           [0,            0, 1, 0],
                           [0,            0, 0, 1]], transform_type=TransformType.ROTATING)
        raise TypeError("Invalid axis!")

    @staticmethod
    def shearing(xy, xz, yx, yz, zx, zy):
        return Matrix([[1,  xy, xz, 0],
                       [yx,  1, yz, 0],
                       [zx, zy,  1, 0],
                       [0,   0,  0, 1]], transform_type=TransformType.SHEARING)

    @staticmethod
    def eye():
        return Matrix([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], transform_type=TransformType.IDENTITY)


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def position(self, d):
        return self.origin + self.direction*d

    def transform(self, m):
        return Ray(self.origin*m, self.direction*m)

    def __repr__(self):
        return f"origin: {self.origin}, direction: {self.direction}"


class Light:
    def __init__(self, position, intensity):
        self.intensity = intensity
        self.position = position


class Material:
    def __init__(self, color=Color(0, 1, 0), ambient=.1, diffuse=.9, specular=.9, shininess=200):
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess


class Sphere:
    def __init__(self, radius, transform=Matrix.eye(), material=Material(color=Color(1, 0, 0))):
        self.id = uuid4()
        self.radius = radius
        self.transform = transform
        self.material = material
        self._hash = self.hash()

    def hash(self):
        return sum([hash(self.id), hash(self.radius), hash(self.transform), hash(self.material.color)])

    def __hash__(self):
        return self._hash

    @cache
    def __and__(self, other):
        if not isinstance(other, Ray):
            return None
        other = other.transform(self.transform.inverse())
        sphere_to_ray = other.origin - Point(0, 0, 0)
        a = other.direction.dot(other.direction)
        b = 2*other.direction.dot(sphere_to_ray)
        c = sphere_to_ray.dot(sphere_to_ray) - 1
        disc = b**2 - (4 * a * c)
        two_a = 2*a
        return None if disc < 0 else ((-b-sqrt(disc))/two_a, (-b+sqrt(disc))/two_a)

    def __rand__(self, other):
        return self.__and__(other)

    def normal(self, point):
        obj_normal =(self.transform.inverse()*point)-Point(0, 0, 0)
        return Vector(*obj_normal*self.transform.submatrix(3, 3).inverse().transpose()).normalize()


class Intersection:
    def __init__(self, t, object, point=None, normal=None, eye=None):
        self.obj = object
        self.t = t
        self.point = point
        self.normal = normal
        self.eye = eye

    def __add__(self, other):
        if isinstance(other, list):
            return other.append(self)
        return [self, other]

    def __radd__(self, other):
        return self.__add__(other)

    @staticmethod
    def hit(l):
        return min((i for i in l if i is not None and i >= 0), default=None)

    def __eq__(self, other):
        if isinstance(other, int):
            return self.t == other
        else:
            return self.t == other.t and self.obj == other.obj

    def __le__(self, other):
        if isinstance(other, int):
            return self.t <= other
        else:
            return self.t <= other.t

    def __ge__(self, other):
        return not self.__lt__(other)

    def __lt__(self, other):
        if isinstance(other, int):
            return self.t < other
        else:
            return self.t < other.t

    def __gt__(self, other):
        return not self.__le__(other) and not self.__eq__(other)


class Scene:
    def __init__(self, camera, background, canvas, shapes, lights):
        self.camera = camera
        self.background = background
        self.canvas = canvas
        self.shapes = shapes
        self.lights = lights
        self.half = background[1] / 2
        self.pixel_size_x = background[0]/self.canvas.width
        self.pixel_size_y = background[1]/self.canvas.height
        self.lock = Lock()

    @staticmethod
    def lighting(object, light):
        material, light, position, eye, normal = object.obj.material, light, object.point, object.eye, object.normal
        effective_color = material.color*light.intensity
        # combine the surface color with the light's color/intensity effective_color ← material.color * light.intensity
        # find the direction to the light source
        lightv = (light.position - position).normalize()  # compute the ambient contribution
        ambient = effective_color * material.ambient
        # light_dot_normal represents the cosine of the angle between the # light vector and the normal vector. A negative number means the # light is on the other side of the surface.
        l_norm = lightv.dot(normal)
        if l_norm < 0:
            diffuse = Color(0, 0, 0)
            specular = Color(0, 0, 0)
        else:
            # compute the diffuse contribution
            diffuse = effective_color * material.diffuse * l_norm
            # reflect_dot_eye represents the cosine of the angle between the
            # reflection vector and the eye vector. A negative number means the # light reflects away from the eye.
            reflect = (-lightv).reflect(normal)
            reflect_dot_eye = reflect.dot(eye)
            if reflect_dot_eye <= 0:
                specular = Color(0, 0, 0)
            else:
                # compute the specular contribution
                factor = math.pow(reflect_dot_eye, material.shininess)
                specular = light.intensity * material.specular * factor
        # Add the three contributions together to get the final shading
        return (ambient+diffuse+specular)

    @staticmethod
    def test_hit(shapes, camera, point):
        intersections = []
        for shape in shapes:
            r = Ray(camera, (point - camera).normalize())
            if b := shape & r:
                for t in b:
                    point = r.position(t)
                    normal = shape.normal(point)
                    eye = -r.direction
                    intersections.append(Intersection(t, shape, point, normal, eye))
        return Intersection.hit(intersections)

    def write_pixel_safe(self, x, y, lights, fut):
        closest_obj = fut.result()
        if closest_obj is None:
            return
        color = Color(0, 0, 0)
        for light in lights:
            color = color + Scene.lighting(closest_obj, light)
        self.canvas.write_pixel(x, y, color)

    def trace(self):
        with ProcessPoolExecutor(4) as pool:
            for y in range(self.canvas.height):
                world_y = self.half - self.pixel_size_y * y
                for x in range(self.canvas.width):
                    world_x = -self.half + self.pixel_size_x * x
                    point = Point(world_x, world_y, self.background[0])
                    pool.submit(self.test_hit, *(self.shapes, self.camera, point)).add_done_callback(partial(self.write_pixel_safe, x, y, self.lights))
            return self.canvas