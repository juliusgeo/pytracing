from math import sqrt
from itertools import takewhile
EPSILON = .0001


class Tuple:
    __slots__ = ("x", "y", "z", "w", "m")
    dims = ("x", "y", "z")

    def __init__(self, x, y, z, w=0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.m = sqrt(self.x**2+self.y**2+self.z**2+self.w**2)

    def __eq__(self, other):
        for i in self.dims:
            if abs(getattr(self, i) - getattr(other, i)) > EPSILON:
                return False
        return True

    def __add__(self, other):
        return self.type_from_op(other)(*[getattr(self, i)+getattr(other, i) for i in self.dims])

    def __sub__(self, other):
        return self.type_from_op(other)(*[getattr(self, i) - getattr(other, i) for i in self.dims])

    def __neg__(self):
        return Tuple(-self.x, -self.y, -self.z, -self.w)

    def __mul__(self, other):
        if not isinstance(other, Vector):
            return type(self)(*[getattr(self, i)*other for i in self.dims])
        else:
            return self.cross(other)

    def __truediv__(self, other):
        if not isinstance(other, Tuple):
            return type(self)(*[getattr(self, i) / other for i in self.dims])
        else:
            raise TypeError

    def __repr__(self):
        return f"{type(self).__name__}({self.x}, {self.y}, {self.z})"

    def __abs__(self):
        return self.m

    def normalize(self):
        return type(self)(self.x/abs(self), self.y/abs(self), self.z/abs(self), self.w/abs(self))

    def dot(self, other):
        return sum([getattr(self, i) * getattr(other, i) for i in self.dims])

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


class Point(Tuple):
    w = 1


class Vector(Tuple):
    w = 0

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

    def __mul__(self, other):
        if not isinstance(other, Color):
            return Color(*[getattr(self, i)*other for i in self.dims])
        return Color(self.x*other.x, self.y*other.y, self.z*other.z)

    def clamp(self):
        clamper = lambda x, l, u: l if x < l else u if x > u else x
        return Color(*[round(clamper(getattr(self, i), 0, 255)) for i in self.dims])

    def __str__(self):
        clamped = (self*255).clamp()
        return f"{clamped.x} {clamped.y} {clamped.z}"


class Canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canv_arr = [Color(0, 0, 0)]*((width*height))

    def write_pixel(self, x, y, color):
        self[(y*self.width)+x] = color

    def __getitem__(self, item):
        return self.canv_arr[item]

    def __setitem__(self, key, value):
        self.canv_arr[key] = value

    def ppm_header(self):
        return f"""P3
{self.width} {self.height}
255
"""

    def chunk_canvas(self):
        string = ' '.join([str(i) for i in self.canv_arr])

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






