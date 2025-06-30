import numpy as np


class Point:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def xy(self):
        return [self.x, self.y]

    def xyz(self):
        return [self.x, self.y, self.z]

    def distance_xy(self, point):
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)

    def distance_xyz(self, point):
        """Computes the Euclidean distance from another point, returning a float."""
        return np.sqrt(
            (self.x - point.x) ** 2 + (self.y - point.y) ** 2 + (self.z - point.z) ** 2
        )

    def midpoint(self, point):
        """Computes the coordinates of the midpoint of the segment connecting the object with another point."""
        return Point(
            0.5 * (self.x + point.x), 0.5 * (self.y + point.y), 0.5 * (self.z + point.z)
        )

    def dist_from_line_xy(self, line_xy) -> float:
        """Computes the distance from a line in the xy plane (defined as a Line_xy object)."""
        dist = np.abs(line_xy.a * self.x + line_xy.b * self.y + line_xy.c) / np.sqrt(
            line_xy.a**2 + line_xy.b**2
        )
        return dist


class Vector:
    def __init__(self, xyz):
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]

    def xy(self):
        return [self.x, self.y]

    def xyz(self):
        return [self.x, self.y, self.z]

    def dot_product_xy(self, vector):
        """Computes dot product in the xy plane"""
        return np.dot(self.xy(), vector.xy())

    def dot_product(self, vector):
        return np.dot(self.xyz(), vector.xyz())

    def cross_product(self, vector):
        return Vector(np.cross(self.xyz(), vector.xyz()))


def diff_vector(point1, point2):
    """Creates a vector that connects two points in the 3D space,
    returning a Vector object. The vector is oriented from point2 towards point1"""
    if point1.z is None or point2.z is None:
        return Vector([point1.x - point2.x, point1.y - point2.y, None])
    else:
        return Vector([point1.x - point2.x, point1.y - point2.y, point1.z - point2.z])


class Line_xy:
    """Line in the xy 2D plane, defined by the equation ax+by+c=0."""

    def __init__(self, point1: Point, point2: Point) -> None:
        m = (point2.y - point1.y) / (point2.x - point1.x)
        q = point1.y - m * point1.x
        self.a = -m
        self.b = 1.0
        self.c = -q

    def intercept(self, line_xy) -> Point:
        """Compute the coordinates of the interception with another line,
        returning a Point object.
        The intercept is found by solving the system AX=B, where A is a matrix
        containing the a and b coefficients of the lines, X contains the coordinates of the
        intercept and B contains the c coefficients of the lines, with a minus sign."""
        A = np.array([[self.a, self.b], [line_xy.a, line_xy.b]])
        B = np.array([-self.c, -line_xy.c])
        X = np.dot(np.linalg.inv(A), B)
        return Point(X[0], X[1], None)


class Plane:
    """Plane defined by means of the equation
    a(x-x_p)+b(y-y_p)+c(z-z_p)=0
    where (x_p, y_p, z_p) are the coordinates of a point belonging
    to the plane, while (a,b,c) are the coordinates of a vector that
    is normal to the Plane.
    """

    def __init__(self, point1: Point, point2: Point, point3: Point):
        self.point = point1
        u = diff_vector(point1, point2)
        v = diff_vector(point1, point3)
        self.vect = u.cross_product(v)

    def contains_point(self, point, eps=1e-6):
        """Checks if a point is in the plane, returning a boolean"""
        u = diff_vector(self.point, point)
        dist = abs(u.dot_product(self.vect))
        if dist < eps:
            return dist, True
        else:
            return dist, False

    def compute_z(self, x, y):
        """Computes the elevation of a point on the plane, given its x and y coordinates"""
        return (
            -1
            / self.vect.z
            * (self.vect.x * (x - self.point.x) + self.vect.y * (y - self.point.y))
            + self.point.z
        )
