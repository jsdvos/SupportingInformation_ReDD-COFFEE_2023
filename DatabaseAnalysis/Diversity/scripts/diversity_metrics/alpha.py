# Code obtained from https://github.com/dwyerk/boundaries/blob/master/concave_hulls.ipynb

from shapely.ops import unary_union, polygonize
from shapely.geometry import MultiLineString, Polygon
from scipy.spatial import Delaunay
from collections import Counter
import numpy as np
import math

def alpha_shape(points, alpha, only_outer = True):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if i < j:
            edges.append((i, j))
        else:
            edges.append((j, i))
    
    coords = np.array([point.coords[0] for point in points])

    tri = Delaunay(coords)
    edges = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        
        # Originally, the radius filter to select the alpha shape edges is
        # defined as circum_r < 1.0/alpha, with
        #       circum_r = a*b*c/(4.0*area)
        # and
        #       area = math.sqrt(s*(s-a)*(s-b)*(s-c)) (Heron's formula)
        #
        # However, the approach to find the area is numerically unstable for
        # triangles with a very small angle when using floating-point arithmetic.
        # The filter is redefined to avoid division by the area, which can
        # become 0 in extreme cases, and to avoid use of the sqrt, as its
        # argument can become negative.
        if (a*b*c*alpha/4)**2 < s*(s-a)*(s-b)*(s-c):
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    
    if only_outer:
        count = Counter(edges)
        edges = [edge for edge in edges if count[edge] == 1]

    m = MultiLineString([coords[[i, j]] for i, j in edges])
    triangles = list(polygonize(m))
    return unary_union(triangles), edges
