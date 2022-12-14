import numpy as np
import mahotas
from detectron2.structures import polygons_to_bitmask


# def render(polygon):
#     xs = [i[0] for i in polygon]
#     ys = [i[1] for i in polygon]
#     minx, maxx = min(xs), max(xs)
#     miny, maxy = min(ys), max(ys)
#     X = maxx - minx + 1
#     Y = maxy - miny + 1
#     newPoly = [(x - minx, y - miny) for (x, y) in polygon]
#
#     grid = np.zeros((X, Y), dtype=np.int8)
#
#     return [(x + minx, y + miny) for (x, y) in np.where(grid)]


def alt_poly_to_mask(polygon, dimensions):
    output = np.zeros(dimensions).astype(bool)
    mahotas.polygon.fill_polygon(polygon, output, color=True)
    return output

    # for a, b in zip(polygon, polygon[1:] + [polygon[0]]):
    #     for x in range(a[0], b[0] + 1):
    #         for y in range(a[1], b[1] + 1):
    #             output[x, y] = True
    #
    # silly_polygon = [x_or_y for coords in polygon for x_or_y in reversed(coords)]
    # fenceposted_mask = polygons_to_bitmask([np.asarray(silly_polygon)], dimensions[0], dimensions[1])
    #
    # return output | fenceposted_mask


if __name__ == '__main__':
    dims = (6, 6)
    poly = [[1, 1], [1, 2], [3, 2], [3, 1]]  # nice, boring rectangle
    expected = np.zeros(dims).astype(bool)
    expected[1:4, 1:3] = True
    assert np.all(alt_poly_to_mask(poly, dims) == expected), "why is my code bad?"
