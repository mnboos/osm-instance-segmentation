from PIL import Image
from typing import Iterable, Tuple
from skimage.measure import approximate_polygon
from pygeotile.tile import Tile, Point
import numpy as np

# numpy directions
UP = (-1, 0)
DOWN = (1, 0)
RIGHT = (0, 1)
LEFT = (0, -1)


class MarchingSquares:
    """
      Implementation of the marching square algorithm to find contours on images. O
      The current implementation finds only one contour per image (the one top-left, to be exact).
    """

    BORDER_SIZE = 1

    def __init__(self, data: np.ndarray):
        c = data.copy()
        self.img = np.pad(c, pad_width=self.BORDER_SIZE, mode='constant')
        self._states = np.zeros(self.img.shape, dtype=np.uint8)
        self._start = None

    @classmethod
    def from_file(cls, img_path: str):
        img = Image.open(img_path).convert("L")
        np_arr = np.asarray(img)
        return cls(np_arr)

    @classmethod
    def from_array(cls, data: np.ndarray):
        return cls(data)

    def find_contour(self, approximization_tolerance: int = 0.01) -> Iterable[Tuple[int, int]]:
        """
         * Returns the first contour found.
        :param approximization_tolerance: tolerance for the douglas-peucker approximization run on the resulting points
        :return:
        """
        if not approximization_tolerance:
            approximization_tolerance = 0.01

        self._calc_cell_states()
        points = []
        if self._start:
            points.append(self._start[::-1])
            current_pos = None
            while current_pos != self._start:
                if not current_pos:
                    current_pos = self._start

                state = self._states[current_pos]
                direction = self._get_next_direction(state)

                current_pos = tuple(map(sum, zip(current_pos, direction)))
                flipped = current_pos[::-1]
                if current_pos != self._start and flipped in points:
                    raise RuntimeError("Invalid contour")
                points.append(flipped)
        if approximization_tolerance:
            c = approximate_polygon(np.array(points), tolerance=approximization_tolerance)
            return c.tolist()
        else:
            return points

    @staticmethod
    def _get_next_direction(state: int) -> Tuple[int, int]:
        """
         * These directions will lead to a clockwise visit of the contour
        :param state:
        :return:
        """
        if state in [1, 5, 9, 13]:
            return DOWN
        elif state in [2, 3, 11]:
            return RIGHT
        elif state in [4, 6, 7, 10]:
            return UP
        elif state in [8, 12, 14]:
            return LEFT
        raise RuntimeError("Illelgal state: {}".format(state))

    def _calc_cell_states(self) -> None:
        for (r, c), value in np.ndenumerate(self.img[:-1, :-1]):
            top_left = self._binarize(value)
            top_right = self._binarize(self.img[r, c+1])
            bottom_right = self._binarize(self.img[r+1, c+1])
            bottom_left = self._binarize(self.img[r+1, c])
            cell_state = (top_left << 3) | (top_right << 2) | (bottom_right << 1) | bottom_left
            self._states[r, c] = cell_state
            if not self._start and 0 < cell_state < 15:
                self._start = (r, c)

    @staticmethod
    def _binarize(val: int) -> int:
        return 1 if val > 0 else 0


def georeference(points: Iterable[Tuple[int, int]], tile: Tile) -> Iterable[Tuple]:
    zoom_level = len(str(tile.quad_tree))
    minx, maxy = tile.bounds[0].pixels(zoom_level)
    maxx, miny = tile.bounds[1].pixels(zoom_level)
    georeferenced = list(
        map(lambda p: tuple(reversed(Point.from_pixel(p[0]+minx, p[1]+miny, zoom_level).latitude_longitude)), points))
    return georeferenced
