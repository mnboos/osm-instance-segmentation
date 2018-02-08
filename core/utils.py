from PIL import Image
from typing import Iterable, Tuple
from skimage.measure import approximate_polygon
from skimage.transform import hough_line, hough_line_peaks
from pygeotile.tile import Tile, Point
from shapely.geometry import LineString
from shapely.affinity import rotate, translate
from shapely import geometry
import cv2
import math
import numpy as np

# numpy directions
UP = (-1, 0)
DOWN = (1, 0)
RIGHT = (0, 1)
LEFT = (0, -1)


class SleeveFitting:

    def __init__(self, start: geometry.Point, starting_angle: int, sleeve_width: float = 3, sleeve_length: int = 15):
        """

        :param start: The starting point of the sleeve
        :param theta: The starting angle of the sleeve in degrees
        :param epsilon: The width of the sleeve
        """
        self._start = start
        self._angle = starting_angle
        self._sleeve_width = sleeve_width
        self._sleeve_length = sleeve_length
        # ls = LineString([start, Point(start.x+sleeve_length, start.y)]).buffer(sleeve_width, cap_style=2)
        ls = LineString([start, Point(start.x+sleeve_length, start.y)])
        self._sleeve: geometry.LineString = rotate(geom=ls, angle=starting_angle, origin=start)
        print("sleeve: \n", self._sleeve.wkt)

    def fit_sleeve(self, points: Iterable[Tuple[int, int]]):
        segments = []
        remaining_points = []
        remaining_points.extend(points)
        i = 0
        while remaining_points[i] != self._start:
            i += 1
        while remaining_points:
            seg_points = []


        pass

    def _align_sleeve(self):
        pass

    def move(self, angle: float = None, step_multiplicator: int = 1) -> None:
        """
         * Moves the sleeve in the direction specified direction.
           If no angle is specified, the current angle will remain.
        :param angle:
        :param step_multiplicator: To calculate the actual step size, this value is multiplied with the sleeve_step
        :return:
        """

        x, y = self.sleeve_step
        if angle:
            self._sleeve = rotate(self._sleeve, angle=angle, origin=self.current_position)
            x, y = self.sleeve_step
        if step_multiplicator:
            self._sleeve = translate(self._sleeve, xoff=x * step_multiplicator, yoff=y * step_multiplicator)

    @property
    def current_position(self) -> Tuple[float, float]:
        return self._sleeve.coords[0]

    @property
    def sleeve_step(self) -> Tuple[float, float]:
        x0, y0 = self._sleeve.coords[0]
        x1, y1 = self._sleeve.coords[-1]
        return (x1-x0) / self._sleeve_length, (y1-y0) / self._sleeve_length

    @property
    def _buffer(self):
        return self._sleeve.buffer(self._sleeve_width, cap_style=2)

    @property
    def wkt(self) -> str:
        return self._buffer.wkt


class MarchingSquares:
    """
      Implementation of the marching square algorithm to find contours on images. O
      The current implementation finds only one contour per image (the one top-left, to be exact).
    """

    BORDER_SIZE = 1

    def __init__(self, data: np.ndarray):
        c = data.copy()
        self.img = np.pad(c, pad_width=self.BORDER_SIZE, mode='constant')
        self._contour = np.zeros(self.img.shape, dtype=np.uint8)
        self._states = np.zeros(self.img.shape, dtype=np.uint8)
        self._start = None
        self._marched = False
        self._points = []

    @classmethod
    def from_file(cls, img_path: str):
        img = Image.open(img_path).convert("L")
        np_arr = np.asarray(img)
        return cls(np_arr)

    @classmethod
    def from_array(cls, data: np.ndarray):
        return cls(data)

    def find_contour(self, approximization_tolerance: float = 0.01) -> Iterable[Tuple[int, int]]:
        """
         * Returns the first contour found.
        :param approximization_tolerance: tolerance for the douglas-peucker approximization run on the resulting points
        :return:
        """
        if approximization_tolerance is None:
            approximization_tolerance = 0.01

        self._calc_cell_states()
        points = []
        self._marched = True
        if self._start:
            self._contour[self._sum_tuple(self._start, (1, 1))] = 1
            points.append(self._start[::-1])
            current_pos = None
            while current_pos != self._start:
                if not current_pos:
                    current_pos = self._start

                state = self._states[current_pos]
                direction = self._get_next_direction(state)

                current_pos = self._sum_tuple(current_pos, direction)
                flipped = current_pos[::-1]
                if current_pos != self._start and flipped in points:
                    raise RuntimeError("Invalid contour")
                self._contour[self._sum_tuple(current_pos, (1, 1))] = 255
                points.append(flipped)
        if approximization_tolerance:
            c = approximate_polygon(np.array(points), tolerance=approximization_tolerance)
            points = c.tolist()

        # ls = LineString()
        # p = geometry.Point()
        # p.di

        self._contour = np.zeros(self.img.shape, dtype=np.uint8)
        for x, y in points:
            self._contour[y, x] = 1

        self._points = points
        return points

    @staticmethod
    def _sum_tuple(t1: Tuple, t2: Tuple) -> Tuple:
        return tuple(map(sum, zip(t1, t2)))

    @property
    def exact_contour(self):
        return self._contour

    @property
    def starting_point(self):
        pass

    def main_orientation(self, angle_in_degrees: bool = False) -> Tuple[int, geometry.Point]:
        if not self._marched:
            raise RuntimeError("To get the main orientation, run 'find_contour' first.")

        # longest_line = None
        # max_length = 0
        # for i, p in enumerate(self._points[1:]):
        #     prev_point = self._points[i-1]
        #     li = LineString([p, prev_point])
        #     if li.length > max_length:
        #         max_length = li.length
        #         longest_line = li

        # parr = np.zeros(self.img.shape, dtype=np.uint8)
        # for x, y in longest_line.coords:
        #     parr[(int(y), int(x))] = 1

        # lines2 = cv2.HoughLines(image=self.exact_contour, rho=1, theta=np.pi / 180, threshold=36)

        max_threshold = 2
        lines = None
        while True:
            new_lines = cv2.HoughLines(image=self.exact_contour, rho=1, theta=np.pi / 180, threshold=max_threshold)
            if new_lines is not None:
                lines = new_lines
            else:
                max_threshold -= 1
                break
            max_threshold += 1
        print("max threshold: ", max_threshold)
        # lines = cv2.HoughLines(image=self.exact_contour, rho=1, theta=np.pi / 180, threshold=20)
        angles = {}
        maxcount = 0
        main_angle = None
        lineimg = np.zeros(self.img.shape, dtype=np.uint8)
        nearest_point = None
        if lines is not None:
            for l in lines:  # rho = distance, theta = angle
                for rho, theta in l:
                    if angle_in_degrees:
                        angle = int(math.degrees(theta))
                    else:
                        angle = theta
                    if angle not in angles:
                        angles[angle] = 0
                    newcount = angles[angle] + 1
                    angles[angle] = newcount
                    if newcount > maxcount:
                        maxcount = newcount
                        main_angle = angle

                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * -b)
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * -b)
                    y2 = int(y0 - 1000 * a)
                    if not nearest_point:
                        dist = None
                        ls = LineString([(x1, y1), (x2, y2)])
                        for px, py in self._points:
                            p = geometry.Point(px, py)
                            new_dist = p.distance(ls)
                            if not nearest_point or new_dist < dist:
                                nearest_point = p
                                dist = new_dist
                    cv2.line(lineimg, (x1, y1), (x2, y2), 255, 1)
            print("\na1: ", angles)

            angle_sum = 0
            counts = 0
            for a in angles:
                angle_sum += a*angles[a]
                counts += angles[a]
            weighted_avg = angle_sum / counts
            print("avg: ", weighted_avg)
            im = Image.fromarray(lineimg, mode="L")
            im.save("hough.bmp")
            return int(round(weighted_avg)), nearest_point

        #     # lines = cv2.HoughLines(image=self.exact_contour, rho=1, theta=np.pi/180*main_angle, srn=1, stn=np.pi/180*90, threshold=9)
        #     thetas = [math.radians(main_angle), math.radians(main_angle+90 % 180)]
        #     lines = cv2.HoughLines(image=self.exact_contour,
        #                            rho=1,
        #                            theta=math.radians(main_angle+90 % 180),
        #                            # min_theta=min(thetas),
        #                            # max_theta=max(thetas),
        #                            threshold=4)
        #
        #     angles = {}
        #     if lines is not None:
        #         for l in lines:  # rho = distance, theta = angle
        #             for rho, theta in l:
        #                 if angle_in_degrees:
        #                     angle = int(math.degrees(theta))
        #                 else:
        #                     angle = theta
        #                 if angle not in angles:
        #                     angles[angle] = 0
        #                 angles[angle] = angles[angle] + 1
        #                 a = np.cos(theta)
        #                 b = np.sin(theta)
        #                 x0 = a * rho
        #                 y0 = b * rho
        #                 x1 = int(x0 + 1000 * (-b))
        #                 y1 = int(y0 + 1000 * (a))
        #                 x2 = int(x0 - 1000 * (-b))
        #                 y2 = int(y0 - 1000 * (a))
        #                 # ls = LineString([(x1, y1), (x2, y2)])
        #                 # for p in self._points:
        #                 #     dist = ls.distance(geometry.Point(p))
        #                     # print("dist: ", dist)
        #
        #                 cv2.line(lineimg, (x1, y1), (x2, y2), 127, 1)
        #         print("\na2: ", angles)
        #     im = Image.fromarray(lineimg, mode="L")
        #     im.save("hough.bmp")
        #
        #     # cv2.imwrite('houghlines3.bmp', self.exact_contour)
        #
        #
        # # lines = cv2.HoughLines(image=self.exact_contour, rho=1, theta=np.pi / 180, threshold=0, srn=np.pi / 180 * (90+main_angle), stn=1)
        # # for i, p in enumerate(self._points[1:]):
        # #     prev_p = self._points[i]
        # #     ls = LineString([p, prev_p])
        # #     angle = np.rad2deg(np.arctan2(p[1] - prev_p[1], p[0] - prev_p[0]))
        # #     print(angle)
        #
        # return main_angle

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
            top_left = value > 0
            top_right = self.img[r, c+1] > 0
            bottom_right = self.img[r+1, c+1] > 0
            bottom_left = self.img[r+1, c] > 0
            cell_state = (top_left << 3) | (top_right << 2) | (bottom_right << 1) | bottom_left
            self._states[r, c] = cell_state
            if not self._start and 0 < cell_state < 15:
                self._start = (r, c)


def georeference(points: Iterable[Tuple[int, int]], tile: Tile) -> Iterable[Tuple]:
    zoom_level = len(str(tile.quad_tree))
    minx, maxy = tile.bounds[0].pixels(zoom_level)
    maxx, miny = tile.bounds[1].pixels(zoom_level)
    georeferenced = list(
        map(lambda p: tuple(reversed(Point.from_pixel(p[0]+minx, p[1]+miny, zoom_level).latitude_longitude)), points))
    return georeferenced


