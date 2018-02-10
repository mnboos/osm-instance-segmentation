from PIL import Image
from typing import Iterable, Tuple, Collection, List
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

    def __init__(self, start_point, starting_angle: float, use_radians: bool = False):
        """

        :param start_point: The starting point of the sleeve
        :param theta: The starting angle of the sleeve in degree
        :param epsilon: The width of the sleeve
        """
        angle_constraint = 45
        self._angle_constraint = angle_constraint if not use_radians else np.radians(angle_constraint)
        if isinstance(start_point, geometry.Point):
            self._position: geometry.Point = start_point
        else:
            self._position = geometry.Point(start_point)
        self._current_angle = starting_angle if not use_radians else starting_angle
        self._use_radians = use_radians

    def fit_sleeve(self, points: List[Tuple[float, float]]):
        # print("fitting to: {}".format(self.wkt))
        # self.move()
        # print("fitting to: {}".format(self.wkt))
        # return
        # segments = []
        # remaining_points = []
        # remaining_points.extend(points)

        nr_points = len(points)
        i = points.index(self.current_position)
        if i is None:
            raise RuntimeError("Start point '{}' could not be found.".format(self.current_position))

        # nr_points_per_check = 10
        measure_distance = 3  # ignore 3 points between the current position and the start of the selected points
        check_ratio_threshold = 0.85

        segments = []
        a = self._current_angle
        cancel_count = 0
        turn_angle = -90  # 90° clockwise
        nr_points_per_check = 10
        original_angle = self._current_angle
        while True:  # todo: find the corret check to determine the end of the search
            seg = [self.current_position]
            any_change = False
            while self._within_ratio(points[i-nr_points_per_check-measure_distance:i-measure_distance]) >= check_ratio_threshold:
            # while self._within_ratio(points[i-1:i]) >= check_ratio_threshold:
            #     i = (i - nr_points_per_check) % nr_points
                nr_points_per_check += 1
                any_change = True

            if any_change:
                # p = geometry.Point(points[i])
                p = geometry.Point(points[(i-nr_points_per_check) % nr_points])
                self._position = self.center_line.interpolate(self.center_line.project(p))

            wkt = self.wkt(points, points[i-nr_points_per_check-measure_distance:i-measure_distance])
            # we arrived at a turning point
            if seg[0] != self.current_position:
                if turn_angle != -90:
                    self._current_angle = original_angle
                turn_angle = -90
                cancel_count = 0
                # self._position = self.center_line.interpolate(corner_radius)
                seg.append(self.current_position)
                segments.append(seg)
                i -= nr_points_per_check
            else:
                cancel_count += 1
            print("current fit: \n", wkt)

            nr_points_per_check = 10
            if cancel_count >= int(math.fabs(360 / turn_angle)):
                # we did 4 full turns here and found nothing, let's search for next direction
                if turn_angle != -30:
                    original_angle = self._current_angle
                    turn_angle = -30
                else:
                    assert False
            self.move(angle=turn_angle, step_size=0)

    def buffer(self):
        cl = LineString([self._position, (self._position.x+30, self._position.y)])
        b: LineString = rotate(geom=cl, angle=self._current_angle, use_radians=self._use_radians, origin=self._position)
        buff = b.buffer(10, cap_style=2)
        return buff

    def _within_ratio(self, points: List[Tuple[float, float]]):
        if not points:
            return False

        within_count = 0
        buff = self.buffer()

        for p in points:
            if self.within_sector(p):
            # if geometry.Point(p).within(buff):
                within_count += 1
        return within_count / len(points)

    def _align_sleeve(self):
        pass

    def move(self, angle: float = None, step_size: int = 10) -> None:
        """
         * Moves the sleeve in the direction specified direction.
           If no angle is specified, the current angle will remain.
        :param angle: The angle will be added to the current angle
        :param step_size: To calculate the actual step size, this value is multiplied with the sleeve_step
        :return:
        """

        # x, y = self.sleeve_step
        if angle:
            self._current_angle += angle
            if self._use_radians:
                self._current_angle %= np.pi*2 if self._current_angle > 0 else -np.pi*2
            else:
                self._current_angle %= 360 if self._current_angle > 0 else -360
            # self._sleeve = rotate(self._sleeve, angle=angle, origin=self.current_position)
            # x, y = self.sleeve_step
        if step_size:
            theta = self._current_angle if self._use_radians else np.radians(self._current_angle)
            x = np.round(self.current_position[0] + step_size * math.cos(theta), decimals=2)
            y = np.round(self.current_position[1] + step_size * math.sin(theta), decimals=2)
            self._position = geometry.Point((x, y))
            # self.current_position
            # self._sleeve = translate(self._sleeve, xoff=x * step_size, yoff=y * step_size)

    @property
    def current_position(self) -> Tuple[float, float]:
        return self._position.x, self._position.y

    def within_sector(self, p: Tuple[float, float]) -> bool:
        """
         * Checks if p fulfills the angle constraint: -c/2 <= alpha_pi - alpha_s <= c/2
            c:          angle constraint
            alpha_pi:   angle between a horizontal vector and a vector from the current sleeve point to p
            alpha_s:    the current orientation of the sleeve / sector
        :param p:
        :return:
        """
        pos = self.current_position
        v0 = np.array((pos[1], pos[0])) - np.array((p[1], p[0]))
        v1 = np.array([0, 0]) - np.array([0, 1])  # horizontal vector (numpy indexing)
        angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
        deg: float = np.degrees(angle)
        return -self._angle_constraint/2 <= deg-self.current_angle() <= self._angle_constraint/2

    def current_angle(self, in_rad=False):
        if in_rad:
            return self._current_angle if self._use_radians else np.radians(self._current_angle)
        else:
            return self._current_angle if not self._use_radians else np.degree(self._current_angle)

    @property
    def center_line(self) -> LineString:
        ls = LineString([self._position, (self._position.x + 1000, self._position.y)])
        return rotate(geom=ls,
                      angle=self._current_angle,
                      use_radians=self._use_radians,
                      origin=self._position)

    def wkt(self, points: Iterable[Tuple[float, float]], single_points: Iterable[Tuple[float, float]]=None) -> str:
        center = rotate(LineString([self._position, (self._position.x+100, self._position.y)]), angle=self._current_angle, use_radians=self._use_radians, origin=self._position)
        ls_left = rotate(LineString([self._position, (self._position.x+100, self._position.y)]), angle=self._current_angle-self._angle_constraint/2, use_radians=self._use_radians, origin=self._position)
        # ls_right = rotate(LineString([self._position, (self._position.x+10, self._position.y)]), angle=self._current_angle+self._angle_constraint/2, use_radians=self._use_radians, origin=self._position)
        ls_right = rotate(ls_left, angle=self._angle_constraint, use_radians=self._use_radians, origin=self._position)
        p = geometry.Polygon(points)
        single_points = single_points if single_points is None else ",".join(map(lambda p: "POINT({} {})".format(p[0], p[1]), single_points))
        return "GEOMETRYCOLLECTION({},{},{},{},{},{},{})".format(p.wkt, self._position.wkt, ls_left.wkt, ls_right.wkt, center.wkt, single_points,self.buffer().wkt)
        # return "GEOMETRYCOLLECTION({},{},{},{})".format(p.wkt, center.wkt, single_points,self.buffer().wkt)


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
            points = list(map(lambda t: (t[0], t[1]), c.tolist()))

        # ls = LineString()
        # p = geometry.Point()
        # p.di

        self._contour = np.zeros(self.img.shape, dtype=np.uint8)
        for x, y in points:
            self._contour[y, x] = 1

        self._points = points
        return self._points

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

            angle_sum = 0
            counts = 0
            for a in angles:
                angle_sum += a*angles[a]
                counts += angles[a]
            weighted_avg = angle_sum / counts
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


