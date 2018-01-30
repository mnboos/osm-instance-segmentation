from PIL import Image, ImageOps
import numpy as np

# numpy index
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

    def __init__(self, img_path):
        img = Image.open(img_path).convert("L")
        img = ImageOps.expand(img, self.BORDER_SIZE)
        np_arr = np.asarray(img)
        self.img = np_arr
        self._states = np.zeros(np_arr.shape, dtype=np.uint8)
        self._start = None

    def find_contour(self):
        self._calc_cell_states()
        points = [self._start]
        if self._start:
            current_pos = None
            while current_pos != self._start:
                if not current_pos:
                    current_pos = self._start

                state = self._states[current_pos]
                direction = self._get_next_direction(state)

                current_pos = tuple(map(sum, zip(current_pos, direction)))
                points.append(current_pos)
        return points

    @staticmethod
    def _get_next_direction(state):
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

    def _calc_cell_states(self):
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
    def _binarize(val):
        return 1 if val > 0 else 0
