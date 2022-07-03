from collections import defaultdict
import numpy as np

from utils import State, CloudStates
from dnn import Dnn


def define_state(state_name: str) -> CloudStates:
    if state_name == CloudStates.OPENING.name:
        return CloudStates.OPENING
    if state_name == CloudStates.CLOSING.name:
        return CloudStates.CLOSING


class ScanLine():
    """
    This entity determines is there gap
    """
    def __init__(self, frame_shape: tuple, width: int, sun_pos: tuple, dnn_model: Dnn):
        self._sun_pos = sun_pos
        self._width = width
        self._x = frame_shape[1]
        self._y = frame_shape[0]
        self._dnn_model = dnn_model
        self._cur_state = None
        self._prev_state = None


    def scanning(self, frame: np.array, state: State):
        frame = self._dnn_model.get_transformed_binary_map(frame, 0.75)

        # Only for left or right movement

        if state is State.RIGHT:
             x_line = self._x / 4
             left_limit = int(x_line - self._width / 2)
             right_limit = int(x_line + self._width / 2)
             is_border, state = self._cloud_borders_lr(frame, left_limit, right_limit, state)
             return (int(x_line), is_border, state)
        elif state is State.LEFT:
            x_line = 3 * self._x / 4
            left_limit = int(x_line - self._width / 2)
            right_limit = int(x_line + self._width / 2)
            is_border, state = self._cloud_borders_lr(frame, left_limit, right_limit, state)
            return (int(x_line), is_border, state)


    def _cloud_borders_lr(
            self,
            frame: np.array,
            left_limit: int,
            right_limit: int,
            state: State,
        ):
        STATES = defaultdict(int)
        up_limit = int(self._sun_pos[1]) - 30
        down_limit = int(self._sun_pos[1]) + 30
        is_good_choose = False
        current_state = CloudStates.NOT_CHANGED

        frame[self._sun_pos[1]][self._sun_pos[0]] = 0

        for i in range(up_limit, down_limit + 1):
            for j in range(left_limit + 1, right_limit + 1):
                if int(frame[i][j]) != int(frame[i][j - 1]):
                    if frame[i][j] > frame[i][j - 1] and state is State.RIGHT:
                        STATES['OPENING'] += 1
                    elif frame[i][j] > frame[i][j - 1] and state is State.LEFT:
                        STATES['CLOSING'] += 1
                    elif frame[i][j] < frame[i][j - 1] and state is State.RIGHT:
                        STATES['CLOSING'] += 1
                    elif frame[i][j] < frame[i][j - 1] and state is State.LEFT:
                        STATES['OPENING'] += 1

        if STATES.items():
            max_state = max(STATES, key=STATES.get)
            current_state = define_state(max_state)


        if current_state == CloudStates.CLOSING and STATES['CLOSING'] > 2 * STATES['OPENING']:
            is_good_choose = True
        elif current_state == CloudStates.OPENING and STATES['OPENING'] > 2 * STATES['CLOSING']:
            is_good_choose = True
        return (is_good_choose, current_state)
