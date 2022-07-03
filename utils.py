from math import sqrt
from enum import Enum
from collections import defaultdict
import numpy as np
import cv2 as cv


class State(Enum):
    UP = 0  # down to up
    DOWN = 1  # up to down
    LEFT = 2  # right to left
    RIGHT = 3  # left to right


class CloudStates(Enum):
    OPENING = 0
    CLOSING = 1
    NOT_CHANGED = 2


STATES = defaultdict(int)
CNT_UPDATE_FRAMES = 10
VAR_UPDATE_DICT = 0


def get_velocity(good_old: np.array, good_new: np.array) -> float:
    sum_dv = 0
    cnt_points = max(len(good_new), len(good_old))

    for pp1, pp0 in zip(good_new, good_old):
        sum_dv += get_distance(pp0, pp1)

    velocity = sum_dv / cnt_points
    return velocity


def get_mean_point(
        left_border: int,
        right_border: int,
        top_border: int,
        bot_border: int,
        points: np.array,
        state: State,
        sun_point: tuple,
) -> tuple:
    valid_points = []
    for point in points:
        x, y = point.ravel()
        if left_border <= x <= right_border and top_border <= y <= bot_border:
            valid_points.append(point)
    return (mean_coord(valid_points, state), sun_point[1])


def mean_coord(points: np.array, state: State) -> float:
    if state.name == 'UP' or state.name == 'DOWN':
        y_max, y_min = -1e9, 1e9
        for point in points:
            y_max = max(y_max, point[1])
            y_min = min(y_min, point[1])
        return (y_max + y_min) / 2
    x_max, x_min = -1e9, 1e9
    for point in points:
        x_max = max(x_max, point[0])
        x_min = min(x_min, point[0])
    return (x_max + x_min) / 2


def get_distance(point1: tuple, point2: tuple) -> float:
    delta_x = point1[0] - point2[0]
    delta_y = point1[1] - point2[1]
    return sqrt(delta_x ** 2 + delta_y ** 2)


def define_state(state_name: str) -> State:
    if state_name == State.UP.name:
        return State.UP
    if state_name == State.DOWN.name:
        return State.DOWN
    if state_name == State.LEFT.name:
        return State.LEFT
    if state_name == State.RIGHT.name:
        return State.RIGHT


def cmp_deltas(old_array: np.array, new_array: np.array, idx: int) -> bool:
    length = min(len(old_array), len(new_array))
    cnt_old = 0
    cnt_new = 0

    for i in range(length):
        if old_array[i][idx] < new_array[i][idx]:
            cnt_new += 1
        else:
            cnt_old += 1

    return cnt_new > cnt_old


def update_dict():
    global VAR_UPDATE_DICT
    VAR_UPDATE_DICT += 1

    if not VAR_UPDATE_DICT % CNT_UPDATE_FRAMES:
        for key in STATES.keys():
            STATES[key] = 0
        VAR_UPDATE_DICT = 0


def get_state(good_old: np.array, good_new: np.array) -> State:
    cnt_points = max(len(good_new), len(good_old))
    dx_sum = 0
    dy_sum = 0
    update_dict()

    for pp1, pp0 in zip(good_new, good_old):
        dx_sum += abs(pp1[0] - pp0[0])
        dy_sum += abs(pp1[1] - pp0[1])

    dx = dx_sum / cnt_points
    dy = dy_sum / cnt_points

    if dx < dy:  # clouds appear on top or bottom of image
        # up to down OR down to up
        if cmp_deltas(good_old, good_new, 1):
            STATES['DOWN'] += 1
            if STATES['DOWN'] == max(STATES, key=STATES.get):
                return State.DOWN
            else:
                max_state_name = max(STATES, key=STATES.get)
                return define_state(max_state_name)
        else:
            STATES['UP'] += 1
            if STATES['UP'] == max(STATES, key=STATES.get):
                return State.UP
            else:
                max_state_name = max(STATES, key=STATES.get)
                return define_state(max_state_name)
    # left to right OR right to left
    if cmp_deltas(good_old, good_new, 0):
        STATES['RIGHT'] += 1
        if STATES['RIGHT'] == max(STATES, key=STATES.get):
            return State.RIGHT
        else:
            max_state_name = max(STATES, key=STATES.get)
            return define_state(max_state_name)
    else:
        STATES['LEFT'] += 1
        if STATES['LEFT'] == max(STATES, key=STATES.get):
            return State.LEFT
        else:
            max_state_name = max(STATES, key=STATES.get)
            return define_state(max_state_name)


def setup_line(frame: np.array, point1: tuple, point2: tuple):
    frame = cv.line(
        frame,
        point1,
        point2,
        [255, 0, 0],
        1,
    )

    return frame
