from dataclasses import dataclass
from math import exp
import numpy as np

from utils import State, CloudStates, get_velocity, get_mean_point
from db_interaction import RespositoryDB


@dataclass
class Prediction():
    frame: int
    illuminance: float
    probability: float
    state: CloudStates

    def get_data(self):
        return f'{self.frame}, {self.illuminance}, {self.probability}, {self.state}\n'


PREDICTIONS = list()


def get_illuminance(
        frame: np.array,
        left_border: int,
        right_border: int,
        top_border: int,
        bot_border: int,
) -> float:
    # BGR frame

    RED_K = 0.299
    BLUE_K = 0.114
    GREEN_K = 0.587

    sum_illuminance = 0
    for i in range(top_border, bot_border):
        for j in range(left_border, right_border):
            sum_illuminance += frame[i][j][0] * BLUE_K + frame[i][j][1] * GREEN_K + frame[i][j][2] * RED_K

    return sum_illuminance


def predict(
        frame: np.array,
        is_border: bool,
        move_state: State,
        cloud_state: CloudStates,
        sun_point: tuple,
        good_old: np.array,
        good_new: np.array,
        line: int,
        points: np.array,
        number_of_frame: int,
        db: RespositoryDB,
):
    if not is_border:
        return

    # We have determined window size as 60 x 60 pixels
    left_border = line - 30
    right_border = line + 30
    top_border = sun_point[1] - 30
    bot_border = sun_point[1] + 30

    illuminance = get_illuminance(frame, left_border, right_border, top_border, bot_border)

    mean_point = get_mean_point(
        left_border,
        right_border,
        top_border,
        bot_border,
        points,
        move_state,
        sun_point,
    )
    distance = sun_point[0] - line
    velocity = get_velocity(good_old, good_new)
    time = int(distance / velocity)

    probability = exp(-(distance / sun_point[0]) ** 2)

    predict_frame = number_of_frame + time

    PREDICTIONS.append(Prediction(predict_frame, illuminance, probability, cloud_state))

    db.add(predict_frame, illuminance, probability, str(cloud_state))

    return mean_point