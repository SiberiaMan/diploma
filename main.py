import numpy as np
import cv2 as cv
import argparse
from enum import Enum
from typing import TypeVar, Tuple

from dnn import Dnn
from optical_flow import OptFlowSettings
from utils import get_state, State, setup_line
from prediction import predict
from db_interaction import RespositoryDB
from scan_line import ScanLine


PATH_TO_MODEL = (
    'PATH/TO/MODEL'
)
Model = TypeVar('Model')
VideoCapture = TypeVar('VideoCapture')
DB = RespositoryDB('frame', 'illuminance', 'probability', 'state')


def define_state(x: float, y: float, frame_shape: tuple, state: Enum) -> bool:
    if state is State.RIGHT:
        return x > frame_shape[1] / 4
    if state is State.LEFT:
        return x < 3 * frame_shape[1] / 4
    if state is State.DOWN:
        return y > frame_shape[0] / 4
    if state is State.UP:
        return y < 3 * frame_shape[0] / 4


def delete_points(
        frame_shape: tuple,
        good_new: np.array,
        good_old: np.array,
        state: Enum,
) -> Tuple[np.array, np.array]:
    """
    We want to get points only from definite side of frame
    """
    idxs = []
    for i, (new, old) in enumerate(zip(good_new.copy(), good_old.copy())):
        x, y = new.ravel()

        if define_state(x, y, frame_shape, state):
            idxs.append(i)
    return np.delete(good_new, idxs, axis=0), np.delete(good_old, idxs, axis=0)


def optical_flow(cap: VideoCapture, dnn_model: Dnn):
    """
    Set up settings
    """
    POINT_SUN: tuple
    lk_flow = OptFlowSettings()
    lk_params = lk_flow.get_lk_params()
    lk_feature_params = lk_flow.feature_params()
    color_points = np.random.randint(0, 255, (1000, 3))

    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **lk_feature_params)
    mask = np.zeros_like(old_frame)

    # variables
    idx: int = 1
    speed_cnt = 0
    ROUND = 30
    SPEED_BOOST = 6
    POINT_SUN = (
        int(old_frame.shape[1] / 2) + 15,
        int(old_frame.shape[0] / 2) - 45,
    )
    scan_line = ScanLine(old_frame.shape, 10, POINT_SUN, dnn_model)

    # Infinity loop start, break when no frames grabbed
    number_of_frame = 1
    while True:
        ret, frame = cap.read()
        speed_cnt += 1
        if speed_cnt % SPEED_BOOST:
            continue
        number_of_frame += 1
        if not ret:
            print('No frames grabbed!')
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if not idx % ROUND: # Update "strong" points
            p0 = cv.goodFeaturesToTrack(
                frame_gray, mask=None, **lk_feature_params,
            )
            ret, frame = cap.read()
            old_gray = frame_gray
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        p1, st, err = cv.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params,
        )
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            if all([len(good_old), len(good_new)]):
                move_state = get_state(good_old, good_new)
                # frame.shape[1] -- x
                # frame.shape[0] -- y
                good_new, good_old = delete_points(
                    frame.shape, good_new, good_old, move_state,
                )
                # Prediction start
                if all([len(good_old), len(good_new)]) and move_state.name == 'LEFT' or move_state.name == 'RIGHT':
                    line, is_border, cloud_state = scan_line.scanning(frame, move_state)
                    mean_point = predict(
                        frame=frame,
                        is_border=is_border,
                        move_state=move_state,
                        cloud_state=cloud_state,
                        sun_point=POINT_SUN,
                        good_old=good_old,
                        good_new=good_new,
                        line=line,
                        points=good_new,
                        number_of_frame=number_of_frame,
                        db=DB,
                    )
                    if mean_point is not None:
                        frame = cv.circle(
                            frame, (int(mean_point[0]), int(mean_point[1])), 3, color_points[34].tolist(), -1,
                        )
                    frame = setup_line(frame, (line, 0), (line, 100000))
                    frame = setup_line(frame, (int(frame.shape[1] - 1), int(POINT_SUN[1]) - 30), (int(frame.shape[1] - 500), int(POINT_SUN[1]) - 30))
                    frame = setup_line(frame, (int(frame.shape[1] - 1), int(POINT_SUN[1]) + 30), (int(frame.shape[1] - 500), int(POINT_SUN[1]) + 30))
                    cv.imshow('frame', frame)
                    k = cv.waitKey(30) & 0xFF
                    if k == 27:
                        break
                        # Show frame
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(
                mask, (int(a), int(b)), (int(c), int(d)), [0, 0, 0], 2,
            )
            frame = cv.circle(
                frame, (int(a), int(b)), 3, color_points[i].tolist(), -1,
            )
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        idx += 1
    DB.save_csv()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Entry point
    dnn_model = Dnn(PATH_TO_MODEL)
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='path to image file')
    args = parser.parse_args()
    cap = cv.VideoCapture(args.image)
    optical_flow(cap, dnn_model)
