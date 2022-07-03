import numpy as np

from keras.models import load_model
from PIL import Image


class Dnn:
    def __init__(self, path: str):
        self._path = path
        self._model = load_model(path)

    def get_transformed_binary_map(self, frame: np.array, threshold: float):
        frame_shape_y = frame.shape[1]
        frame_shape_x = frame.shape[0]

        frame = self.get_binary_map(frame, threshold)
        img = Image.fromarray(frame)
        img = img.resize((frame_shape_y, frame_shape_x))

        return np.array(img)

    def get_binary_map(self, frame: np.array, threshold: float):
        img = self._get_image(frame=frame)

        binary_map = self._calculate_map(img)

        binary_map[binary_map < threshold] = 0
        binary_map[binary_map == threshold] = 0
        binary_map[binary_map > threshold] = 1

        return binary_map

    def _calculate_map(self, input_image: np.array) -> np.array:
        test_image = np.expand_dims(input_image, axis=0)
        decoded_img = self._model.predict(test_image)
        show_decoded_image = np.squeeze(decoded_img)
        return show_decoded_image

    def _get_image(self, frame: np.array) -> np.array:
        img = np.array(frame)[:, :, ::-1]
        img = Image.fromarray(img)

        return img.resize((300, 300))
