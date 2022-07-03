import cv2 as cv


class OptFlowSettings:
    def __init__(
            self,
            maxCorners: int = 300,
            qualityLevel: float = 0.1,
            minDistance: int = 1,
            blockSize: int = 1,
            winSize: tuple = (15, 15),
            maxLevel: int = 2,
    ):
        self._maxCorners = maxCorners
        self._qualityLevel = qualityLevel
        self._minDistance = minDistance
        self._blockSize = blockSize
        self._winSize = winSize
        self._maxLevel = maxLevel

    def get_lk_params(self) -> dict:
        """
        Lucas Kanade optical flow params
        """
        return dict(
            winSize=self._winSize,
            maxLevel=self._maxLevel,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def feature_params(self) -> dict:
        """
        ShiTomasi corner detection
        """
        return dict(
            maxCorners=300, qualityLevel=0.1, minDistance=1, blockSize=1,
        )
