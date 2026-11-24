from cv2.typing import MatLike
import cv2
import numpy as np
from typing import Literal, Sequence
from dataclasses import dataclass

from numpy._typing import NDArray

@dataclass
class PointCloud:
    cam_pos_lst: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]

@dataclass
class FrameData:
    frame: MatLike
    keypoints: Sequence[cv2.KeyPoint]
    descriptors: MatLike

class PointCloudEstimator:
    # Constants
    BF_MATCH_THRESHOLD: float = 0.7

    # Data structures 
    frame_data_lst: list[FrameData]

    def __init__(self) -> None:
        self.frame_data_lst = []

    def generate_positions(self, video: cv2.VideoCapture) -> PointCloud:
        self.filter_frames(video)

        # TODO: continue the skeleton
        raise NotImplementedError("not finish")

    def filter_frames(self, video: cv2.VideoCapture) -> list[FrameData]:
        self.frame_data_lst = []

        sift = cv2.SIFT.create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # First frame
        _, first_frame = video.read()
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        first_frame_keypoints, first_frame_descriptors = sift.detectAndCompute(
            first_frame, None)

        # Package first frame to data class
        latest_keyframe_data: FrameData = FrameData(
            first_frame, 
            first_frame_keypoints, 
            first_frame_descriptors)
        self.frame_data_lst.append(latest_keyframe_data)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(frame, None)
            current_frame_data: FrameData = FrameData(frame, keypoints, descriptors)

            # Here, we match the descriptors, and apply Lowe's ratio test. 
            # Throw away frames with minimal matches.
            matches = bf.knnMatch(latest_keyframe_data.descriptors, descriptors, k=2)
            good_matches = [
                pair[0] for pair in matches 
                if len(pair) == 2 and 
                pair[0].distance < self.BF_MATCH_THRESHOLD * pair[1].distance]

            if self.keyframe_check(
                latest_keyframe_data, 
                current_frame_data, 
                good_matches
            ):
                self.frame_data_lst.append(current_frame_data)
                latest_keyframe_data = current_frame_data

        return self.frame_data_lst

    def keyframe_check(
            self, 
            latest_keyframe: FrameData, 
            current_frame: FrameData, 
            good_matches: list[cv2.DMatch]
    ) -> bool:
        # Constants
        RANSAC_REPROJ_HOMOGRAPHY = 5.0
        RANSAC_REPROJ_FUNDAMENTAL = 3.0
        SYMMETRIC_TRANSFER_SCORE_THRESHOLD = 5.99 # chi-sq 2 DoF 95% confidence
        SYMMETRIC_EPIPOLAR_SCORE_THRESHOLD = 3.84 # chi-sq 1 DoF 95% confidence
        ACCEPTANCE_RATIO = 0.45 # From ORB-SLAM original formulation
        GOOD_MATCHES_MIN_SIZE = 30 # Too few stuffs => there is essentially nothing to match.

        if len(good_matches) < GOOD_MATCHES_MIN_SIZE:
            return False

        pts_1 = np.array([
            latest_keyframe.keypoints[m.queryIdx].pt for m in good_matches
        ]).astype(np.float32)
        pts_2 = np.array([
            current_frame.keypoints[m.trainIdx].pt for m in good_matches
        ]).astype(np.float32)

        # Parallax section

        H, _ = cv2.findHomography(
            srcPoints= pts_1, 
            dstPoints= pts_2, 
            method= cv2.RANSAC, 
            ransacReprojThreshold= RANSAC_REPROJ_HOMOGRAPHY
        )
        F, _ = cv2.findFundamentalMat(
            points1= pts_1, 
            points2= pts_2, 
            method= cv2.FM_RANSAC, 
            ransacReprojThreshold= RANSAC_REPROJ_FUNDAMENTAL
        )

        if H is None or F is None:
            return False
        if F.shape != (3, 3):
            return False

        H_score = self.symmetric_transfer_score(
            H, 
            pts_1, 
            pts_2, 
            SYMMETRIC_TRANSFER_SCORE_THRESHOLD
        )
        F_score = self.symmetric_epipolar_score(
            F,
            pts_1,
            pts_2,
            SYMMETRIC_EPIPOLAR_SCORE_THRESHOLD
        )

        return (H_score / (H_score + F_score)) <= ACCEPTANCE_RATIO
        
    def symmetric_transfer_score(
            self, 
            H: MatLike, 
            pts_1: NDArray[np.float32], 
            pts_2: NDArray[np.float32], 
            threshold: float
    ) -> float:
        def reprojection_error(
                H: MatLike, 
                p1: NDArray[np.float32],
                p2: NDArray[np.float32]
        ) -> float:
            p1_homo = np.array([p1[0], p1[1], 1.0]).astype(np.float32)
            proj_homo = H @ p1_homo
            proj = np.array([proj_homo[0], proj_homo[1]] / proj_homo[2]).astype(np.float32)
            err = proj - p2
            return err.dot(err)

        H_inv = np.linalg.inv(H)
        score = 0.0
        for p1, p2 in zip(pts_1, pts_2):
            d1: float = reprojection_error(H, p1, p2)
            d2: float = reprojection_error(H_inv, p2, p1)
            for d in (d1, d2):
                score += max(threshold - d, 0)
        return score

    def symmetric_epipolar_score(
            self, 
            F: MatLike, 
            pts_1: NDArray[np.float32], 
            pts_2: NDArray[np.float32], 
            threshold: float
    ) -> float:
        def epipolar_dist(
                F: MatLike, 
                p1: NDArray[np.float32], 
                p2: NDArray[np.float32]
        ) -> float:
            p1_h = np.array([p1[0], p1[1], 1]).astype(np.float32)
            p2_h = np.array([p2[0], p2[1], 1]).astype(np.float32)
            l2 = F @ p1_h
            residual = p2_h @ l2
            residual_normalized = residual ** 2 / (l2[0] ** 2 + l2[1] ** 2)
            return residual_normalized

        score = 0.0
        for p1, p2 in zip(pts_1, pts_2):
            d1: float = epipolar_dist(F, p1, p2)
            d2: float = epipolar_dist(F.T, p2, p1)
            for d in (d1, d2):
                score += max(threshold - d, 0)
        return score
