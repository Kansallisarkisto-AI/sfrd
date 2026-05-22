import cv2
import numpy as np
import copy
import math

from .transforms import apply_affine_numba
from .config import config

FLANN_INDEX_KDTREE = 1
# 1 thread for now because we use multiprocessing elsewhere
cv2.setNumThreads(1)


class SIFTFeatures:
    """
    Calculates and stores SIFT keypoints and descriptors for an image.
    """
    def __init__(self, sift_matcher, image):
        """
        Converts image to grayscale and calculates SIFT keypoints and descriptors.
        Results are stored as tuples to facilitate easy serialization.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, self.descriptors = sift_matcher.sift.detectAndCompute(
            gray_image, None)
        self.keypoints = [(point.pt, point.size, point.angle, point.response,
                           point.octave, point.class_id) for point in keypoints]

    def get_cv2_keypoints(self):
        """
        Returns stored keypoints converted to OpenCV format.
        """
        return [cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2],
                             response=point[3], octave=point[4], class_id=point[5]) for point in self.keypoints]


class SIFTFeaturesCopy:
    """
    Serializable minimal copy of SIFTFeatures
    """
    def __init__(self, keypoints, descriptors):
        self.keypoints, self.descriptors = keypoints, descriptors

    def get_cv2_keypoints(self):
        """
        Returns stored keypoints converted to OpenCV format.
        """
        return [cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2],
                             response=point[3], octave=point[4], class_id=point[5]) for point in self.keypoints]


class SIFTMatcher:
    """
    Matches SIFT feature descriptors using FLANN.
    Perform's Lowe's ratio filtering and estimates affine transformation using RANSAC.
    """
    def __init__(self, lowes_ratio_threshold=0.7, min_matches=10, nfeatures=10000, num_trees=5):
        self.lowes_ratio_threshold = lowes_ratio_threshold
        self.min_matches = min_matches
        self.nfeatures = nfeatures

        # SIFT detector/descriptor
        # these are proven defaults which are not recommended to change
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=config["sift"]["contrast_threshold"],
            edgeThreshold=config["sift"]["edge_threshold"],
            sigma=config["sift"]["sigma"]
        )
        # FLANN for SIFT (float descriptors)
        self.index_params = dict(
            algorithm=FLANN_INDEX_KDTREE,
            trees=num_trees
        )
        self.search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(
            self.index_params, self.search_params)

    def match_features(self, feats_a, feats_b):
        """Matches SIFT descriptors between two SIFTFeatures objects.

        Uses FLANN-based nearest neighbor matching followed by Lowe's ratio test.

        Args:
            feats_a: First SIFTFeatures object.
            feats_b: Second SIFTFeatures object.

        Returns:
            list[cv2.DMatch]: Filtered descriptor matches passing Lowe's ratio test.
        """
        matches = []

        if feats_a.descriptors is None or feats_b.descriptors is None:
            return []

        desc_a = np.asarray(feats_a.descriptors, dtype=np.float32)
        desc_b = np.asarray(feats_b.descriptors, dtype=np.float32)

        if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
            return []

        all_matches = self.flann.knnMatch(desc_a, desc_b, k=2)

        if len(all_matches) > 0:
            for m, n in all_matches:
                # Lowe's ratio test
                if m.distance < self.lowes_ratio_threshold * n.distance:
                    matches.append(m)

        return matches

    def match_ransac_affine(self, feats_a, feats_b,
                            ransac_reproj_threshold=3.0,
                            max_iters=5000,
                            confidence=0.999):
        """Estimates an affine transform using matched SIFT features.

        Feature matches are filtered using Lowe's ratio test and refined with
        RANSAC-based affine estimation.

        Args:
            feats_a: Source image features.
            feats_b: Destination image features.
            ransac_reproj_threshold: Maximum reprojection error allowed for RANSAC.
            max_iters: Maximum number of RANSAC iterations.
            confidence: Desired RANSAC confidence level.

        Returns:
            tuple:
                - np.ndarray | None: 2x3 affine transform matrix mapping
                image A to image B.
                - list[tuple[int, int]]: Inlier keypoint index pairs.
                - float | None: Inlier ratio after RANSAC filtering.
        """
        matches = self.match_features(feats_a, feats_b)

        # early return: require at least min_matches to perform RANSAC
        if len(matches) < self.min_matches:
            return None, [], None

        feats_a_keypoints = feats_a.get_cv2_keypoints()
        feats_b_keypoints = feats_b.get_cv2_keypoints()
        points_a = np.float32(
            [feats_a_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points_b = np.float32(
            [feats_b_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # restricted to partial affine transformations for robustness
        M, inliers = cv2.estimateAffinePartial2D(
            points_a, points_b,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_reproj_threshold),
            maxIters=int(max_iters),
            confidence=float(confidence)
        )
        if M is None or inliers is None:
            return None, [], None

        inliers = inliers.reshape(-1).astype(bool)
        # require at least min_matches inliers to return answer
        if int(inliers.sum()) < self.min_matches:
            return None, [], None

        inlier_pairs = []
        for keep, m in zip(inliers, matches):
            if keep:
                inlier_pairs.append((int(m.queryIdx), int(m.trainIdx)))

        # this inlier ratio is calculated
        # after Lowe's ratio test and the FLANN matcher
        inlier_ratio = len(inlier_pairs) / len(matches)

        return M.astype(np.float64), inlier_pairs, inlier_ratio


def estimate_affine_from_landmarks(image_index: int, fi, kpnode_to_lm, lm_canon,
                                   ransac_thresh=3.0, min_matches=10, max_iters=5000, confidence=0.999):
    """
    Estimate affine transformation from image i -> root using correspondences between page keypoints and root canonical keypoints.

    Args:
        image_index: Index of the image being aligned.
        fi: SIFTFeatures corresponding to image_index
        kpnode_to_lm: Mapping from keypoint nodes to landmark identifiers.
        lm_canon: Mapping from landmark identifiers to canonical coordinates.
        ransac_thresh: RANSAC reprojection threshold.
        min_matches: Minimum required correspondences.
        max_iters: Maximum number of RANSAC iterations.
        confidence: Desired RANSAC confidence level.
    
    Returns:
        tuple | None:
            - np.ndarray: 2x3 affine transform matrix.
            - int: Number of inlier correspondences.
            - float: Mean residual error for inliers only.

        Returns None if estimation fails.
    """
    src = []
    dst = []
    for kpidx, kp in enumerate(fi.keypoints):
        lm = kpnode_to_lm.get((image_index, kpidx))
        if lm is None:
            continue
        canon = lm_canon.get(lm)
        if canon is None:
            continue
        # kp[0] = kp.pt, keypoint location in image
        src.append([kp[0][0], kp[0][1]])
        # canon keypoint location (in root frame)
        dst.append([float(canon[0]), float(canon[1])])

    if len(src) < min_matches:  # require minimum number of matches
        return None

    src = np.asarray(src, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=max_iters,
        confidence=confidence
    )
    if M is None or inliers is None:
        return None

    # residual calculation (ONLY inliers)
    inlier_mask = inliers.ravel().astype(bool)

    src_in = src[inlier_mask].reshape(-1, 2)
    dst_in = dst[inlier_mask].reshape(-1, 2)

    proj = apply_affine_numba(M, src_in)

    residuals = np.linalg.norm(proj - dst_in, axis=1)
    mean_residual = float(
        residuals.mean()) if residuals.size > 0 else float("inf")

    return M.astype(np.float64), int(inlier_mask.sum()), mean_residual


def affine_is_valid(M,
                    image_shape_hw,
                    max_translation=config["transform_limits"]["max_translation"],
                    min_scale=config["transform_limits"]["min_scale"],
                    max_scale=config["transform_limits"]["max_scale"],
                    max_rotation_deg=config["transform_limits"]["max_rotation_deg"]):
    """
    Heuristic for transformation validity, using the range of typical maximum offsets on a camera table, relative to image size.

    Args:
        M: 2x3 affine transformation matrix.
        image_shape_hw: Tuple of image height and width.
        max_translation: Maximum normalized translation magnitude.
        min_scale: Minimum allowed scale factor.
        max_scale: Maximum allowed scale factor.
        max_rotation_deg: Maximum allowed absolute rotation angle.

    Returns:
        bool: True if the transform is considered valid, otherwise False.
    """
    if M is None:
        return False

    tx, ty = M[0, 2], M[1, 2]
    h, w = image_shape_hw
    diag = np.hypot(w, h)
    normalized_translation = np.hypot(tx, ty) / diag
    if normalized_translation > max_translation:
        return False

    scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
    if not (min_scale <= scale <= max_scale):
        return False

    angle = abs(math.degrees(math.atan2(M[0, 1], M[0, 0])))
    if angle > max_rotation_deg:
        return False

    return True
