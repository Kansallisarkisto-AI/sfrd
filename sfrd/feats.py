import cv2
import numpy as np
import copy
import math

from .transforms import apply_affine_numba, invert_affine_numba
from .config import config
from .ioutils import read_image_cv

# B-Spline and TPS fine-grained deformation correction
import SimpleITK as sitk
from torch_tps import ThinPlateSpline
import torch

FLANN_INDEX_KDTREE = 1
# 1 thread for now because we use multiprocessing elsewhere
cv2.setNumThreads(1)

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


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

def bspline_mesh_by_short_side(width, height, short_side_cells):
    if width <= height:
        mx = short_side_cells
        my = round(short_side_cells * height / width)
    else:
        my = short_side_cells
        mx = round(short_side_cells * width / height)

    return [max(1, int(mx)), max(1, int(my))]

# GRID

def displacement_stats(transform, width, height, n=20):
    xs = np.linspace(0, width - 1, n)
    ys = np.linspace(0, height - 1, n)

    mags = []

    for y in ys:
        for x in xs:
            tx, ty = transform.TransformPoint((float(x), float(y)))

            dx = tx - x
            dy = ty - y

            mags.append(np.hypot(dx, dy))

    mags = np.asarray(mags)

    print("mean disp:", mags.mean())
    print("median   :", np.median(mags))
    print("max disp :", mags.max())

def apply_sitk_transform_to_points(transform, points):
    points = np.asarray(points, dtype=np.float64)
    if transform is None:  # do not apply transform if it doesn't exist
        return points

    return np.around(np.asarray(
        [transform.TransformPoint((float(x), float(y))) for x, y in points],
        dtype=np.float64,
    )).astype(np.int32)

def fullscale_bsplinegrid_from_landmarks(page, page_idx, M_full, page_scale, root_scale, fi, kpnode_to_lm, lm_canon,
                                         order=3):
    """
    Estimate a B-spline grid transformation using correspondences between page keypoints and root canonical keypoints.
    Transformation is calculated from root to page, in current page coordinate system instead of the root's coordinate system.
    That is because the B-spline transformation will be applied as a post-correction _after_ the root annotation 
    has been converted to the page coordinate sytem (with the inverse affine transformation).
    Here we use all keypoints (not only inliers), since we are trying to correct for the small-scale deformations
    which might mean points that are not inliers after the simpler RANSAC/affine fit. Hopefully any erraneous matches
    won't affect the estimation too much or the errors will balance out.

        Returns None if estimation fails.
    """
    root_landmarks = [] # in page coordinate system (not in the root's coordinate system!)
    page_landmarks = [] # in page coordinate system
    for kpidx, kp in enumerate(fi.keypoints):
        lm = kpnode_to_lm.get((page_idx, kpidx))
        if lm is None:
            continue
        canon = lm_canon.get(lm)
        if canon is None:
            continue
        
        # Landmarks' (X, Y) pairs are flattened into 1-d lists.
        # kp[0] = kp.pt, keypoint location in image
        page_landmarks.append((kp[0][0] / page_scale, kp[0][1] / page_scale))  # scale into fullres
        # canon keypoint location (in root frame)
        root_landmarks.append((float(canon[0]) / root_scale, float(canon[1]) / root_scale))  # scale into fullres
    
    
    M_root_to_page_full = invert_affine_numba(M_full)

    root_landmarks_array = np.asarray(root_landmarks, dtype=np.float64)
    root_pts_in_page = apply_affine_numba(
        M_root_to_page_full,
        root_landmarks_array,
    )

    # Set up the LandmarkBasedTransformInitializerFilter.
    landmark_initializer = sitk.LandmarkBasedTransformInitializerFilter()
    
    img = sitk.ReadImage(str(page))
    w, h = img.GetSize()

    fixed_image = sitk.Image(w, h, sitk.sitkFloat32)
    fixed_image.SetOrigin((0.0, 0.0))
    fixed_image.SetSpacing((1.0, 1.0))
    fixed_image.SetDirection((1.0, 0.0, 0.0, 1.0))

    landmark_initializer.SetReferenceImage(fixed_image)

    fixed_pts = np.asarray(page_landmarks, dtype=np.float64).reshape(-1, 2)
    moving_pts = np.asarray(root_pts_in_page, dtype=np.float64).reshape(-1, 2)

    # ensure landmarks stay inside fixed image domain
    mask = (
        (fixed_pts[:, 0] >= 0) & (fixed_pts[:, 0] < w) &
        (fixed_pts[:, 1] >= 0) & (fixed_pts[:, 1] < h) &
        (moving_pts[:, 0] >= 0) & (moving_pts[:, 0] < w) &
        (moving_pts[:, 1] >= 0) & (moving_pts[:, 1] < h)
    )

    fixed_pts = fixed_pts[mask]
    moving_pts = moving_pts[mask]

    mesh_size = bspline_mesh_by_short_side(w, h, config["bspline"]["cells_on_shorter_side"])

    # From docs at https://simpleitk.readthedocs.io/en/master/registrationOverview.html
    # The goal of registration is to estimate the transformation which maps points from one
    # image to the corresponding points in another image. The transformation estimated via
    # registration is said to map points from the fixed image coordinate system to
    # the moving image coordinate system.

    # THEREFORE, we need to swap the landmarks here, so that we get a transformation
    # from the moving image (root) to the fixed image
    landmark_initializer.SetMovingLandmarks(fixed_pts.ravel().tolist())
    landmark_initializer.SetFixedLandmarks(moving_pts.ravel().tolist())

    transform = sitk.BSplineTransformInitializer(
        fixed_image,
        transformDomainMeshSize=mesh_size,
        order=order
    )

    if config["debug"]["enabled"]:
        print(f"Landmarks for B-Spline: {fixed_pts.shape[0]}")

    if fixed_pts.shape[0] < transform.GetNumberOfParameters():
        print("Not enough matches for B-Spline fit! Need at least as many landmarks as number of parameters"
        " in B-Spline mesh (this equals a 2x parameter count margin since each landmark has x and y)" \
        "\nTry decreasing B-Spline cells_on_shorter_side or disabling B-Splines.")
        return None

    # Compute the landmark-fitted BSpline transform.
    output_transform = landmark_initializer.Execute(transform)

    if config["debug"]["enabled"]:
        displacement_stats(output_transform, w, h)

        before = np.linalg.norm(
            fixed_pts - moving_pts,
            axis=1
        )

        warped = np.array([
            output_transform.TransformPoint(tuple(p))
            for p in moving_pts
        ])

        after = np.linalg.norm(
            fixed_pts - warped,
            axis=1
        )

        print("Before mean:", before.mean())
        print("After mean :", after.mean())
        print("Reduction  :", before.mean() - after.mean())
        print("Ratio      :", before.mean() / after.mean())

    return output_transform

def apply_tps(tps, pts, W, H):
    if tps is None:
        return pts
    pts = normalize_pts(pts, W, H)
    transformed_pts = tps.transform(torch.tensor(pts, dtype=torch.float32))
    transformed_pts = denormalize_pts(transformed_pts, W, H)
    return np.around(transformed_pts.numpy()).astype(np.int32)

def normalize_pts(pts, W, H):
    if isinstance(pts, torch.Tensor):
        pts = pts.float().clone()
    else:
        pts = np.asarray(pts, dtype=np.float32).copy()

    pts[:, 0] = 2 * pts[:, 0] / (W - 1) - 1
    pts[:, 1] = 2 * pts[:, 1] / (H - 1) - 1
    return pts


def denormalize_pts(pts, W, H):
    if isinstance(pts, torch.Tensor):
        pts = pts.float().clone()
    else:
        pts = np.asarray(pts, dtype=np.float32).copy()

    pts[:, 0] = (pts[:, 0] + 1) * (W - 1) / 2
    pts[:, 1] = (pts[:, 1] + 1) * (H - 1) / 2
    return pts

def fullscale_thinplatespline_from_landmarks(page, page_idx, M_full, page_shape, page_scale, root_scale, fi, kpnode_to_lm, lm_canon,
                                             regularization_parameter=0.05):
    """
    Estimate a thin plate spline (TPS) transformation using correspondences between page keypoints and root canonical keypoints.
    Transformation is calculated from root to page, in current page coordinate system instead of the root's coordinate system.
    That is because the TPS transformation will be applied as a post-correction _after_ the root annotation 
    has been converted to the page coordinate sytem (with the inverse affine transformation).
    Here we use all keypoints (not only inliers), since we are trying to correct for the small-scale deformations
    which might mean points that are not inliers after the simpler RANSAC/affine fit. Hopefully any erraneous matches
    won't affect the estimation too much or the errors will balance out with regularization.

        Returns None if estimation fails.
    """
    root_landmarks = [] # in page coordinate system (not in the root's coordinate system!)
    page_landmarks = [] # in page coordinate system
    for kpidx, kp in enumerate(fi.keypoints):
        lm = kpnode_to_lm.get((page_idx, kpidx))
        if lm is None:
            continue
        canon = lm_canon.get(lm)
        if canon is None:
            continue
        
        # Landmarks' (X, Y) pairs are flattened into 1-d lists.
        # kp[0] = kp.pt, keypoint location in image
        page_landmarks.append((kp[0][0] / page_scale, kp[0][1] / page_scale))  # scale into fullres
        # canon keypoint location (in root frame)
        root_landmarks.append((float(canon[0]) / root_scale, float(canon[1]) / root_scale))  # scale into fullres
    
    #page_landmarks = np.asarray(page_landmarks, dtype=np.float32)
    page_landmarks = torch.tensor(page_landmarks, dtype=torch.float32)
    M_root_to_page_full = invert_affine_numba(M_full)

    root_landmarks_array = np.asarray(root_landmarks, dtype=np.float64)

    root_landmarks = apply_affine_numba(
        M_root_to_page_full,
        root_landmarks_array,
    )

    root_landmarks = torch.tensor(root_landmarks, dtype=torch.float32)

    #root_landmarks = root_landmarks.reshape(1, -1, 2)
    #page_landmarks = page_landmarks.reshape(1, -1, 2)

    # Initialize TPS transformer
    tps = cv2.createThinPlateSplineShapeTransformer()

    #img = read_image_cv(page)
    w = page_shape[1] / page_scale  # recover original full-resolution shape
    h = page_shape[0] / page_scale
    #diagonal = math.sqrt(img.shape[0]**2 + img.shape[1]**2)

    root_landmarks = normalize_pts(root_landmarks, w, h)
    page_landmarks = normalize_pts(page_landmarks, w, h)

    tps = ThinPlateSpline(alpha=regularization_parameter)

    tps.fit(root_landmarks, page_landmarks)

    if config["debug"]["enabled"]:
        #src = root_landmarks.reshape(-1, 2).astype(np.float64)
        #dst = page_landmarks.reshape(-1, 2).astype(np.float64)
        src = root_landmarks.numpy()
        dst = page_landmarks.numpy()

        print(f"Landmarks for TPS: {src.shape[0]}")

        before = np.linalg.norm(dst - src, axis=1)

        #warped = apply_tps(tps, root_landmarks)
        warped = tps.transform(root_landmarks)
        #warped = np.asarray(warped, dtype=np.float64).reshape(-1, 2)

        after = np.linalg.norm(dst - warped.numpy(), axis=1)

        print("Before mean:", before.mean())
        print("After mean :", after.mean())
        print("Reduction  :", before.mean() - after.mean())
        print("Ratio      :", before.mean() / after.mean() if after.mean() > 0 else np.inf)

    return {"tps": tps, "page_shape" : (h, w)}