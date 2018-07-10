# Fast Spatial Matching
import _fast_spatial_matching
from .geometric_transforms import Transformation, AffineFeatureMatch



class FastSpatialMatching ():
    def __init__(self):
        self._impl = _fast_spatial_matching.PyFastSpatialMatching()

    def perform_spatial_verification(self, matches):
        best_num_inliers, transform, inliers = self._impl.PerformSpatialVerification(matches)
        return transform, inliers