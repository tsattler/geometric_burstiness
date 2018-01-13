# Fast Spatial Matching
import _fast_spatial_matching
from .geometric_transforms import Transformation, AffineFeatureMatch



class FastSpatialMatching ():
    def __init__(self):
        print("init FastSpatialMatching class")
        self._impl = _fast_spatial_matching.PyFastSpatialMatching()

    def perform_spatial_verification(self, matches):
        # if len(matches) == 0:
        #     # handling passed empty list of matches
        #     return None, []
        print("invoke PerformSpatialVerification()")
        best_num_inliers, transform, inliers = self._impl.PerformSpatialVerification(matches)
        print("best_num_inliers:", best_num_inliers)
        return transform, inliers