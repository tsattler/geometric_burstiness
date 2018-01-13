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

        out_transform = Transformation()
        out_inliers = []
        print("invoke PerformSpatialVerification()")
        self._impl.PerformSpatialVerification(matches, out_transform, out_inliers)
        return out_transform, out_inliers

    
        
     
