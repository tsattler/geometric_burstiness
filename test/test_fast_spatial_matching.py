import unittest
from geoburst import FastSpatialMatching
from geoburst.geometric_transforms import AffineFeatureMatch
from geoburst.geometric_transforms import FeatureGeometryAffine
import numpy


class TestFastSpatialMatching(unittest.TestCase):
    def setUp(self):
        self.matcher = FastSpatialMatching()
    def test_matching_with_empty_match(self):
        mathces = []
        self.matcher.perform_spatial_verification(mathces)

    def test_matching_with_single_match(self):
        matches = []

        feature1 = FeatureGeometryAffine()
        features2 = []
        features2.append(FeatureGeometryAffine())
        features2.append(FeatureGeometryAffine())
        word_ids = [0, 1]

        match = AffineFeatureMatch(feature1, features2, word_ids)
        matches.append(match)

        self.matcher.perform_spatial_verification(matches)
