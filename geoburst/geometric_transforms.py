# Geomteric Transformation
# For now, all python bindings are defined in _fast_spatial_matching.
# However, if the number of objects increaes, we should separate them into different file like this. 
from _fast_spatial_matching import Transformation, FeatureGeometryAffine
from _fast_spatial_matching import PyAffineFeatureMatch as AffineFeatureMatch
import _fast_spatial_matching


# class AffineFeatureMatch:
#     def __init__(self, feature1, features2, word_ids):
#         """

#         """
#         print("init AffineFeatureMatch")
#         print("feature1:", feature1)
#         print("features2:", features2)
#         print("word_ids:", word_ids)
#         self._match=_fast_spatial_matching.AffineFeatureMatch()

#         feature1 = _fast_spatial_matching.FeatureGeometryAffine()
#         # match.feature1_.feature_id_ = query_feature_id;
#         # match.feature1_.x_ << query_descriptors[query_feature_id].x,
#         #                       query_descriptors[query_feature_id].y;
#         # match.feature1_.a_ = query_descriptors[query_feature_id].a;
#         # match.feature1_.b_ = query_descriptors[query_feature_id].b;
#         # match.feature1_.c_ = query_descriptors[query_feature_id].c;
        
#         self._match.feature1_ = feature1
#         self._match.features2_ = features2
#         self._match.word_ids_ = word_ids

# #         geometry::FeatureGeometryAffine f;
# #         f.feature_id_ = used_db_feature_id;
# #         f.x_ << entry.x, entry.y;
# #         f.a_ = entry.a;
# #         f.b_ = entry.b;
# #         f.c_ = entry.c;
# # //        f.scale_ = entry.scale;
# # //        f.orientation_ = entry.orientation;
# #         match.features2_.push_back(f);
# #         match.word_ids_.push_back(init_match.db_feature_word);

