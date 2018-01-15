import cv2
import numpy as np
from skimage.measure import ransac as _ransac
from skimage.transform import AffineTransform
import matplotlib.pyplot as plt
import numpy as np
import time

import geoburst
from geoburst.geometric_transforms import FeatureGeometryAffine, AffineFeatureMatch
fsmmatcher = geoburst.FastSpatialMatching()

class AffineMatch:
    def __init__(self, query_feature_index, query_keypoint):
        self.feature1 = FeatureGeometryAffine()
        self.feature1.feature_id_ = query_feature_index
        self.feature1.setPosition(query_keypoint[0], query_keypoint[1])
        self.feature1.a_ = query_keypoint[2]
        # TODO: check if 2b is correct or b is correct.

        # Input Keypoint: u,v,a,b,c    in    a(x-u)(x-u)+2b(x-u)(y-v)+c(y-v)(y-v)=1
        #     with (0,0) at image top left corner
        # GeoBurst compatible Keypoint:
        #     a * x^2 + b * xy + c * y^2 = 1 describes all points on the sphere
        self.feature1.b_ = query_keypoint[3]
        self.feature1.c_ = query_keypoint[4]

        self.features2 = []

        # For SFM, below var does not affet the result
        self.word_ids = []

    def get_object(self):
        return AffineFeatureMatch(self.feature1, self.features2, self.word_ids)

def get_matches(des1, des2, method='BRUTE_FORCE', is_binary_descriptor=False):
    # create BFMatcher object
    if method == 'BRUTE_FORCE':
        # BEWARE that, distance is different for binary descriptor and float descriptor. See http://answers.opencv.org/question/59996/flann-error-in-opencv-3/
        if is_binary_descriptor: # ORB, BRIEF, BRISK
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else: # SIFT, SURF
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
    elif method == 'FLANN':
        if is_binary_descriptor:
            raise "Not supported yet"
        else:
            # FLANN parameters
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des1.astype(np.float32),des2.astype(np.float32),k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good.append(m)
                # matchesMask[i]=[1,0]
        matches = good
    return matches

def sift_to_rootsift(descs):
        if descs.dtype != np.float:
            descs = descs.astype(np.float32)
        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        eps = 1e-10
        l1_norm = np.linalg.norm(descs, 1)
        descs /= (l1_norm + eps)
        descs = np.sqrt(descs)
        return descs

def get_ransac_inlier(kp1, kp2, des1, des2,
                      is_binary_descriptor=False,
                      matcher_method="FLANN",
                      verification_method="opencv_ransac_6dof",
                      min_samples=3,
                      residual_threshold=5.0,
                      max_trials=1000):
    """
    Args:
      verification_method: "opencv_ransac_6dof", "scikit", "fsm_5dof"
    """
    des1 = sift_to_rootsift(des1)
    des2 = sift_to_rootsift(des2)
    matches = get_matches(des1, des2, matcher_method, is_binary_descriptor=False)
    if len(matches) < 5:
        # print("not enough matches: {}".format(len(matches)))
        return []

    # Perform geometric verification using RANSAC.
    if verification_method == "opencv_ransac_6dof":
        src_pts = np.float32([cv2.KeyPoint(kp1[m.queryIdx][0], kp1[m.queryIdx][1], 1).pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([cv2.KeyPoint(kp2[m.trainIdx][0], kp2[m.trainIdx][1], 1).pt for m in matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, residual_threshold)

        matchesMask = mask.ravel().tolist()

        inlier_idxs = np.nonzero(matchesMask)[0]
    elif verification_method == "scikit":
        locations_1_to_use = []
        locations_2_to_use = []
        for match in matches:
            locations_1_to_use.append((kp1[match.queryIdx].pt[1], kp1[match.queryIdx].pt[0])) # (row, col)
            locations_2_to_use.append((kp2[match.trainIdx].pt[1], kp2[match.trainIdx].pt[0]))

        locations_1_to_use = np.array(locations_1_to_use)
        locations_2_to_use = np.array(locations_2_to_use)
        _, inliers = _ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials)

        inlier_idxs = np.nonzero(inliers)[0]
    elif verification_method == "fsm_5dof":
        affine_matches = []
        # TODO: handle multi match.
        # when one key points matches multiple key points. check data structure for this and fsm.

        for match_idx, m in enumerate(matches):
            am = AffineMatch(m.queryIdx, kp1[m.queryIdx])

            feature2 = FeatureGeometryAffine()
            feature2.feature_id_ = m.trainIdx
            feature2.setPosition(kp2[m.trainIdx][0], kp2[m.trainIdx][1])
            feature2.a_ = kp2[m.trainIdx][2]
            feature2.b_ = kp2[m.trainIdx][3]
            feature2.c_ = kp2[m.trainIdx][4]

            am.features2.append(feature2)
            am.word_ids.append(m.trainIdx) # pass dummy word_id. We may don't use value for FSM.

            affine_matches.append(am)

        # TODO: check do we need sort by size of features2 (smaller first) for matches. for FSM
        # print("num matches:", len(affine_matches))
        match_list = affine_matches
        match_obj_list = []
        for m in affine_matches:
            match_obj_list.append(m.get_object())


        transform, inliers = fsmmatcher.perform_spatial_verification(match_obj_list)

        # print("transform from SFM:", transform)
        # print("num inliers from SFM:", len(inliers))

        inlier_idxs = []
        for match_idx, feature_idx in inliers:
            inlier_idxs.append(match_idx)
    else:
        raise Exception()

    inlier_match = []
    for idx in inlier_idxs:
        inlier_match.append(matches[idx])

    # print("num kp1: {}, kp2: {}, match: {}, inlier: {}".format(len(kp1), len(kp2), len(matches), len(inlier_match)))
    return inlier_match

def draw_ransac(img1, img2, kp1, kp2, des1, kes2,
                is_binary_descriptor=False,
                match_method='BRTUE_FORCE',
                verification_method="opencv",
                min_samples=3,
                residual_threshold=5.0,
                max_trials=1000,
                description=None):
    inlier_match = get_ransac_inlier(kp1, kp2, des1, kes2, is_binary_descriptor, match_method, verification_method, min_samples, residual_threshold, max_trials)
    # Does ransac consider size and orientation of keypoints?

    cvkp1 = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in kp1]
    cvkp2 = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in kp2]
    ransac_img = cv2.drawMatches(img1, cvkp1, img2, cvkp2, inlier_match, None, flags=0)

    ransac_img = cv2.cvtColor(ransac_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16, 12))
    plt.title(description)
    plt.imshow(ransac_img)
    plt.show()
    return len(inlier_match)

def parse_sift_output(target_path):
    """
    Return:
        kp: keypoint of hessian affine descriptor. location, orientation etc... OpenCV KeyPoint format.
        des: 128d uint8 np array
    """
    import os
    # print(os.listdir("./sample"))
    kp = []
    des = []
    with open(target_path, "r") as f:
        lines = list(map(lambda x: x.strip(), f.readlines()))
        num_descriptor = int(lines[1])
        lines = lines[2:]
        for i in range(num_descriptor):
            # print(i, lines[i])
            val = lines[i].split(" ")
            x = float(val[0])
            y = float(val[1])
            a = float(val[2])
            b = float(val[3])
            c = float(val[4])
            # u,v,a,b,c    in    a(x-u)(x-u)+2b(x-u)(y-v)+c(y-v)(y-v)=1
            # with (0,0) at image top left corner

            # TODO: generate ellipse shaped key point
            # Refer: https://math.stackexchange.com/questions/1447730/drawing-ellipse-from-eigenvalue-eigenvector
            # Refer: http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/display_features.m
            # Refer: http://www.robots.ox.ac.uk/~vgg/research/affine/detectors.html
            key_point = np.array([x, y, a, b, c])
            # key_point = cv2.KeyPoint(x, y, 1)
            sift_descriptor = np.array(list(map(lambda x: int(x), val[5:])), dtype=np.uint8)
            kp.append(key_point)
            des.append(sift_descriptor)


    return kp, np.array(des)

def timeit(func, iter_num=20, msg="function"):
    start = time.time()
    for i in range(iter_num):
        retval = func()
    # print('num_inliner:', len(inliner))
    end = time.time()
    elapsed_time = end - start
    print("timeit for {} (iter: {}): avg: {} total: {}".format(msg, iter_num, elapsed_time/iter_num, elapsed_time))
    return retval, elapsed_time/iter_num


def main():
    kp1, des1 = parse_sift_output(target_path="../sample/all_souls_000026.jpg.hesaff.sift")
    kp2, des2 = parse_sift_output(target_path="../sample/all_souls_000055.jpg.hesaff.sift")


    img1 = cv2.imread('../sample/all_souls_000026.jpg') # query image
    img2 = cv2.imread('../sample/all_souls_000055.jpg')


    verification_method_list = ["opencv_ransac_6dof", "fsm_5dof"]
    limit_num_kp_list = np.logspace(1, 3.8, 30).tolist()
    print("num_kp_candidates:", limit_num_kp_list)

    data_points = []

    for method_idx, verification_method in enumerate(verification_method_list):
        data_points.append([])
        for limit_num_kp in limit_num_kp_list:
            limit_num_kp = int(limit_num_kp)
            def func():
                return get_ransac_inlier(kp1[:limit_num_kp], kp2[:limit_num_kp], des1[:limit_num_kp], des2[:limit_num_kp], False, 'BRUTE_FORCE', verification_method)
            inlier, elapsed_time = timeit(func, msg="geoverif with {} with number of key points {}".format(verification_method, limit_num_kp))
            print("num inliers:", len(inlier))
            data_points[-1].append(elapsed_time)

    # Draw performance comparision graph.
    fig = plt.figure()
    fig.suptitle('geoverif matching time. avg of repeat 20 times', fontsize=20)

    plt.plot(limit_num_kp_list, data_points[0])
    plt.plot(limit_num_kp_list, data_points[1])

    plt.xlabel('number of correspondences', fontsize=18)
    plt.ylabel('time(sec)', fontsize=16)

    plt.xscale('log')
    plt.yscale('log')

    plt.legend(verification_method_list, loc='upper left')

    plt.show()

    # inlier = get_ransac_inlier(kp1, kp2, des1, des2, False, 'BRUTE_FORCE', verification_method)

    # ransac and plot
    # draw_ransac(img1, img2, kp1, kp2, des1, des2, False, 'BRUTE_FORCE', verification_method)

if __name__ == "__main__":
    main()