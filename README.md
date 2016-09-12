#  Large-Scale Location Recognition And The Geometric Burstiness Problem

## About
This is an implementation of the state-of-the-art visual place recognition algorithm described in 

    @inproceedings{Sattler16CVPR,
        author = {Sattler, Torsten and Havlena, Michal and Schindler, Konrad and Pollefeys},
        title = {Large-Scale Location Recognition And The Geometric Burstiness Problem},
        booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }

It implements the DisLocation retrieval engine described in

    @Inproceedings{Arandjelovic14a,
        author       = "Arandjelovi\'c, R. and Zisserman, A.",
        title        = "{DisLocation}: {Scalable} descriptor distinctiveness for location recognition",
        booktitle    = "Asian Conference on Computer Vision",
        year         = "2014",
    }
    
as a baseline system and adds different scoring functions based on which the images are ranked:
* re-ranking based on the raw retrieval scores,
* re-ranking based on the number of inlier found for each of the top-ranked database images when fitting an approximate affine transformation,
* re-ranking based on the effective inlier count that takes the distribution of the inlier features in the query image into account,
* re-ranking based on the inter-image geometric burstiness measure, which downweights the impact of features found in the query image that are inliers to many images in the database,
* re-ranking based on the inter-place geometric burstiness measure, which downweights the impact of features found in the query image that are inliers to many places in the scene,
* re-ranking based on the inter-place geometric burstiness measure, which downweights the impact of features found in the query image that are inliers to many places in the scene and also takes the popularity of the different places into account.

## License and Citing 
This software is licensed under the BSD 3-Clause License (also see https://opensource.org/licenses/BSD-3-Clause):

    Copyright (c) 2016, ETH Zurich
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without modification, 
    are permitted provided that the following conditions are met:
    
    1. Redistributions of source code must retain the above copyright notice, this 
    list of conditions and the following disclaimer.
    
    2. Redistributions in binary form must reproduce the above copyright notice, 
    this list of conditions and the following disclaimer in the documentation and/or 
    other materials provided with the distribution.
    
    3. Neither the name of the copyright holder nor the names of its contributors 
    may be used to endorse or promote products derived from this software without 
    specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
If you are using this software for a scientific publication, you need to cite the two following publications:

    @inproceedings{Sattler16CVPR,
        author = {Sattler, Torsten and Havlena, Michal and Schindler, Konrad and Pollefeys},
        title = {Large-Scale Location Recognition And The Geometric Burstiness Problem},
        booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }
    
    @Inproceedings{Arandjelovic14a,
        author       = "Arandjelovi\'c, R. and Zisserman, A.",
        title        = "{DisLocation}: {Scalable} descriptor distinctiveness for location recognition",
        booktitle    = "Asian Conference on Computer Vision",
        year         = "2014",
    }
    
## Compilation & Installation
### Requirements
In order to compile the software, the following software packages are required:
 * CMake version 2.6 or higher (http://www.cmake.org/)
 * Eigen 3.2.1 or higher (http://eigen.tuxfamily.org/)
 * FLANN (https://github.com/mariusmuja/flann)
 * C++11 or higher

### Compilation & Installation
In order to compile the software under Linux, follow the instructions below:
* Go into the order into which you cloned the repository.
* Make sure the include and library directories in ```FindEIGEN.cmake``` and ```FindFLANN.cmake``` in ```cmake/``` are correctly set up.
* Create a directory for compilation: ```mkdir build/```
* Create a directory for installation: ```mkdir release/```
* Compile the source code:
 * ```cd build/```
 * ```cmake cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../release ..```
 * ```make -j```
 * ```make install```
    
This software was developer under Linux and has only been tested on Linux so far. However, it should also compile under Windows and Mac OS X as it does not depend on Linux-specific libraries. It might be necessary to adjust the CMake files though.

# Running the Software
After compilation and installation, the binaries required to run the software are located in the ```release/``` directory.
## Building an Inverted Index.
Before being able to perform queries against a database, it is necessary to build an inverted index. This is done in three stages by executing ```compute_hamming_thresholds```, ```build_partial_index```, and ```compute_index_weights```.

In order to begin the process of building an inverted index, the following files are required:
* A set of database images, e.g., stored in a directory ```db/``` (the actual filename does not matter and subdirectories can also be used). For each database image ```db/a.jpg```, a binary file ```db/a.bin``` needs to exists that stores the features extracted from the image as well as their visual word assignment. Three types of binary files are supported: ``VL_FEAT``, ``VGG binary``, ``HesAff binary``. All of which follow a similar format (see also the function ```LoadAffineSIFTFeaturesAndMultipleNNWordAssignments``` in ```src/structures.h```):
 * The number of features stored in the file, stored as an ```uint32_t```.
 * For each feature, 5 ```float``` values: ``x``, ``y``, ``a``, ``b``, ``c``. Here, ``x`` and ``y`` describes the position of the feature in the image. ``a``, ``b``, and ``c`` specify the elliptical region that defines the feature (more on this below).
 * One visual word assignment, stored as a ``uint32_t``.
 * The SIFT feature descriptor for that feature (stored as ``float``s or ``uint8_t``s.
 * The software currently supports three types of affine covariant features:
  * 

### Computing the Hamming Thresholds

### Building a Partial Index

### Finalizing the Index

# Acknowledgements
This work was supported by Google’s Project Tango and EC Horizon 2020 project REPLICATE (no. 687757). The authors thank Relja Arandjelović for his invaluable help with the DisLoc algorithm.
