## Important Bugfixes / Notes
* 01/08/2018: Please note that the RootSIFT features used in this program have descriptor entries in the range [0, 255]. This quantization is used to severely reduce memory requirements (1 byte per entry instead of 4 bytes). Please make sure that you do not normalize the RootSIFT descriptors before training the codebook.
* 10/27/2016: Fixed bug in the inter-place burstiness weighting schemes. In order to reproduce the results from the paper, the input inliers_per_db for the function ReRankingInterPlaceGeometricBurstiness should be sorted in descending number of inliers. Previously, it was sorted based on the retrieval scores only. This is now fixed.

#  Large-Scale Location Recognition And The Geometric Burstiness Problem

## About
This is an implementation of the state-of-the-art visual place recognition algorithm described in 

    @inproceedings{Sattler16CVPR,
        author = {Sattler, Torsten and Havlena, Michal and Schindler, Konrad and Pollefeys, Marc},
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
This software is licensed under the BSD 3-Clause License (also see https://opensource.org/licenses/BSD-3-Clause) for **non-commercial use**:

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
        author = {Sattler, Torsten and Havlena, Michal and Schindler, Konrad and Pollefeys, Marc},
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

If you are interested in using this software **commercially**, please contact Torsten Sattler (torsten.sattler@inf.ethz.ch).
    
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
 * ```cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../release ..```
 * ```make -j```
 * ```make install```
    
This software was developer under Linux and has only been tested on Linux so far. However, it should also compile under Windows and Mac OS X as it does not depend on Linux-specific libraries. It might be necessary to adjust the CMake files though.

# Running the Software
After compilation and installation, the binaries required to run the software are located in the ```release/``` directory. 
Calling any executable without any parameters will display the parameters of the binary together with a short explanation of them.
## Building an Inverted Index
Before being able to perform queries against a database, it is necessary to build an inverted index. This is done in three stages by executing ```compute_hamming_thresholds```, ```build_partial_index```, and ```compute_index_weights```.

In order to begin the process of building an inverted index, the following files are required:
* A set of database images, e.g., stored in a directory ```db/``` (the actual filename does not matter and subdirectories can also be used). For each database image ```db/a.jpg```, a binary file ```db/a.bin``` needs to exists that stores the features extracted from the image as well as their visual word assignment. Three types of binary files are supported: ``VL_FEAT``, ``VGG binary``, ``HesAff binary``. All of which follow a similar format (see also the function ```LoadAffineSIFTFeaturesAndMultipleNNWordAssignments``` in ```src/structures.h```):
 * The number of features stored in the file, stored as an ```uint32_t```.
 * For each feature, 5 ```float``` values: ``x``, ``y``, ``a``, ``b``, ``c``. Here, ``x`` and ``y`` describes the position of the feature in the image. ``a``, ``b``, and ``c`` specify the elliptical region that defines the feature (more on this below).
 * One visual word assignment, stored as a ``uint32_t``.
 * The SIFT feature descriptor for that feature (stored as ``float``s or ``uint8_t``s.
 * The software currently supports three feature file formats:
     * ``VL_FEAT``: For this type of binary files, the descriptor entries are stored using ``float`` values. The affine parameters ``a``, ``b``, and ``c`` define a matrix ``M = [a, b; b, c]`` whose inverse is ``M = [a', b'; b' c']``, where the ellipse centered around the position ``x``, ``y`` of the keypoint is defined as ``a'^2 * (u - x)^2 + 2 * b' * (u - x) * (v - y) + c'^2 * (v - y)^2 = 1``.
     * ``VGG binary`` and ``HesAff binary``: For these two type of binary files, the descriptor entries are stored using ``uint8_t`` values. The affine parameters ``a``, ``b``, and ``c`` directly define the elliptical region as ``a^2 * (u - x)^2 + 2 * b * (u - x) * (v - y) + c^2 * (v - y)^2 = 1``.
     * We provide two executables as an aid for creating these files: ```hesaff_sift_to_binary_root_sift``` and ```compute_word_assignments```:
       * ``hesaff_sift_to_binary_root_sift``` can be used to convert SIFT features extracted with the upright Hessian affine region detector from https://github.com/perdoch/hesaff to RootSIFT features stored in binary files. The input of the method is a text file containing the list of descriptor file names that should be converted as well as a list of output filenames, one for each input filename.
       * ``compute_word_assignments``: Given a visual vocabulary, stored in a text file such that every row corresponds to one visual word, the number of words in that text file, the number of nearest words that should be computed (1 for the database images), and a list of filenames of the files created by ``hesaff_sift_to_binary_root_sift``, the executable computes the visual word assignments and writes the features and visual word assignments in the format described above. Each file ``x.desc`` in the list of filenames results in a file ``x.desc.postfix``, where ``postfix`` is one parameter of the method (e.g., use ``bin``). Please note that the RootSIFT features used in this program have descriptor entries in the range [0, 255]. This quantization is used to severely reduce memory requirements (1 byte per entry instead of 4 bytes). Please make sure that you do not normalize the RootSIFT descriptors before training the codebook.
* A text file storing a 64x128 projection matrix that can be used to compute the Hamming embeddings (see the original publication on Hamming embeddings, https://hal.inria.fr/inria-00316866/document). The matrix is stored in row-major order, e.g., each line stores one row of the projection matrix.

### Computing the Hamming Thresholds
Once the data is prepared, the first step is to compute the threshold for each visual word that is used for the Hamming embedding. This is done using the executable ``compute_hamming_thresholds``. The input to the method are
* ``list_bin``: A text file, e.g., ``my_binary_list.txt``, containing the filenames of all binary files containing the features and word assignments of the database images. If the database contains the images ``db/a.jpg`` and ``db/b.jpg``, ``list_bin`` should contain two lines ``db/a.bin`` and ``db/b.bin``, where ``db/a.bin`` and ``db/b.bin`` are the binary descriptor files (described above) for the two database images.
* ``projection``: A text file, e.g., ``projection_matrix.txt``, containing the projection matrix described above.
* ``out``: The filename of a text file, e.g., ``thresholds_and_projection.txt``, in which the thresholds and the projection matrix will be stored.
* ``sift_type``: The type of features, i.e., ``0`` for ``VL_FEAT``, ``1`` for ``VGG binary``, and ``2`` for ``HesAff``.
* ``nn``: The number of nearest visual words stored in the binary files. Set this to ``1``.
* 
For HesAff features and the filename examples from above, the call to the executable is 

``compute_hamming_thresholds my_binary_list.txt projection_matrix.txt thresholds_and_projection.txt 2 1``.

### Building a Partial Index
The next step is to build the inverted index, which can be done with the command line tool ``build_partial_index``.
The tool expects the following parameters:
* ``list_bin``: A text file, e.g., ``my_binary_list.txt``, containing the filenames of all binary files containing the features and word assignments of the database images. This is the same file already used for ``compute_hamming_thresholds``.
* ``hamming_and_projection``: The filename of the text file generated by ``compute_hamming_thresholds``.
* ``out``: The filename of the binary file into which the inverted index will be written.
* ``sift_type``: The type of features, i.e., ``0`` for ``VL_FEAT``, ``1`` for ``VGG binary``, and ``2`` for ``HesAff``.
* ``nn``: The number of nearest visual words stored in the binary files. Set this to ``1``.
* 
Following the example above, the inverted index would be constructed by calling 

``build_partial_index my_binary_list.txt thresholds_and_projection.txt inverted_index.bin 2 1``.

``compute_hamming_thresholds my_binary_list.txt projection_matrix.txt thresholds_and_projection.txt 2 1``.
### Finalizing the Index
The final step is to compute the idf-weights for each image in the database, which will be stored in a separate file.
The commandline tool designed for this purpose is ``compute_index_weights``, which requires only one parameters ``index`` that specifies the filename of the inverted index created via ``build_partial_index``. 

An example call is ``compute_index_weights inverted_index.bin``, which will create a binary file ``inverted_index.bin.weights`` that contains the weights.

## Querying an Inverted Index
Given the inverted index constructed as described above, we can now query the index with a set of query images using the executable ``query``. For this, a binary file containing the features and visual word assignments is required for each query image. The file format is the same as for building the query with the exception that the 5 closest words need to be stored in the files for each feature in the query image. Also, notice that the command line tools ``hesaff_sift_to_binary_root_sift`` and ``compute_word_assignments`` are provided for convinience but were not used to compute the visual word assignments used in the paper.

In addition to the query images and their descriptor files, a geo-coordinate is required for each database image. These geo-coordinates should be given in a coordinate system where the Euclidean distance between two points corresponds to a distance in meters.

In order to query an index ``inverted_index.bin``, the executable ``query`` is used. It has the following parameters:
* ``list_sifts``: A text file containing the list of filename prefixes for the query images. For each query image ``query/a.jpg`` a line ``query/a.`` should be included in the file. The program assumes that the binary descriptor file of ``a.jpg`` is stored under ``query/a.bin``.
* ``list_db``: A text file containing the filenames of the database images. The ordering in this file has to be the same as in the ``list_bin`` file used for ``build_partial_index``.
* ``index``: The filename of the inverted index that should be used.
* ``outfile_prefix``: ``query`` will generate 6 text files, one for each type of re-ranking (based on the retrieval scores, the number of inliers, the effective inlier count, the inter-image geometric burstiness measure, the inter-place burstiness measure, and the inter-place plus popularity measure). The files will be stored as ``outfile_prefix.retrieval_ranking.txt``, ``outfile_prefix.inlier_count_ranking.txt``, etc. Each line has the format ``query_name.jpg database_name score``, where ``query_name.`` is one of the filenames from ``list_sifts`` and ``database_name`` is the filename of a database image (extracted from ``list_db``). ``score`` is the corresponding re-ranking score for the two images. For each query image, the database images are giving in decreasing order of re-ranking scores and the pairs are ordered based on the query images (i.e., first all retrieved images for the first query image are reported in descending order of importance, than the same for the second image, etc.).
* ``sift_type``: Same as above.
* ``pos_db``: A text file containing a 2D geo-location for each database image, one line per database image. The file needs to have the same ordering as ``list_db``.

An example call to ``query`` is

``query list_my_queries.txt list_my_db_images.txt inverted_index.bin results/my_dataset 2 list_my_db_image_positions.txt`` .

# Disclaimer
This is a revised version of the implementation used for the CVPR 2016 paper on geometric burstiness. Compared to the original implementation used for the experiments presented in the paper, a bug preventing that votes are cast for the last image in an inverted file was fixed (we thank Johannes L. Schönberger for pointing out the bug). Originally, the re-ranking code was implemented in Matlab. As such, the results obtained with our re-implementation in C++ might differ slightly.

Notice that some parameters of our method, such as the number of nearest words (5) used in ``query`` or the number of visual words expected in the vocabulary (200k) in ``compute_hamming_thresholds`` are hardcoded.

The main intention for releasing this software is to stimulate further research on visual place recognition and to allow other researchers to compare their method against our approach. While we might update the software from time to time, this package will not be regularly maintained or updated.

# Questions & Suggestions
For questions or suggestions, please contact Torsten Sattler (torsten.sattler@inf.ethz.ch).


# Acknowledgements
This work was supported by Google’s Project Tango and EC Horizon 2020 project REPLICATE (no. 687757). The authors thank Relja Arandjelović for his invaluable help with the DisLoc algorithm.
