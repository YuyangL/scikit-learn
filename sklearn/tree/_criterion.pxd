# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# License: BSD 3 clause

# See _criterion.pyx for implementation details.

import numpy as np
cimport numpy as np

from ._tree cimport DTYPE_t          # Type of X
from ._tree cimport DOUBLE_t         # Type of y, sample_weight
from ._tree cimport SIZE_t           # Type for indices and counters
from ._tree cimport INT32_t          # Signed 32 bit integer
from ._tree cimport UINT32_t         # Unsigned 32 bit integer

cdef class Criterion:
    # The criterion computes the impurity of a node and the reduction of
    # impurity of a split on that node. It also computes the output statistics
    # such as the mean in regression and class probabilities in classification.

    # Internal structures
    cdef const DOUBLE_t[:, ::1] y        # Values of y
    # Tensor basis inputs declaration
    cdef DOUBLE_t[:, :, ::1] tb
    cdef DOUBLE_t* sample_weight         # Sample weights

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t start                    # samples[start:pos] are the samples in the left node
    cdef SIZE_t pos                      # samples[pos:end] are the samples in the right node
    cdef SIZE_t end

    cdef SIZE_t n_outputs                # Number of outputs
    cdef SIZE_t n_samples                # Number of samples
    cdef SIZE_t n_node_samples           # Number of samples in the node (end-start)
    cdef double weighted_n_samples       # Weighted number of samples (in total)
    cdef double weighted_n_node_samples  # Weighted number of samples in the node
    cdef double weighted_n_left          # Weighted number of samples in the left node
    cdef double weighted_n_right         # Weighted number of samples in the right node

    cdef double* sum_total          # For classification criteria, the sum of the
                                    # weighted count of each label. For regression,
                                    # the sum of w*y. sum_total[k] is equal to
                                    # sum_{i=start}^{end-1} w[samples[i]]*y[samples[i], k],
                                    # where k is output index.
    cdef double* sum_left           # Same as above, but for the left side of the split
    cdef double* sum_right          # same as above, but for the right side of the split

    # The criterion object is maintained such that left and right collected
    # statistics correspond to samples[start:pos] and samples[pos:end].

    # Methods
    # Added kwargs of tb, bij
    # Since in tensor basis criterion, y is g and changing, y is not constant
    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end,
                  DOUBLE_t[:, :, ::1] tb=*) nogil except -1
    cdef int reset(self) nogil except -1
    cdef int reverse_reset(self) nogil except -1
    cdef int update(self, SIZE_t new_pos) nogil except -1
    cdef double node_impurity(self) nogil
    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil
    cdef void node_value(self, double* dest) nogil
    cdef double impurity_improvement(self, double impurity) nogil
    cdef double proxy_impurity_improvement(self) nogil
    # Additional method to put self.update() and self.proxy_impurity_improvement
    # in a pipeline,
    # with checks of minimum samples in leaf and minimum sample weights in leaf
    cdef double proxy_impurity_improvement_pipeline(self, double split_pos, SIZE_t min_samples_leaf,
                                                    double min_wight_leaf,
                                                    double alpha_g_split=*) nogil
    # Additional method to reconstruct anisotropy tensors
    cdef double* _reconstructAnisotropyTensor(self, SIZE_t pos1, SIZE_t pos2, double alpha=*, double cap=*) nogil

cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    cdef SIZE_t* n_classes
    cdef SIZE_t sum_stride

cdef class RegressionCriterion(Criterion):
    """Abstract regression criterion."""

    cdef double sq_sum_total
    # Tensor basis related declaration that are used in reconstructAnisotropyTensor()
    # Deviatoric SE scalar, used to replace functionality of sum_total/left/right
    cdef double se_dev
    # L2 regularization fraction for LS fit of g to penalize large optimal g
    cdef double alpha_g_fit
    # Cap of g magnitude after LS fit
    cdef double g_cap
    # Tensor basis criterion switch.
    # If tb is provided in DecisionTreeRegressor.fit(), then tb_mode is 1/True
    cdef bint tb_mode
    # Turn on verbose in DecisionTreeRegressor.fit() for debugging of tensor basis criterion
    cdef bint tb_verbose
    # Each pointer has to be declared separately
    cdef double* tb_node
    cdef double* tb_transpose_node
    cdef double* bij_node
    cdef double* bij_hat_node
    # Temporary 10 g at node / left/right child, changing every time depending on split location
    cdef double* g_node_tmp
    # Found 10 g to be stored as future prediction for this node.
    # Although only leaf nodes' g matters, other nodes' g are stored too for inspection/interpretation
    cdef double* g_node
    # Also declare left and right children optimal g of current node
    cdef double* g_node_l
    cdef double* g_node_r

    # dgelsd() related declaration
    cdef double* ls_s
    cdef double* ls_work
    cdef int* ls_iwork
