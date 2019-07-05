# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy, memmove
from libc.string cimport memset
from libc.math cimport fabs, nearbyint
# To verbose tensor basis criterion
from libc.stdio cimport printf

import numpy as np
cimport numpy as np
np.import_array()
# For early stop in proxy_impurity_improvement_pipeline()
cdef double INFINITY = np.inf

from ._utils cimport log
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray
from ._utils cimport WeightedMedianCalculator
# Least-squares of Ax = b with nogil capability
cimport scipy.linalg.cython_lapack as cython_lapack

cdef class Criterion:
    """Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics.
    """

    def __dealloc__(self):
        """Destructor."""

        free(self.sum_total)
        free(self.sum_left)
        free(self.sum_right)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end,
                  # Additional kwarg
                  DOUBLE_t[:, :, ::1] tb=None) nogil except -1:
        """Placeholder for a method which will initialize the criterion.
        For tensor basis criterion, tb, bij need to be supplied and y (which is derived best g) is not used.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables.
            If in tensor basis criterion, then y is
            anisotropy tensor bij, n_samples x 9 components, used for tensor basis criterion
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample
        weighted_n_samples : DOUBLE_t
            The total weight of the samples being considered
        samples : array-like, dtype=DOUBLE_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node
        tb : array-like, dtype=DOUBLE_t, or None
            Tensor basis matrix T, n_samples x 9 components x 10 bases, used for tensor basis criterion
        """

        pass

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start.

        This method must be implemented by the subclass.
        """

        pass

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end.

        This method must be implemented by the subclass.
        """
        pass

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        This updates the collected statistics by moving samples[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos : SIZE_t
            New starting index position of the samples in the right child
        """

        pass

    cdef double node_impurity(self) nogil:
        """Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of samples[start:end]. This is the
        primary function of the criterion class.
        """

        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of samples[start:pos] + the impurity
        of samples[pos:end].

        Parameters
        ----------
        impurity_left : double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : double pointer
            The memory address where the impurity of the right child should be
            stored
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """Placeholder for storing the node value.

        Placeholder for a method which will compute the node value
        of samples[start:end] and save the value into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address where the node value should be stored.
        """

        pass

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef double impurity_improvement(self, double impurity) nogil:
        """Compute the improvement in impurity

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity : double
            The initial impurity of the node before the split

        Return
        ------
        double : improvement in impurity after the split occurs
        """

        # TODO: L2 regularization is not implemented here
        cdef double impurity_left
        cdef double impurity_right

        # children_impurity() takes pointers which point to the address of impurity_*
        self.children_impurity(&impurity_left, &impurity_right)

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity - (self.weighted_n_right /
                             self.weighted_n_node_samples * impurity_right)
                          - (self.weighted_n_left /
                             self.weighted_n_node_samples * impurity_left)))

    cdef double* _reconstructAnisotropyTensor(self, SIZE_t pos1, SIZE_t pos2, double alpha=0., double cap=INFINITY) nogil:
        """
        Placeholder for a method in RegressionCriterion(Criterion).
        
        Compute 10 tensor basis coefficients g and deviatoric summed square error se_dev that replaces the 
        functionality of sum_total/left/right.
        
        Parameters
        ----------
        pos1 : SIZE_t
            The starting index of the sorted samples
        pos2 : SIZE_t
            The end index of the sorted samples
        alpha : double
            The L2 regularization fraction to penalize large optimal g, from
            min_g||bij - (1 + alpha)Tij*g||^2
        cap : double
            Cap of g magnitude during LS fit

        Return
        ------
        double* : Pointer to the solved g array, used by BestSplitter(Splitter)
        """

        pass

    cdef double proxy_impurity_improvement_pipeline(self, double split_pos,
                                                    SIZE_t min_samples_leaf,
                                                    double min_weight_leaf, double
    alpha_g_split=0.) nogil:
        """
        Placeholder for a method in RegressionCriterion(Criterion).
        
        Compute the proxy impurity improvement 
        by putting Criterion.update() and Criterion.proxy_impurity_improvement() in a pipeline.
        In addition, check for minimum samples and minimum sample weights in a leaf. 
        Used in BestSplitter.node_split().
        
        Parameters
        ----------
        split_pos : double
            The splitting double index of samples at this node, converted to double.
        min_samples_leaf : SiZE_t
            Minimum samples acceptable in a leaf, should be supplied from Splitter.
            Can end function prematurely and return -np.inf.
        min_weight_leaf : double
            Minimum sum of sample weights acceptable in a leaf, should be supplied from Splitter.
            Can end function prematurely and return -np.inf. 
        alpha_g_split : double, optional (default=0.)
            Small positive L2 regularization coefficient to penalize large optimal g when finding the best split,
            with min_split(proxy_impurity_improvement - ||alpha_g_split*g||^2)
            Should be supplied from Splitter.

        Return
        ------
        double : Pseudo impurity improvement or -123456789.0 if one of checks failed. 
        Higher is better.
        Actual impurity improvement is calculated in impurity_improvement() when best split has been found.
        """

        pass


cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    def __cinit__(self, SIZE_t n_outputs,
                  np.ndarray[SIZE_t, ndim=1] n_classes):
        """Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        """

        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Count labels for each output
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL
        self.n_classes = NULL

        safe_realloc(&self.n_classes, n_outputs)

        cdef SIZE_t k = 0
        cdef SIZE_t sum_stride = 0

        # For each target, set the number of unique classes in that target,
        # and also compute the maximal stride of all targets
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > sum_stride:
                sum_stride = n_classes[k]

        self.sum_stride = sum_stride

        cdef SIZE_t n_elements = n_outputs * sum_stride
        self.sum_total = <double*> calloc(n_elements, sizeof(double))
        self.sum_left = <double*> calloc(n_elements, sizeof(double))
        self.sum_right = <double*> calloc(n_elements, sizeof(double))

        if (self.sum_total == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""
        free(self.n_classes)

    def __reduce__(self):
        return (type(self),
                (self.n_outputs,
                 sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight, double weighted_n_samples,
                  SIZE_t* samples, SIZE_t start, SIZE_t end,
                  # Extra kwarg, not used
                  DOUBLE_t[:, :, ::1] tb=None) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
        children samples[start:start] and samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            The target stored as a buffer for memory efficiency
        sample_weight : array-like, dtype=DTYPE_t
            The weight of each sample
        weighted_n_samples : SIZE_t
            The total weight of all samples
        samples : array-like, dtype=SIZE_t
            A mask on the samples, showing which ones we want to use
        start : SIZE_t
            The first sample to use in the mask
        end : SIZE_t
            The last sample to use in the mask
        """

        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0

        for k in range(self.n_outputs):
            memset(sum_total + offset, 0, n_classes[k] * sizeof(double))
            offset += self.sum_stride

        for p in range(start, end):
            i = samples[p]
            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0
            if sample_weight != NULL:
                w = sample_weight[i]

            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                c = <SIZE_t> self.y[i, k]
                sum_total[k * self.sum_stride + c] += w

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memset(sum_left, 0, n_classes[k] * sizeof(double))
            memcpy(sum_right, sum_total, n_classes[k] * sizeof(double))

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.end

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memset(sum_right, 0, n_classes[k] * sizeof(double))
            memcpy(sum_left, sum_total, n_classes[k] * sizeof(double))

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : SIZE_t
            The new ending position for which to move samples from the right
            child to the left child.
        """
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    label_index = k * self.sum_stride + <SIZE_t> self.y[i, k]
                    sum_left[label_index] += w

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    label_index = k * self.sum_stride + <SIZE_t> self.y[i, k]
                    sum_left[label_index] -= w

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                sum_right[c] = sum_total[c] - sum_left[c]

            sum_right += self.sum_stride
            sum_left += self.sum_stride
            sum_total += self.sum_stride

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] and save it into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """

        cdef double* sum_total = self.sum_total
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memcpy(dest, sum_total, n_classes[k] * sizeof(double))
            dest += self.sum_stride
            sum_total += self.sum_stride


cdef class Entropy(ClassificationCriterion):
    r"""Cross Entropy impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end], using the cross-entropy criterion."""

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double entropy = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_total[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * log(count_k)

            sum_total += self.sum_stride

        return entropy / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node
        impurity_right : double pointer
            The memory address to save the impurity of the right node
        """

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double entropy_left = 0.0
        cdef double entropy_right = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_left[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= count_k * log(count_k)

                count_k = sum_right[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= count_k * log(count_k)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        impurity_left[0] = entropy_left / self.n_outputs
        impurity_right[0] = entropy_right / self.n_outputs


cdef class Gini(ClassificationCriterion):
    r"""Gini Index impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} count_k (1 - count_k)
              = 1 - \sum_{k=0}^{K-1} count_k ** 2
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end] using the Gini criterion."""


        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double gini = 0.0
        cdef double sq_count
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            sq_count = 0.0

            for c in range(n_classes[k]):
                count_k = sum_total[c]
                sq_count += count_k * count_k

            gini += 1.0 - sq_count / (self.weighted_n_node_samples *
                                      self.weighted_n_node_samples)

            sum_total += self.sum_stride

        return gini / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]) using the Gini index.

        Parameters
        ----------
        impurity_left : DTYPE_t
            The memory address to save the impurity of the left node to
        impurity_right : DTYPE_t
            The memory address to save the impurity of the right node to
        """

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double gini_left = 0.0
        cdef double gini_right = 0.0
        cdef double sq_count_left
        cdef double sq_count_right
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            sq_count_left = 0.0
            sq_count_right = 0.0

            for c in range(n_classes[k]):
                count_k = sum_left[c]
                sq_count_left += count_k * count_k

                count_k = sum_right[c]
                sq_count_right += count_k * count_k

            gini_left += 1.0 - sq_count_left / (self.weighted_n_left *
                                                self.weighted_n_left)

            gini_right += 1.0 - sq_count_right / (self.weighted_n_right *
                                                  self.weighted_n_right)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        impurity_left[0] = gini_left / self.n_outputs
        impurity_right[0] = gini_right / self.n_outputs


cdef class RegressionCriterion(Criterion):
    r"""Abstract regression criterion.

    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples, bint tb_mode=0, bint tb_verbose=0, double alpha_g_fit=0.,
                  double g_cap=INFINITY):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted

        n_samples : SIZE_t
            The total number of samples to fit on

        tb_mode : bint, optional (default=0)
            On/off flag of tensor basis criterion, based on whether both tb and bij are supplied as inputs in
            DecisionTreeRegressor.fit()

        tb_verbose : bint, optional (default=0)
            Whether to verbose tensor basis criterion related information for verbose

        alpha_g : double, optional (default=0.)
            Small positive L2 regularization fraction
            to penalize large g during LS fit of min_g(bij - (1 + alpha)Tij*g).
            If 0, then no regularization during the LS fit to find optimal g is done.
        """

        # Default values
        self.sample_weight = NULL
        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        # sq_sum_total is the non-deviatoric part of SE
        self.sq_sum_total = 0.0
        # se_dev is the deviatoric SE and will replace sum_* n_outputs array in tensor basis criterion
        self.se_dev = 0.
        # L2 regularization fraction for LS fit of g
        self.alpha_g_fit = alpha_g_fit
        # Cap of g magnitude during LS fit
        self.g_cap = g_cap

        # Tensor basis criterion switch and verbose option
        self.tb_mode, self.tb_verbose = tb_mode, tb_verbose

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL

        # Tensor basis criterion related arrays
        self.tb_node, self.tb_transpose_node = NULL, NULL
        self.bij_node, self.bij_hat_node = NULL, NULL
        self.g_node_tmp, self.g_node = NULL, NULL
        self.g_node_l, self.g_node_r = NULL, NULL
        # Outputs of least-squares fit of Ax = b dgelsd().
        # ls_s is singular value of A in decreasing order.
        # On exit, if INFO = 0, ls_work(1) returns the optimal ls_lwork
        self.ls_s, self.ls_work, self.ls_iwork= NULL, NULL, NULL

        # Allocate memory for the accumulators.
        # In tensor basis criterion, functionality of sum_* is mostly changed but usage not
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))

        # Allocate tensor basis related arrays over all samples in current tree,
        # represented by 1D flattened array of memory addresses.
        # The values are initialized as 0
        if tb_mode:
            self.tb_transpose_node = <double*> calloc(n_samples*n_outputs*10, sizeof(double))
            self.bij_node = <double*> calloc(n_samples*n_outputs, sizeof(double))
            self.bij_hat_node = <double*> calloc(n_samples*n_outputs, sizeof(double))
            self.g_node_tmp = <double*> calloc(10, sizeof(double))
            self.g_node = <double*> calloc(10, sizeof(double))
            self.g_node_l = <double*> calloc(10, sizeof(double))
            self.g_node_r = <double*> calloc(10, sizeof(double))

            # dgelsd() least-squares fit of Ax = b related outputs.
            # Dimension min(A's row, col)
            self.ls_s = <double*> calloc(10, sizeof(double))
            # Dimension max(1, lwork), where lwork is at least
            # 12col + 2col*SMLSIZ + 8col*NLVL + col*nrhs + (SMLSIZ + 1)**2 if row >= col,
            # SMLSIZ is the maximum size of subproblems at bottom of computation tree (usually 25, take 26),
            # NLVL = max(0, int(log_2(min(row, col)/(SMLSIZ + 1))) + 1) = 0 take 1,
            # => 12*10 + 2*10*26 + 8*10*1 + 10*1 + (26 + 1)**2 = 1459, take 1500
            self.ls_work = <double*> calloc(1500, sizeof(double))
            # Dimension max(1, 3*min(A's row, col)*NVLV + 11*min(A's row, col)), see above,
            # = max(1, 3*10*1 + 11*10) = 140
            self.ls_iwork = <int*> calloc(140, sizeof(int))

        if (self.sum_total == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

        if tb_mode:
            # Raise memory error for large tensor basis related arrays
            if (self.tb_transpose_node == NULL or
                self.bij_node == NULL or
                self.bij_hat_node == NULL or
                self.ls_work == NULL):
                raise MemoryError()

    def __dealloc__(self):
        """Destructor. Additional destruction of newly introduced tensor basis arrays. Does nothing if array is NULL?"""
        if self.tb_mode:
            # Deallocate tensor basis criterion related memory blocks
            free(self.tb_transpose_node)
            free(self.bij_node)
            free(self.bij_hat_node)
            free(self.g_node_tmp)
            free(self.g_node)
            free(self.g_node_l)
            free(self.g_node_r)
            # Deallocate S and work array used in dgelsd() least-squares fit of Ax = b
            free(self.ls_s)
            free(self.ls_work)
            free(self.ls_iwork)
            if self.tb_verbose:
                printf("\n   Array memory blocks are destroyed ")

    def __reduce__(self):
        # For pickling. Addition args of tb_mode, tb_verbose,
        # alpha_g_fit, g_cap
        return (type(self), (self.n_outputs, self.n_samples, self.tb_mode, self.tb_verbose, self.alpha_g_fit,
                             self.g_cap),
                self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end,
                  DOUBLE_t[:, :, ::1] tb=None) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end].
        For tensor basis criterion only:
        tb is n_samples x n_outputs x 10 bases or None.
        ::1 means C-contiguous
        """
        # Initialize fields
        # If in tensor basis criterion, then y is bij
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.
        # Extra initialization of tensor basis Tij
        self.tb = tb

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0

        self.sq_sum_total = 0.0

        if self.tb_mode:
            # Calculate sum_total[n_outputs], i.e. deviatoric SE for samples from start to end index at this node.
            # Results are stored in self.se_dev, and g stored in self.g_node,
            # later stored in dest of Criterion.node_value() as future prediction values
            if self.tb_verbose:
                printf("\n   Evaluating deviatoric SE for samples[start:end] of size %d ", self.n_node_samples)

            _ = self._reconstructAnisotropyTensor(start, end, self.alpha_g_fit, self.g_cap)
            # Then store 10 g for this node to be saved as prediction values of this node.
            # memcpy(dest, src) doesn't alter content dest is pointing to
            # even if src's content is changed later
            memcpy(self.g_node, self.g_node_tmp, 10*sizeof(double))
            # Then set self.sum_total n_outputs array to self.se_dev scalar.
            # memset only accept int value for initialization thus not using it
            for k in range(self.n_outputs):
                self.sum_total[k] = self.se_dev

        # Else if default sum_total behavior, then initialize sum_total to 0 integer for "+=" later
        else:
            memset(self.sum_total, 0, self.n_outputs * sizeof(double))

        for p in range(start, end):
            # Original unsorted sample index at this node, for sorted sample index in [start, end)
            i = samples[p]

            # Sample weights are 0 for unsampled points in RF bootstrap
            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                # If not in tensor basis mode, then calculate sum_total[n_outputs] (a.k.a. n_samples*y_hat) as usual
                if not self.tb_mode:
                    self.sum_total[k] += w_y_ik

                # In tensor basis case, this is
                # sum^n_samples[sum^n_outputs(bij^2)],
                # a.k.a. non-deviatoric part of SE
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef double* _reconstructAnisotropyTensor(self, SIZE_t pos1, SIZE_t pos2, double alpha=0., double cap=INFINITY) nogil:
        """For tensor basis MSE criterion, where Tij is supplemented,
        calculate the deviatoric summed square error self.se_dev scalar of sorted samples[pos1:pos2]
        to replace the functionality of self.sum_total/left/right n_outputs array.
        Given Tij and bij (which is y), summed square error is 
        SE = sum^n_samples||bij - sum^n_bases(Tij*g)||^2
           = sum^n_samples[sum^n_components(bij^2)]
             - sum^n_samples[sum^n_components(2bij*bij_hat + bij_hat^2)]
           = sum^n_samples[sum^n_components(bij^2)] - se_dev.
        alpha is L2 regularization fraction to prevent g from getting too large, min_g||bij - (1 + alpha)Tij*g||^2.
        Although optimal g is not affect by sample weights, SE is. 
        
        First, collect bij and Tij that lies in sorted samples[pos1:pos2] of this node.
        Next, find one set of 10 g that best fit Tij*g = bij for all samples in this node via LS fit. 
            1). Compress Tij[n_samples x n_outputs x 10] to Tij[n_samples*n_outputs x 10]
            2). Find 10 g by LS fit of the over-determined linear system of 
            Tij[n_samples*n_outputs x 10]*g[10 x 1] = bij[n_samples*n_outputs x 1]
        Then, compute the reconstructed bij_hat of each sample in this node using 
        bij_hat[n_samples*n_outputs x 1] = Tij[n_samples*n_outputs x 10]*g[10 x 1].
        Finally, replace mostly the functionality of self.sum_total/sum_left/sum_right n_outputs array
        from self.sum_*[n_outputs] = sum^n_samples(y[n_outputs]) to a deviatoric SE scalar,
        self.se_dev = sum^n_samples[sum^n_outputs(2bij*bij_hat - bij_hat^2)],
        where SE = sum^n_samples[sum^n_outputs(bij^2)] - se_dev,
        and MSE = SE/n_samples."""

        # Index array (samples) and variables definition
        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t w = 1.
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef DOUBLE_t weighted_n_node_samples = 0.
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t i1, i2, p, p0, i
        # abs() not possible for nogil
        cdef SIZE_t n_samples = pos2 - pos1
        cdef SIZE_t nelem_bij_node = n_samples*n_outputs
        # Number of rows in Tij_node and bij_node, has to >= ncol
        # TODO: (Probably unnecessary) In case of only one sample, i.e. Tij[9 or 6 x 10] and bij[9 or 6 x 1],
        #  the system is under-determined,
        #  fill 0 to Tij and bij until they are [10 x 10] and [10 x 1] to avoid any functional problem.
        cdef int row = max(nelem_bij_node, 10)
        # L1 regularization to penalize large 10 g after LS-fit
        cdef DOUBLE_t amplifier = 1. + alpha
        # Verbose option for debugging
        cdef bint* verbose = &self.tb_verbose

        # Reallocate the memory size of dynamic size arrays to number of Tij/bij elements in this node,
        # minimum [10 x 10] (e.g. Tij) or [10 x 1] (e.g. bij).
        # realloc() tries to preserve old memory as much as possible,
        # i.e. old addresses' values are untouched and will be overwritten.
        # Not realloc self.ls_work since it doesn't have to be resized.
        # &self.bij_node equals the address of self.bij_node pointer
        # since safe_realloc takes the address of a pointer.
        safe_realloc(&self.tb_transpose_node, row*10)
        safe_realloc(&self.bij_node, row)
        safe_realloc(&self.bij_hat_node, row)

        # Initialize (or reset) memory block of flattened array that involves "+=", to 0 integer
        memset(self.bij_hat_node, 0, row*sizeof(double))

        # Tensor basis related flattened array pointers for samples from pos1 to pos2 at this node.
        # Pointer assignments here are mainly for convenience of dumping "self.".
        # However, ptr[i] will point to the value at ith address in val. When bij_node is changed, self.bij_node is too
        cdef DOUBLE_t* tb_transpose_node = self.tb_transpose_node
        cdef DOUBLE_t* bij_node = self.bij_node
        cdef DOUBLE_t* g_node_tmp = self.g_node_tmp
        cdef DOUBLE_t* bij_hat_node = self.bij_hat_node
        # The following is only used when debugging
        cdef DOUBLE_t se = 0.
        cdef DOUBLE_t mse

        # Least-squares fit dgelsd() related variables
        cdef int col = 10
        cdef int nrhs = 1
        cdef int lda = row
        cdef int ldb = row
        cdef DOUBLE_t rcond = -1
        cdef int rank, info
        # Dimension of ls_work, at least
        # 12col + 2col*SMLSIZ + 8col*NLVL + col*nrhs + (SMLSIZ + 1)**2 if row >= col,
        # SMLSIZ is the maximum size of subproblems at bottom of computation tree (usually 25, take 26),
        # NLVL = max(0, int(log_2(min(row, col)/(SMLSIZ + 1))) + 1) = 0 take 1,
        # => 12*10 + 2*10*26 + 8*10*1 + 10*1 + (26 + 1)**2 = 1459, take 1500
        cdef int lwork = 1500

        # Reset deviatoric SE scalar to 0, replacing functionality of self.sum_* n_outputs array
        self.se_dev = 0.
        # Flatten Tij, bij and pick up Tij, bij for samples from pos1 to pos2 index at current node,
        # where p is n_samples (3rd axis), i1 is output (row), i2 is basis (column)
        for p in range(pos1, pos2):
            # Actual index of the original unsorted X
            i = samples[p]
            # Sample weight of unpicked samples is 0 in TBRF bootstrap with replacement.
            # This means LS fit becomes w*Tij*g = w*bij -- g unaffected
            if sample_weight != NULL:
                w = sample_weight[i]

            # Sample index but starts at 0 instead
            p0 = p - pos1
            for i1 in range(n_outputs):
                for i2 in range(10):
                    # # n_samples*n_outputs x 10 flattened to 1D memory addresses, C-contiguous.
                    # # tb_node's index is in the order of basis -> component -> n_samples
                    # tb_node[p0*n_outputs*10 + i1*10 + i2] = self.tb[i, i1, i2]

                    # C-contiguous flattened Tij^T, also interpreted as Fortran-contiguous Tij, 10 x n_samples*n_outputs.
                    # tb_transpose_node's index is in the order of component -> n_samples -> basis.
                    # In Tij^T matrix, the order of recording is: go through each row of a col then jump to next col
                    tb_transpose_node[nelem_bij_node*i2 + p0*n_outputs + i1] = w*self.tb[i, i1, i2]*amplifier

                # bij at this node, n_samples*n_outputs x 1, will contain g solutions after dgelsd()
                bij_node[p0*n_outputs + i1] = w*self.y[i, i1]

        # Least-squares fit with dgelsd() to solve g from
        # min_g J = ||Tij*g - bij||^2.
        # The solution of 10 g is contained in bij_node after cython_lapack.dgelsd().
        # Note that dgelsd() is written in Fortran thus every matrix needs to be Fortran-contiguous.
        # dgelss() uses Singular Value Decomposition and can solve rank-deficient A matrix.
        # dgelsd() uses SVD with an algorithm based on divide and conquer and is significantly faster than dgelss().
        # Takes Fortran-style pointer arguments.
        # row: number of rows of A; col: number of columns of A; nrhs: number of RHS columns
        # tb_transpose_node: A, destroyed on exit;
        # lda: lead dimension of A, i.e. row;
        # bij_node: b, overwritten by col x 1 solution vector X on exit;
        # ldb: leading dimension of b, max(row, col);
        # ls_s: singular values of A in decreasing order, min(row, col);
        # rcond: if < 0, machine precision is used to determine effective rank of A;
        # ls_work: if INFO = 0 on exit, returns optimal lwork, max(1, lwork);
        # lwork: dimension of ls_work;
        # ls_iwork: integer array max(1, LIWORK), ls_iwork(1) returns minimum LIWORK on exit with INFO = 0;
        # info: 0 means successful exit, if -i, ith arg has illegal value,
        # if i, i off-diagonal elements of an intermediate bi-diagonal form didn't converge to 0.
        cython_lapack.dgelsd(&row, &col, &nrhs,
                             tb_transpose_node, &lda, bij_node, &ldb,
                             self.ls_s, &rcond, &rank,
                             self.ls_work, &lwork, self.ls_iwork, &info)
        if verbose[0]:
            printf("\n       g solved, exit code %d, A effective rank %d, %d rows, %d output(s)", info, rank, row, n_outputs)

        # Since g[10 x 1] is stored in bij_node after dgelsd(),
        # go through each basis and get corresponding g_node at this node
        # TODO: not sure if bij_node is shrinked from row x 1 to 10 x 1
        memcpy(g_node_tmp, bij_node, 10*sizeof(double))
        # Cap g magnitude if a positive cap is provided
        if cap != INFINITY and cap >= 0.:
            for i2 in range(10):
                if fabs(g_node_tmp[i2]) > cap:
                    g_node_tmp[i2] = cap if g_node_tmp[i2] > 0. else -cap

        # Calculate reconstructed bij_hat at this node from Tij and g.
        # Manual dot product by aggregating each column (basis) in each row (component).
        for p in range(pos1, pos2):
            i = samples[p]
            if sample_weight != NULL:
                w = sample_weight[i]

            if verbose[0]:
                weighted_n_node_samples += w

            p0 = p - pos1
            for i1 in range(n_outputs):
                # For each component, perform bij_hat = sum^n_bases(Tij*g)
                for i2 in range(10):
                    # Flattened bij_hat construction complete at this node, n_samples*n_outputs x 1
                    bij_hat_node[p0*n_outputs + i1] += self.tb[i, i1, i2]*g_node_tmp[i2]

                # Replacing (almost) the functionality of self.sum_total/sum_left/sum_right with deviatoric SE
                # Originally, self.sum_total/sum_left/sum_right is sum^n_samples(y).
                # Now, se_dev is the deviatoric SE and is sum^n_samples[sum^n_outputs(2bij*bij_hat - bij_hat^2)],
                # and SE = sum^n_samples[sum^n_outputs(bij^2)] - se_dev;
                # MSE = sum^n_samples[sum^n_outputs(bij^2)]/n_samples - se_dev/n_samples.
                self.se_dev += w*(2.*self.y[i, i1]*bij_hat_node[p0*n_outputs + i1]
                - bij_hat_node[p0*n_outputs + i1]**2.)
                # Calculate SE if debugging is on.
                # Currently aggregating the non-deviatoric part of SE and later removing deviatoric part from it
                if verbose[0]:
                    se += w*self.y[i, i1]**2.

        # If debugging, remove deviatoric SE from non-deviatoric SE to derive accumulative SE of all components
        if verbose[0]:
            se -= self.se_dev
            mse = se/weighted_n_node_samples
            printf("\n       MSE = %8.8f ", mse)

        return g_node_tmp

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_left, 0, n_bytes)
        memcpy(self.sum_right, self.sum_total, n_bytes)

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_right, 0, n_bytes)
        memcpy(self.sum_left, self.sum_total, n_bytes)

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef double* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that, by default,
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known,
        # we are going to update sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        # However in tensor basis criterion,
        #           sum_*[any output] = deviatoric SE scalar of the left/right child node samples,
        # and sum_left + sum_right != sum_total.
        # Thus separate set of g needs to be derived for both left/right child nodes and no shortcut.
        # Default sum_left and sum_right behavior
        if not self.tb_mode:
            # pos has been reset to self.start previously in reset() at start
            # and will be new_pos at the end of this function
            if (new_pos - pos) <= (end - new_pos):
                for p in range(pos, new_pos):
                    i = samples[p]
                    if sample_weight != NULL:
                        w = sample_weight[i]

                    # Calculate sum_left as usual if not in tensor basis mode
                    for k in range(self.n_outputs):
                        sum_left[k] += w * self.y[i, k]

                    self.weighted_n_left += w
            # Else if calculating from new_pos to end is easier than from pos to new_pos
            else:
                # reverse_reset() sets sum_left to sum_total
                self.reverse_reset()
                for p in range(end - 1, new_pos - 1, -1):
                    i = samples[p]
                    if sample_weight != NULL:
                        w = sample_weight[i]

                    for k in range(self.n_outputs):
                        sum_left[k] -= w * self.y[i, k]

                    self.weighted_n_left -= w

            for k in range(self.n_outputs):
                sum_right[k] = sum_total[k] - sum_left[k]

        # Else if tensor basis criterion
        else:
            # First solve for g for left child node samples
            if self.tb_verbose:
                printf("\n      Evaluating deviatoric SE for samples[start:split] of size %d ", (new_pos - self.start))

            # By default, sum_left is accumulative,
            # i.e. sum_left[start:new_pos] is sum_left[start:pos] + sum_left[pos:new_pos],
            # instead of calculating expensive sum_left[start:new_pos] every time new_pos is supplied.
            # However, in tensor basis criterion, sum_left has to be re-calculated in [start:new_pos] every time
            _ = self._reconstructAnisotropyTensor(self.start, new_pos, self.alpha_g_fit, self.g_cap)
            # Assign g_node_tmp to g left to the split as g_node_tmp will soon be overwritten by right child
            memcpy(self.g_node_l, self.g_node_tmp, 10*sizeof(double))
            # Then set sum_left n_outputs array to self.se_dev scalar.
            # memset only accepts int values for initialization thus not using it
            for k in range(self.n_outputs):
                self.sum_left[k] = self.se_dev

            # Also get number of samples in left child node, with weight according to TBRF bootstrap with replacement
            for p in range(self.start, new_pos):
                i = samples[p]
                if sample_weight != NULL:
                    w = sample_weight[i]

                self.weighted_n_left += w

            # Do the same for the right child node samples
            if self.tb_verbose:
                printf("\n      Evaluating deviatoric SE for samples[split:end] of size %d ", (end - new_pos))

            _ = self._reconstructAnisotropyTensor(new_pos, end, self.alpha_g_fit, self.g_cap)
            memcpy(self.g_node_r, self.g_node_tmp, 10*sizeof(double))
            for k in range(self.n_outputs):
                self.sum_right[k] = self.se_dev

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest.
        dest is used as constant prediction when supplying new samples for prediction after training.
        dest at each leaf node is y_pred.
        
        Definition of dest is changed for tensor basis criterion. 
        By default, dest[output] is mean y[output] at this node.
        However, in tensor basis criterion, dest[n_bases] is 
        the best found 10 g that fit this whole node without any splitting.
        
        By default, dest has n_outputs of y. In tensor basis criterion, dest has 10 regardless of y."""

        cdef SIZE_t k

        if not self.tb_mode:
            for k in range(self.n_outputs):
                dest[k] = self.sum_total[k] / self.weighted_n_node_samples

        else:
            # self.g_node is the best 10 g that fit this whole node without any splitting.
            # Thus it's the constant value to be predict if any new samples fall and end up in this node.
            # Using memmove instead memcpy
            # since memcpy fails when object pointed by src overlaps with object pointed by dest
            memmove(dest, self.g_node, 10*sizeof(double))

    cdef double proxy_impurity_improvement_pipeline(self, double split_pos,
                                                    SIZE_t min_samples_leaf,
                                                    double min_weight_leaf, double
    alpha_g_split=0.) nogil:
        """
        Pipeline of Criterion.update(pos) -> Criterion.proxy_impurity_improvement(),
        with checks of min_samples_leaf and min_weight_leaf for early stop.
        Used in BestSplitter.node_split().
        Since higher is better in the output, an inverse is taken during Brent optimization to minimize, 
        if using Brent optimization.
        """
        cdef double current_proxy_improvement
        cdef SIZE_t i
        cdef double g_l_penalty = 0.
        cdef double g_r_penalty = 0.
        # Cast int type on the double sample split index, take nearest integer (still float)
        cdef SIZE_t split_pos_tmp = <SIZE_t> nearbyint(split_pos)
        # Reject if min_samples_leaf is not guaranteed.
        # This should never be triggered in "brent" split_finder
        if (((split_pos_tmp - self.start) < min_samples_leaf) or
                ((self.end - split_pos_tmp) < min_samples_leaf)):
            return -INFINITY

        # sum_left/right, pos and weighted_n_left/right
        # are updated using current split index double
        self.update(split_pos_tmp)
        # Reject if min_weight_leaf is not satisfied.
        # This should never be triggered in "brent" split finder
        if ((self.weighted_n_left < min_weight_leaf) or
                (self.weighted_n_right < min_weight_leaf)):
            return -INFINITY

        # For tensor basis criterion, only sum_left/right from Criterion.update() are useful here,
        # to get pseudo impurity improvement at current split
        current_proxy_improvement = self.proxy_impurity_improvement()

        # If L2 regularization coefficient of the minimum split objective function is provided,
        # perform Frobenius L2-norm on ||alpha*g||^2
        if self.tb_mode and alpha_g_split > 0.:
            # Accumulative penalty of all tensor bases
            for i in range(10):
                g_l_penalty += (alpha_g_split*self.g_node_l[i])**2.
                g_r_penalty += (alpha_g_split*self.g_node_r[i])**2.

            # Since impurity improvement is higher is better,
            # penalize it if optimal g_l and/or g_r are too large
            current_proxy_improvement -= (g_l_penalty + g_r_penalty)

        return current_proxy_improvement


cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.

        MSE = var_left + var_right
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end].
           This method incorporates a tensor basis criterion if Tij is given along with X, 
           and y (bij) in DecisionTreeRegressor.fit()."""

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        # If tensor basis criterion, then self.sum_*[any index] is
        # sum^n_samples[sum^n_outputs(2bij*bij_hat - bij_hat^2)],
        # and MSE = (sq_sum_total - sum_total)/n_samples
        if self.tb_mode:
            impurity -= sum_total[0]/self.weighted_n_node_samples

        # Otherwise, MSE = [sq_sum_total - sum^n_outputs(y_bar^2)]/n_samples
        else:
            for k in range(self.n_outputs):
                impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

            impurity /= self.n_outputs

        return impurity

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        
        The calculation for proxy_impurity_improvement is slightly different 
        based how sum_left/right is defined in tensor basis criterion vs. any other criterion.
        Nonetheless, the definition of proxy_impurity_improvement remains unchanged.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        # By default, proxy_impurity_left[k] is
        # weighted_n_left*y_bar[k]^2 = sum_left[k]^2/weighted_n_left,
        # and proxy_impurity_left = sum_1^n_outputs(proxy_impurity_left[k]), higher is better.
        #
        # However, in tensor basis criterion, proxy_impurity_left is deviatoric SE_left directly.
        # Therefore, proxy_impurity_left = sum_left[any index], higher is better
        if not self.tb_mode:
            for k in range(self.n_outputs):
                proxy_impurity_left += sum_left[k] * sum_left[k]
                proxy_impurity_right += sum_right[k] * sum_right[k]

            return (proxy_impurity_left / self.weighted_n_left +
                    proxy_impurity_right / self.weighted_n_right)

        else:
            proxy_impurity_left = sum_left[0]
            proxy_impurity_right = sum_right[0]

            return proxy_impurity_left + proxy_impurity_right

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
       left child (samples[start:pos]) and the impurity the right child (samples[pos:end]).
       The calculation of impurity_left/right is different for tensor basis criterion since the definition of 
       sum_left/right in it is slightly altered.
       Nonetheless, the definition of sq_sum_left/right as well as impurity_left/right remain unchanged."""

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        # By the time this function gets called in BestSplitter.node_split,
        # self.pos is best.pos, right after Criterion.update(best.pos)
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef DOUBLE_t y_ik

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        for p in range(start, pos):
            i = samples[p]

            # Sample weight is 0 for unsampled points in RF bootstrap
            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                # Definition of sq_sum_* remains unchanged in tensor basis criterion
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left
        # [0] of this pointer accesses its value
        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        # Just like in MSE.node_impurity(), since definition of sum_* is slightly changed,
        # impurity_* calculation is too, for tensor basis criterion
        if not self.tb_mode:
            for k in range(self.n_outputs):
                impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
                impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0

            impurity_left[0] /= self.n_outputs
            impurity_right[0] /= self.n_outputs

        else:
            impurity_left[0] -= sum_left[0]/self.weighted_n_left
            impurity_right[0] -= sum_right[0]/self.weighted_n_right

cdef class MAE(RegressionCriterion):
    r"""Mean absolute error impurity criterion

       MAE = (1 / n)*(\sum_i |y_i - f_i|), where y_i is the true
       value and f_i is the predicted value."""
    def __dealloc__(self):
        """Destructor."""
        free(self.node_medians)

    cdef np.ndarray left_child
    cdef np.ndarray right_child
    cdef DOUBLE_t* node_medians

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples,
                  bint tb_mode=0, bint tb_verbose=0, double alpha_g_fit=0.,
                  double g_cap=INFINITY):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted

        n_samples : SIZE_t
            The total number of samples to fit on

        tb_mode : bint, optional (default=0)
            On/off flag of tensor basis criterion, based on whether tb is supplied as input in
            DecisionTreeRegressor.fit()

        tb_verbose : bint, optional (default=0)
            Whether to verbose tensor basis criterion related information for verbose
        """

        # Default values
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        # L2 regularization fraction to penalize large optimal g from LS fit
        self.alpha_g_fit = alpha_g_fit
        # Magnitude cap of found 10 optimal tensor basis coefficients g
        self.g_cap = g_cap

        # Tensor basis criterion switch
        self.tb_mode, self.tb_verbose = tb_mode, tb_verbose
        # TODO: tensor basis calculations for MAE
        #  Off at the moment
        if tb_mode:
            printf('\n [Not Implemented] MAE for tensor basis criterion has not been implemented!\n')

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.node_medians = NULL

        # Allocate memory for the accumulators
        safe_realloc(&self.node_medians, n_outputs)

        self.left_child = np.empty(n_outputs, dtype='object')
        self.right_child = np.empty(n_outputs, dtype='object')
        # initialize WeightedMedianCalculators
        for k in range(n_outputs):
            self.left_child[k] = WeightedMedianCalculator(n_samples)
            self.right_child[k] = WeightedMedianCalculator(n_samples)

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end,
                  DOUBLE_t[:, :, ::1] tb=None) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""

        cdef SIZE_t i, p, k
        cdef DOUBLE_t w = 1.0

        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.
        # Extra initialization of tensor basis Tij, anisotropy tensor bij
        self.tb = tb

        # Wtf double **... o.O
        cdef void** left_child
        cdef void** right_child

        left_child = <void**> self.left_child.data
        right_child = <void**> self.right_child.data

        for k in range(self.n_outputs):
            (<WeightedMedianCalculator> left_child[k]).reset()
            (<WeightedMedianCalculator> right_child[k]).reset()

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                # push method ends up calling safe_realloc, hence `except -1`
                # push all values to the right side,
                # since pos = start initially anyway
                (<WeightedMedianCalculator> right_child[k]).push(self.y[i, k], w)

            self.weighted_n_node_samples += w
        # calculate the node medians
        for k in range(self.n_outputs):
            self.node_medians[k] = (<WeightedMedianCalculator> right_child[k]).get_median()

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        cdef SIZE_t i, k
        cdef DOUBLE_t value
        cdef DOUBLE_t weight

        cdef void** left_child = <void**> self.left_child.data
        cdef void** right_child = <void**> self.right_child.data

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start

        # reset the WeightedMedianCalculators, left should have no
        # elements and right should have all elements.

        for k in range(self.n_outputs):
            # if left has no elements, it's already reset
            for i in range((<WeightedMedianCalculator> left_child[k]).size()):
                # remove everything from left and put it into right
                (<WeightedMedianCalculator> left_child[k]).pop(&value,
                                                               &weight)
                # push method ends up calling safe_realloc, hence `except -1`
                (<WeightedMedianCalculator> right_child[k]).push(value,
                                                                 weight)
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end

        cdef DOUBLE_t value
        cdef DOUBLE_t weight
        cdef void** left_child = <void**> self.left_child.data
        cdef void** right_child = <void**> self.right_child.data

        # reverse reset the WeightedMedianCalculators, right should have no
        # elements and left should have all elements.
        for k in range(self.n_outputs):
            # if right has no elements, it's already reset
            for i in range((<WeightedMedianCalculator> right_child[k]).size()):
                # remove everything from right and put it into left
                (<WeightedMedianCalculator> right_child[k]).pop(&value,
                                                                &weight)
                # push method ends up calling safe_realloc, hence `except -1`
                (<WeightedMedianCalculator> left_child[k]).push(value,
                                                                weight)
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef void** left_child = <void**> self.left_child.data
        cdef void** right_child = <void**> self.right_child.data

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i, p, k
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # We are going to update right_child and left_child
        # from the direction that require the least amount of
        # computations, i.e. from pos to new_pos or from end to new_pos.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    # remove y_ik and its weight w from right and add to left
                    (<WeightedMedianCalculator> right_child[k]).remove(self.y[i, k], w)
                    # push method ends up calling safe_realloc, hence except -1
                    (<WeightedMedianCalculator> left_child[k]).push(self.y[i, k], w)

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    # remove y_ik and its weight w from left and add to right
                    (<WeightedMedianCalculator> left_child[k]).remove(self.y[i, k], w)
                    (<WeightedMedianCalculator> right_child[k]).push(self.y[i, k], w)

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        self.pos = new_pos
        return 0

    cdef void node_value(self, double* dest) nogil:
        """Computes the node value of samples[start:end] into dest."""

        cdef SIZE_t k
        for k in range(self.n_outputs):
            dest[k] = <double> self.node_medians[k]

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]"""

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t i, p, k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t impurity = 0.0

        for k in range(self.n_outputs):
            for p in range(self.start, self.end):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                impurity += fabs(self.y[i, k] - self.node_medians[k]) * w

        return impurity / (self.weighted_n_node_samples * self.n_outputs)

    cdef void children_impurity(self, double* p_impurity_left,
                                double* p_impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end]).
        """

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef SIZE_t start = self.start
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef SIZE_t i, p, k
        cdef DOUBLE_t median
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t impurity_left = 0.0
        cdef DOUBLE_t impurity_right = 0.0

        cdef void** left_child = <void**> self.left_child.data
        cdef void** right_child = <void**> self.right_child.data

        for k in range(self.n_outputs):
            median = (<WeightedMedianCalculator> left_child[k]).get_median()
            for p in range(start, pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                impurity_left += fabs(self.y[i, k] - median) * w
        p_impurity_left[0] = impurity_left / (self.weighted_n_left *
                                              self.n_outputs)

        for k in range(self.n_outputs):
            median = (<WeightedMedianCalculator> right_child[k]).get_median()
            for p in range(pos, end):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                impurity_right += fabs(self.y[i, k] - median) * w
        p_impurity_right[0] = impurity_right / (self.weighted_n_right *
                                                self.n_outputs)


cdef class FriedmanMSE(MSE):
    """Mean squared error impurity criterion with improvement score by Friedman

    Uses the formula (35) in Friedman's original Gradient Boosting paper:

        diff = mean_left - mean_right
        improvement = n_left * n_right * diff^2 / (n_left + n_right)
    """

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double total_sum_left = 0.0
        cdef double total_sum_right = 0.0

        cdef SIZE_t k
        cdef double diff = 0.0

        for k in range(self.n_outputs):
            total_sum_left += sum_left[k]
            total_sum_right += sum_right[k]

        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right)

        return diff * diff / (self.weighted_n_left * self.weighted_n_right)

    cdef double impurity_improvement(self, double impurity) nogil:
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double total_sum_left = 0.0
        cdef double total_sum_right = 0.0

        cdef SIZE_t k
        cdef double diff = 0.0

        for k in range(self.n_outputs):
            total_sum_left += sum_left[k]
            total_sum_right += sum_right[k]

        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right) / self.n_outputs

        return (diff * diff / (self.weighted_n_left * self.weighted_n_right *
                               self.weighted_n_node_samples))
