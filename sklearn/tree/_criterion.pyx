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
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs
from libc.stdio cimport printf

import numpy as np
cimport numpy as np
np.import_array()

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
                  DOUBLE_t[:, :, ::1] tb=None) nogil except -1:
        # TODO: necessary to make sure init() here and init() in RegressionCriterion has same signatures?
        """Placeholder for a method which will initialize the criterion.
        For tensor basis criterion, tb, tb_tb, and tb_bij need to be supplied.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array-like, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
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

        cdef double impurity_left
        cdef double impurity_right

        # children_impurity() takes pointers which point to the address of impurity_*
        self.children_impurity(&impurity_left, &impurity_right)

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity - (self.weighted_n_right /
                             self.weighted_n_node_samples * impurity_right)
                          - (self.weighted_n_left /
                             self.weighted_n_node_samples * impurity_left)))


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
                  DOUBLE_t[:, :, ::1] tb=None) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
        children samples[start:start] and samples[start:end].
        tb, tb_tb, and tb_bij are for tensor basis criterion and have no effect here.

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
        tb : None
            Tensor basis Tij. Has no effect in Classification
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

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples, bint tb_mode=0):
        """Initialize parameters for this criterion.
        For

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted

        n_samples : SIZE_t
            The total number of samples to fit on

        tb_mode : bint
            On/off flag of tensor basis criterion, based on whether tb is supplied as input in
            DecisionTreeRegressor.fit()
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

        # Tensor basis criterion switch
        self.tb_mode = tb_mode
        # ls_lwork for dgelss() is calculated as
        # min(row, col)*3 + max(max(row, col), min(row, col)*2, nrhs)
        # = 30 + max(max(row, 10), 20)
        self.ls_lwork = 30 + max(max(n_samples*n_outputs, 10), 20)

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL

        # Tensor basis criterion related arrays
        self.tb_node, self.tb_transpose_node = NULL, NULL
        self.bij_node, self.bij_hat_node = NULL, NULL
        self.g_node, self.se_dev = NULL, NULL
        # Outputs of least-squares fit of Ax = b dgelss().
        # ls_s is singular value of A in decreasing order.
        # On exit, if ls_info = 0, ls_work(1) returns the optimal ls_lwork
        self.ls_s, self.ls_work = NULL, NULL

        # Allocate memory for the accumulators.
        # In tensor basis criterion, functionality of sum_* is mostly changed but usage not
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))

        # Allocate tensor basis related arrays over all samples in current tree,
        # represented by 1D flattened array of memory addresses.
        # The values are initialized as 0
        if tb_mode == 1:
            self.tb_node = <double*> calloc(n_samples*n_outputs*10, sizeof(double))
            self.tb_transpose_node = <double*> calloc(n_samples*n_outputs*10, sizeof(double))
            self.bij_node = <double*> calloc(n_samples*n_outputs, sizeof(double))
            self.bij_hat_node = <double*> calloc(n_samples*n_outputs, sizeof(double))
            self.g_node = <double*> calloc(10, sizeof(double))
            # Deviatoric SE will replace the functionality of self.sum_*
            self.se_dev = <double*> calloc(n_outputs, sizeof(double))
            # dgelss() least-squares fit of Ax = b related outputs.
            # Dimension min(A's row, A's col)
            self.ls_s = <double*> calloc(10, sizeof(double))
            # Dimension max(1, ls_lwork)
            self.ls_work = <double*> calloc(self.ls_lwork, sizeof(double))

        if (self.sum_total == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

        if tb_mode == 1:
            # Raise memory error for large tensor basis related arrays
            if (self.tb_node == NULL or
                self.tb_transpose_node == NULL or
                self.bij_node == NULL or
                self.bij_hat_node == NULL or
                self.ls_work == NULL):
                raise MemoryError()

    def __dealloc__(self):
        """Destructor. Additional destruction of newly introduced tensor basis arrays. Does nothing if array is NULL?"""
        if self.tb_mode == 1:
            # Deallocate tensor basis criterion related memory blocks
            free(self.tb_node)
            free(self.tb_transpose_node)
            free(self.bij_node)
            free(self.bij_hat_node)
            free(self.g_node)
            free(self.se_dev)
            # Deallocate S and work array used in dgelss least-squares fit of Ax = b
            free(self.ls_s)
            free(self.ls_work)

    def __reduce__(self):
        # For pickling. Addition arg of tb_mode
        return (type(self), (self.n_outputs, self.n_samples, self.tb_mode), self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end,
                  DOUBLE_t[:, :, ::1] tb=None) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end].
        For tensor basis criterion only:
        Tij is n_samples x 9 components x 10 bases or None.
        ::1 means C-contiguous
        """
        # Initialize fields
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
            printf("\nMSE criterion in tensor basis mode... ")
            # Calculate sum_total, i.e. deviatoric SE for samples from start to end index at this node.
            # Results are stored in self.se_dev, and g stored in self.g_node
            _ = self.reconstructAnisotropyTensor(start, end)
            # Then assign self.sum_total to self.se_dev
            memcpy(self.sum_total, self.se_dev, self.n_outputs*sizeof(double))
            # for k in range(self.n_outputs):
            #     self.sum_total[k] = self.se_dev[k]
        # If default sum_total behavior, then initialize sum_total to 0 for "+=" later
        else:
            memset(self.sum_total, 0, self.n_outputs * sizeof(double))

        for p in range(start, end):
            # Original unsorted sample index at this node, for sorted sample index in [start, end)
            i = samples[p]

            # Doesn't make sense for tensor basis mode to have custom weights at this moment
            if sample_weight != NULL and not self.tb_mode:
                w = sample_weight[i]
            else:
                w = 1.0

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                # If not in tensor basis mode, then calculate sum_total (a.k.a. n_samples*y_hat) as usual
                if not self.tb_mode:
                    self.sum_total[k] += w_y_ik

                # In tensor basis case, this is
                # sum((every bij component at every sample in this node)^2),
                # a.k.a. Non-deviatoric part of SE
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reconstructAnisotropyTensor(self, SIZE_t pos1, SIZE_t pos2) nogil except -1:
        """Reconstruct the anistropy tensor bij_hat of samples in the index list of samples[pos1:pos2] at a node of interest,
        and calculate the deviatoric SE component to mostly replace the functionality of 
        self.sum_total/sum_left/sum_right.
        First, collect bij (y), and Tij that lies in samples[pos1:pos2] of this node.
        Next, find one set of 10 g that best fit Tij*g = bij for all samples in this node via LS. 
            1). Compress Tij[n_samples x 9 x 10] to Tij[n_samples*9 x 10]
            # [DEPRECATED]
            # 2). Solve min_g J = ||Tij*g - bij||^2 by letting dJ/dg = ||Tij^T*g - Tij^T*bij|| = 0
            # 3). Find 10 g by LS of the under-determined linear system of 
            # (Tij^T*Tij)*g = (Tij^T*bij), dimensionally [10 x 10] x [10 x 1] = [10 x 1].
        Then, compute the reconstructed bij_hat of each sample in this node using bij_hat = Tij*g, 
        dimensionally [n_samples*9 x 1] = [n_samples*9 x 10]*[10 x 1].
        Finally, mostly replace the functionality of self.sum_total/sum_left/sum_right
        from self.sum_*[output] = sum_i^n_samples(yi[output]) to a deviatoric SE,
        se_dev[output] = sum_i^n_samples(2bij[output]*bij_hat[output] - bij_hat[output]^2),
        where SE = sum_i^n_samples(||yi||^2) + sum(se_dev)."""

        # Index array (samples) and variables definition
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t i1, i2, p, p0, i
        cdef SIZE_t nelem_bij_node = abs(pos2 - pos1)*n_outputs
        # Number of rows in Tij_node and bij_node, has to >= 10 for dgelss()
        cdef int row = max(nelem_bij_node, 10)

        # Reallocate the memory size of dynamic size arrays to number of Tij/bij elements in this node,
        # minimum [10 x 10] (e.g. Tij) or [10 x 1] (e.g. bij).
        # realloc() tries to perseve old memory as much as possible
        # -- old addresses' values are untouched and will be overwritten.
        # TODO: not realloc self.ls_work since it doesn't have to be resized?
        safe_realloc(&self.tb_node, row*10)
        safe_realloc(&self.tb_transpose_node, row*10)
        safe_realloc(&self.bij_node, row)
        safe_realloc(&self.bij_hat_node, row)

        # Initialize (or reset) memory block of flattened arrays that involves "+=" to 0
        # TODO: memset of arrays of no "+=" not necessary?
        # memset(self.tb_node, 0, row*10*sizeof(double))
        # memset(self.tb_transpose_node, 0, row*10*sizeof(double))
        # memset(self.bij_node, 0, row*sizeof(double))
        # # memset(self.tb_bij_node, 0, 10*sizeof(double))
        # # memset(self.tb_tb_node, 0, 100*sizeof(double))
        # # memset(self.tb_tb_node_fortran, 0, 100*sizeof(double))
        # memset(self.g_node, 0, 10*sizeof(double))
        memset(self.bij_hat_node, 0, row*sizeof(double))
        memset(self.se_dev, 0, n_outputs*sizeof(double))

        # Tensor basis related flattened array pointers for samples from pos1 to pos2 at this node.
        # Mainly for convenience of dumping "self.".
        # tb_node points to the address self.tb_node.
        # However, ptr[i] will point to the value at ith address in val. When tb_node is changed, self.tb_node is too
        cdef DOUBLE_t* tb_node = self.tb_node
        cdef DOUBLE_t* tb_transpose_node = self.tb_transpose_node
        cdef DOUBLE_t* bij_node = self.bij_node
        # cdef DOUBLE_t* tb_bij_node = self.tb_bij_node
        # cdef DOUBLE_t* tb_tb_node = self.tb_tb_node
        # cdef DOUBLE_t* tb_tb_node_fortran = self.tb_tb_node_fortran
        cdef DOUBLE_t* g_node = self.g_node
        cdef DOUBLE_t* bij_hat_node = self.bij_hat_node
        cdef DOUBLE_t* se_dev = self.se_dev
        # TODO: would the following work?
        # cdef DOUBLE_t* tb = &self.tb

        # Least-squares fit dgelss() related variables
        cdef int col = 10
        cdef int nrhs = 1
        cdef int lda = row
        cdef int ldb = row
        cdef DOUBLE_t rcond = -1
        cdef int rank, info
        # Dimension max(1, ls_lwork). ls_lwork is calculated as
        # min(row, col)*3 + max(max(row, col), min(row, col)*2)
        cdef int lwork = self.ls_lwork

        # Flatten Tij, bij and pick up Tij, bij for samples from pos1 to pos2 index at current node,
        # where p is n_samples (3rd axis), i1 is component (row), i2 is basis (column)
        for p in range(pos1, pos2):
            # Actual index of the original unsorted X
            i = samples[p]
            # Sample index but starts at 0 instead
            p0 = p - pos1
            for i1 in range(n_outputs):
                for i2 in range(10):
                    # n_samples*9 x 10 flattened to 1D memory addresses, C-contiguous.
                    # tb_node's index is in the order of basis -> component -> n_samples
                    tb_node[p0*n_outputs*10 + i1*10 + i2] = self.tb[i, i1, i2]
                    # C-contiguous flattened Tij^T, also interpreted as Fortran-contiguous Tij, 10 x n_samples*9.
                    # tb_transpose_node's index is in the order of component -> n_samples -> basis
                    tb_transpose_node[nelem_bij_node*i2 + p0*n_outputs + i1] = self.tb[i, i1, i2]

                # n_samples*9 x 1
                bij_node[p0*n_outputs + i1] = self.y[i, i1]

        # Least-squares fit with dgelss() to solve g from
        # min_g J = ||Tij*g - bij||^2.
        # The solution of 10 g is contained in bij_node after cython_lapack.dgelss().
        # Note that dgelss() is written in Fortran thus every matrix needs to be Fortran-contiguous.
        # TODO: why is reference used here?
        # TODO: explain 13 args
        cython_lapack.dgelss(&row, &col, &nrhs,
                             tb_transpose_node, &lda, bij_node, &ldb,
                             self.ls_s, &rcond, &rank,
                             self.ls_work, &lwork, &info)
         # Since g[10 x 1] is stored in bij_node after dgelss(),
        # go through each basis and get corresponding g_node at this node
        for i2 in range(10):
            g_node[i2] = bij_node[i2]

        """
        [DEPRECATED]
        
        # # Calculate Tij^T*Tij in both C-contiguous and Fortran-contiguous format and Tij^T*bij.
        # # i1 is row of Tij^T*Tij; row of Tij^T; row of Tij^T*bij
        # for i1 in range(10):
        #     # Index along the [n_samples*9] axis of Tij^T (column)
        #     for p in range(nelem_bij_node):
        #         # Tij^T*bij is [10 x n_samples*9]*[n_samples x 1] = [10 x 1]
        #         # For Tij^T, skip through rows
        #         tb_bij_node[i1] += tb_transpose_node[i1*nelem_bij_node + p]*bij_node[p]
        #
        #     # i2 is column of Tij^T*Tij; column of Tij
        #     for i2 in range(10):
        #         # Index along the [n_samples*9] axis of Tij^T (column) and Tij (row)
        #         for p in range(nelem_bij_node):
        #             # FIXME: declare, init, alloc, dealloc tb_tb_node, tb_tb_node_fortran
        #             # Tij^T*Tij is [10 x n_samples*9]*[n_samples*9 x 10] = [10 x 10].
        #             # For Tij^T, skip through rows; for Tij, skip through columns
        #             tb_tb_node[i1*10 + i2] += tb_transpose_node[i1*nelem_bij_node + p]*tb_node[i2*nelem_bij_node + p]
        #
        #         # On the other hand, Fortran-contiguous Tij^T*Tij stores matrix in memory column wise.
        #         # Moreover, C-contiguous A = Fortran-contiguous A^T. Hence i1 is column, i2 is row of tb_tb_fortran_node
        #         tb_tb_node_fortran[i2*10 + i1] = tb_tb_node[i1*10 + i2]
        #
        # # Least-squares fit with dgelss() to solve g from
        # # min_g J = ||Tij*g - bij||^2,
        # # => dJ/dg = 0,
        # # => Tij^T*Tij*g = Tij^T*bij..
        # # The solution of 10 g is contained in tb_bij_node after cython_lapack.dgelss().
        # # Note that dgelss() is written in Fortran thus every matrix needs to be Fortran-contiguous.
        # # TODO: why is reference used here?
        # # TODO: explain 13 args
        # cython_lapack.dgelss(&row, &col, &nrhs,
        #                      tb_tb_node_fortran, &lda, tb_bij_node, &ldb,
        #                      self.ls_s, &rcond, &rank,
        #                      self.ls_work, &lwork, &info)
        #  # Since g[10 x 1] is stored in tb_bij_node after dgelss(),
        # # go through each basis and get corresponding g_node at this node
        # for i2 in range(10):
        #     g_node[i2] = tb_bij_node[i2]
        """

        # Calculate reconstructed bij_hat at this node from Tij and g.
        # Manual dot product by aggregating each column (basis) in each row (component).
        for p in range(pos1, pos2):
            i = samples[p]
            p0 = p - pos1
            for i1 in range(n_outputs):
                # For each component, perform bij_hat = sum_1^10(Tij*g)
                for i2 in range(10):
                    # Flattened bij_hat construction complete at this node, n_samples*9 x 1
                    bij_hat_node[p0*n_outputs + i1] += self.tb[i, i1, i2]*g_node[i2]

                # Replacing (almost) the functionality of self.sum_total/sum_left/sum_right with deviatoric SE
                # Originally, self.sum_total/sum_left/sum_right is sum_i^n(yi).
                # Now, se_dev is the deviatoric SE and is sum_i^n(2bij*bij_hat - bij_hat^2),
                # and SE_ij = sum_i^n(bij^2) - se_dev[i1]; MSE_ij = sum_i^n(bij^2)/n - se_dev[i1]/n.
                # (M)MSE = sum_1^n_outputs(MSE_ij)/n_outputs
                se_dev[i1] += 2.0*self.y[i, i1]*bij_hat_node[p0*n_outputs + i1] \
                - bij_hat_node[p0*n_outputs + i1]**2.0

        return 0

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
        #           sum_left/right = deviatoric SE of the left/right child node samples,
        # and sum_left + sum_right != sum_total.
        # Thus separate set of g needs to be derived for both left/right child nodes and no shortcut then

        # Default sum_left and sum_right behavior
        if not self.tb_mode:
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
            # pos has been reset to self.start previously in reset()
            _ = self.reconstructAnisotropyTensor(pos, new_pos)
            # Then assign sum_left to self.se_dev, component by component
            # TODO: what if memcpy(sum_left) instead of memcpy(self.sum_left)?
            memcpy(self.sum_left, self.se_dev, self.n_outputs*sizeof(double))
            # for k in range(self.n_outputs):
            #     sum_left[k] = self.se_dev[k]
            # Also get number of samples in left child node, with weight disabled
            for p in range(pos, new_pos):
                # weighted_n_left has been reset to 0 in reset()
                self.weighted_n_left += w

            # Do the same for the right child node samples
            _ = self.reconstructAnisotropyTensor(new_pos, end)
            # TODO: what if memcpy(sum_right) instead of memcpy(self.sum_right)?
            memcpy(self.sum_right, self.se_dev, self.n_outputs*sizeof(double))
            # for k in range(self.n_outputs):
            #     sum_right[k] = self.se_dev[k]

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
        Definition of dest is changed for tensor basis criterion. By default, dest[k] is mean y[k] at this node.
        However, in tensor basis criterion, dest[k] is deviatoric MSE[k] = MSE[k] - sum_i^n_samples(yi[k]^2)/n_samples, 
        larger is better."""

        cdef SIZE_t k

        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples


cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.

        MSE = var_left + var_right
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end].
           This method incorporates a tensor basis criterion if Tij is given along with X, 
           and y in DecisionTreeRegressor.fit()."""

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            # If tensor basis criterion, then self.sum_*[k] is
            # sum_1^n_samples(2bij[k]*bij_hat[k] - bij_hat[k]^2),
            # and MSE = [sq_sum_total - sum_k(sum_total[k])]/9n_samples
            if self.tb_mode:
                impurity -= sum_total[k]/self.weighted_n_node_samples
            # Otherwise, MSE = sq_sum_total - sum_k^n_outputs(y_bar[k]^2)
            else:
                impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_outputs

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

        for k in range(self.n_outputs):
            # By default, proxy_impurity_left[k] is
            # weighted_n_left*y_bar[k]^2 = sum_left[k]^2/weighted_n_left,
            # and proxy_impurity_left = sum_1^n_outputs(proxy_impurity_left[k]), higher is better.
            #
            # However, in tensor basis criterion, proxy_impurity_left[k] is deviatoric SE_left[k] = sum_left[k].
            # Therefore, proxy_impurity_left[k] = sum_left[k], higher is better
            if not self.tb_mode:
                proxy_impurity_left += sum_left[k] * sum_left[k]
                proxy_impurity_right += sum_right[k] * sum_right[k]
            else:
                proxy_impurity_left += sum_left[k]
                proxy_impurity_right += sum_right[k]

        if not self.tb_mode:

            return (proxy_impurity_left / self.weighted_n_left +
                    proxy_impurity_right / self.weighted_n_right)
        else:

            return proxy_impurity_left + proxy_impurity_right

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end]).
           The calculation of impurity_left/right is different for tensor basis criterion since the definition of 
           sum_left/right in it is slightly altered.
           Nonetheless, the definition of sq_sum_left/right as well as impurity_left/right remain unchanged."""

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
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

            # Sample weight is disabled for tensor basis criterion
            if sample_weight != NULL and not self.tb_mode:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                # Definition of sq_sum_* remains unchanged in tensor basis criterion
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left
        # [0] of this pointer accesses its value
        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            # Just like in MSE.node_impurity(), since definition of sum_* is slightly changed,
            # impurity_* calculation is too, for tensor basis criterion
            if not self.tb_mode:
                impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
                impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0
            else:
                impurity_left[0] -= sum_left[k]/self.weighted_n_left
                impurity_right[0] -= sum_right[k]/self.weighted_n_right

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs

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

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples, bint tb_mode=0):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted

        n_samples : SIZE_t
            The total number of samples to fit on

        tb_mode : bint
            On/off flag of tensor basis criterion, based on whether tb is supplied as input in
            DecisionTreeRegressor.fit()
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

        # Tensor basis criterion switch
        self.tb_mode = tb_mode
        # TODO: tensor basis calculations for MAE
        # Off at the moment
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
        # Extra initialization of tensor basis Tij
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
