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
#
# License: BSD 3 clause

from ._criterion cimport Criterion

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.string cimport memset
# Verbose best.pos for split and best.improvement
from libc.stdio cimport printf
from libc.math cimport sqrt, fabs, nearbyint

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import csc_matrix

from ._utils cimport log
from ._utils cimport rand_int
from ._utils cimport rand_uniform
from ._utils cimport RAND_R_MAX
from ._utils cimport safe_realloc

cdef double INFINITY = np.inf

# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparseSplitter
cdef DTYPE_t EXTRACT_NNZ_SWITCH = 0.1

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY

cdef class Splitter:
    """Abstract splitter class.

    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state, bint presort,
                  # Split finding scheme "encoded" to integer
                  int split_finder_code=1,
                  # Verbose on node splitting
                  bint split_verbose=0,
                  # L2 regularization coefficient on finding split
                  double alpha_g_split=0.):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness

        split_finder_code : int, optional (default=1)
            The split finding scheme "encoded" to integer, either 1: "brute", 0: "brent", or 1000: "1000"

        split_verbose : bint, optional (default=0)
            Verbose in node_split()

        alpha_g_split : double, optional (default=0.)
            L2 regularization coefficient to penalize large optimal g during different splits.
            The best split in this case is sub-optimal as well as g, but reduces overfitting
        """

        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.sample_weight = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.presort = presort
        # Also initialize split_finder
        self.split_finder_code, self.split_verbose = split_finder_code, split_verbose
        self.alpha_g_split = alpha_g_split

    def __dealloc__(self):
        """Destructor."""

        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self,
                   object X,
                   const DOUBLE_t[:, ::1] y,
                   DOUBLE_t* sample_weight,
                   np.ndarray X_idx_sorted=None,
                   # Additional kwarg
                   DOUBLE_t[:, :, ::1] tb=None) except -1:
        """Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.
        For tensor basis criterion, tensor basis tb of shape (n_samples, n_outputs, n_bases) needs to be supplied,
        and y is anisotropy tensor of shape (n_samples, n_outputs).

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : numpy.ndarray, dtype=DOUBLE_t
            This is the vector of targets, or true labels, for the samples.
            For tensor basis criterion, this is anisotropy tensor bij of shape (n_samples, n_outputs).

        sample_weight : numpy.ndarray, dtype=DOUBLE_t (optional)
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight.
            
        tb : np.ndarray, dtype=DOUBLE_t, or None (optional)
            Tensor basis matrix Tij, n_samples x n_outputs x n_bases, used for tensor basis criterion.
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        self.y = y
        # Initialize tensor basis Tij
        self.tb = tb

        self.sample_weight = sample_weight
        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples : numpy.ndarray, dtype=double pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        # Additional args of tb, bij for tensor basis criterion
        self.criterion.init(self.y,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end,
                            self.tb)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef BrentResults _brentSplitFinder(self, double a, double b, double epsi=1e-6, double t=1e-6) nogil:
        """Find the best split x by using Brent's optimization to find local minimum of f(x). 
        
        Used in BestSplitter.node_split() if DecisionTreeRegressor.split_finder is "brent".
        
        This is a placeholder method and is overridden in BaseDenseSplitter(Splitter). 
        """

        pass

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will be done
        here.

        It should return -1 upon errors.
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()

    # cdef double* bestTensorBasisCoefficients(self, SIZE_t pos1, SIZE_t pos2) nogil:
    #     """Return the best found 10 g's pointer by inserting the best sample intervals."""
    #
    #     return self.criterion._reconstructAnisotropyTensor(pos1, pos2)

cdef class BaseDenseSplitter(Splitter):
    cdef const DTYPE_t[:, :] X

    cdef np.ndarray X_idx_sorted
    cdef INT32_t* X_idx_sorted_ptr
    cdef SIZE_t X_idx_sorted_stride
    cdef SIZE_t n_total_samples
    cdef SIZE_t* sample_mask

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state, bint presort,
                  # Additional kwargs
                  int split_finder_code=1,
                  bint split_verbose=0,
                  double alpha_g_split=0.):

        self.X_idx_sorted_ptr = NULL
        self.X_idx_sorted_stride = 0
        self.sample_mask = NULL
        self.presort = presort
        # Additional kwarg
        self.split_finder_code, self.split_verbose = split_finder_code, split_verbose
        self.alpha_g_split = alpha_g_split

    def __dealloc__(self):
        """Destructor."""
        if self.presort == 1:
            free(self.sample_mask)

    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  np.ndarray X_idx_sorted=None,
                  # Additional kwarg
                  DOUBLE_t[:, :, ::1] tb=None) except -1:
        """Initialize the splitter
        For tensor basis criterion, tensor basis tb of shape (n_samples, n_outputs, n_bases) needs to be supplied,
        and y is anisotropy tensor of shape (n_samples, n_outputs).

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        # Call parent init
        # Additional args of tb
        # X_idx_sorted in Splitter.init() has no effect
        Splitter.init(self, X, y, sample_weight, X_idx_sorted, tb)

        self.X = X

        if self.presort == 1:
            self.X_idx_sorted = X_idx_sorted
            self.X_idx_sorted_ptr = <INT32_t*> self.X_idx_sorted.data
            # Each axis of the array has a stride length,
            # which is the number of bytes needed to go from one element on this axis to the next element.
            # strides[1] means how many element-bytes to go to next column.
            # Note that when presort == 1,
            # X_idx_sorted has been converted to Fortran-contiguous in BaseDecisionTree.fit(),
            # and strides[1] varies based on whether C- or Fortran-contiguous.
            # C-contiguous memory address is row based, i.e. 1 element-byte needed to go to next column.
            # Fortran-contiguous memory address is column based, i.e., n_rows element-bytes needed to go to next column.
            # itemsize means bytes of such dtype.
            # Thus, X_idx_sorted_stride means how many elements (samples for Fortran-contiguous)
            # to go to next column (feature).
            self.X_idx_sorted_stride = (<SIZE_t> self.X_idx_sorted.strides[1] /
                                        <SIZE_t> self.X_idx_sorted.itemsize)

            self.n_total_samples = X.shape[0]
            safe_realloc(&self.sample_mask, self.n_total_samples)
            memset(self.sample_mask, 0, self.n_total_samples*sizeof(SIZE_t))

        return 0

    cdef BrentResults _brentSplitFinder(self, double a, double b, double epsi=1e-8, double t=1e-7) nogil:
        """
        _brentSplitFinder seeks a local minimum of a function F(X) in an interval (A, B)
        Modified to use Criterion.proxy_impurity_improvement_pipeline() inherently.
        Since higher proxy impurity improvement is better, F(x) becomes -F(X) to minimize. 
        
        Discussion:
            The method used is a combination of golden section search and successive parabolic interpolation. 
            Convergence is never much slower than that for a Fibonacci search. 
            If F has a continuous second derivative which is positive at the minimum (which is not at A or B), 
            then convergence is superlinear, and usually of the order of about 1.324....
            
            The values EPSI and T define a tolerance TOL = EPSI * abs ( X ) + T.
            F is never evaluated at two points closer than TOL.
            
            If F is a unimodal function and the computed values of F are always unimodal 
            when separated by at least SQEPS * abs ( X ) + (T/3),
            then LOCAL_MIN approximates the abscissa of the global minimum of F on the interval [A,B] 
            with an error less than 3*SQEPS*abs(LOCAL_MIN)+T.
            
            If F is not unimodal, then LOCAL_MIN may approximate a local, 
            but perhaps non-global, minimum to the same accuracy.
            
            Thanks to Jonathan Eggleston for pointing out a correction to the golden section step, 01 July 2013.
        
        Licensing:
            This code is distributed under the GNU LGPL license.
        
        Modified:
            01 June 2019
        
        Author:
            Original FORTRAN77 version by Richard Brent.
            Python version by John Burkardt.
            Cython version by Yuyang Luan.
        
        Reference:
            Richard Brent,
            Algorithms for Minimization Without Derivatives,
            Dover, 2002,
            ISBN: 0-486-41998-3,
            LC: QA402.5.B74.
        
        Parameters:
            Input, real A, B, the endpoints of the interval.
            
            Input, real EPSI, a positive relative error tolerance.
            EPSI should be no smaller than twice the relative machine precision,
            and preferably not much less than the square root of the relative machine precision.
            
            Input, real T, a positive absolute error tolerance.
                        
            Output, real X, the estimated value of an abscissa for which F attains a local minimum value in [A, B].
            
            Output, real FX, the value F(X).
        """

        # Bringing tolerance out of while loop since it's constant here
        cdef double tol = 0.25
        # Twice the tolerance
        cdef double t2 = 2.*tol
        # Make sure impurity improvement can be evaluated at a and b border
        a -= tol
        b += tol
        # v = w = x = a + (3 - sqrt(5))/2*(b - a) = a + 0.381966(b - a)
        cdef double c = 0.5*(3. - sqrt(5.))
        # Lower and upper bound of x
        cdef double sa = a
        cdef double sb = b
        # x is least value of f, or most recent evaluation if tied
        cdef double x = sa + c*(b - a)
        # w is point with next lowest value of f
        cdef double w = x
        # v is previous w
        cdef double v = w
        cdef double d
        cdef double e = 0.
        cdef double fx, fu, fv, fw
        cdef double m
        # u is last point at which f has been evaluated
        cdef double p, q, r, u
        # Since there're multiple returns, returns are grouped in BrentResults struct
        cdef BrentResults returns

        # Initial f(x), can be negative
        fx = self.criterion.proxy_impurity_improvement_pipeline(x,
                                                                self.min_samples_leaf,
                                                                self.min_weight_leaf,
                                                                self.alpha_g_split)
        if self.split_verbose:
            printf("\n     Initial proxy impurity improvement: %8.8f ", fx)

        # # First avoid FPE
        # if -1e-10 < fx < 1e-10:
        #     fx = 1e-10 if fx > 0. else -1e-10

        # Since we want to minimize f(x), take inverse
        fx = -fx
        # # Since we want to minimize f(x), take the reciprocal of proxy_impurity_improvement
        # if fx > 0.:
        #     fx = 1./fx
        # # However, if f(x) was negative
        # else:
        #     # If -f(x) < 1, i.e. a fraction, use 1/fraction to amplify it (lower is better)
        #     # Else if -f(x) >= 1, use -fx to enlarge it
        #     # Finally penalize it with 2 multiplier
        #     fx = 2./(-fx) if -fx < 1. else 2.*(-fx)

        fw = fx
        fv = fw
        while 1:
            # Middle point
            m = 0.5*(sa + sb)
            # # Calculate tolerance, f is never evaluated at 2 points closer than tol
            # tol = epsi*fabs(x) + t
            # # Twice the tolerance
            # t2 = 2.*tol

            # Check the stopping criterion
            # When x is close enough to the middle point between sa and sb
            if fabs(x - m) <= t2 - 0.5*(sb - sa):
                break

            # Fit a parabola
            r = 0.
            q = r
            p = q
            # If tolerance is smaller than e
            if tol < fabs(e):
                r = (x - w)*(fx - fv)
                q = (x - v)*(fx - fw)
                p = (x - v)*q - (x - w)*r
                q = 2.*(q - r)
                # Switch sign of p if q > 0
                if q > 0.: p = -p
                # Ensure non-negative q
                q = fabs(q)
                r = e
                # d is defined below
                e = d

            if fabs(p) < fabs(0.5*q*r) and \
                q*(sa - x) < p and \
                p < q*(sb - x):
                # Take the parabolic interpolation step
                d = p/q
                u = x + d
                # f(x) must not be evaluated too close to lower or upper bound
                if (u - sa) < t2 or (sb - u) < t2:
                    # d is positive if x is left to the middle point
                    d = tol if x < m else -tol

            # Otherwise, switch to golden-section
            else:
                # Since sb is right of x, e is set to positive if x is left to the middle point
                e = sb - x if x < m else sa - x
                # d is e*golden ratio
                d = c*e

            # f(x) must not be evaluated too close to x
            if tol <= fabs(d):
                u = x + d
            elif d > 0.:
                u = x + tol
            else:
                u = x - tol

            # Update f(x) after incorporating the tolerance
            fu = self.criterion.proxy_impurity_improvement_pipeline(u,
                                                                    self.min_samples_leaf,
                                                                self.min_weight_leaf,
                                                                self.alpha_g_split)
            # # Again, avoid f(u) FPE when getting the reciprocal
            # if -1e-10 < fu < 1e-10:
            #     fu = 1e-10 if fu > 0. else -1e-10

            # Again, get inverse of f(u) to minimize
            fu = -fu
            # # Again, get reciprocal if positive and penalize if negative
            # if fu > 0.:
            #     fu = 1./fu
            # else:
            #     fu = 2./(-fu) if -fu < 1. else 2.*(-fu)

            # Update a, b, v, w, and x
            # If a lower f(x) is found
            if fu <= fx:
                if self.split_verbose:
                    printf("\n     Proxy impurity improvement from %8.8f to %8.8f ", -fx, -fu)

                # Then if u is left of x, shrink sb to x, x becomes the new upper bound
                if u < x:
                    if self.split_verbose:
                        printf("\n     New x bounded from [%8.2f, %8.2f] to [%8.2f, %8.2f]", sa, sb, sa, x)

                    sb = x
                # Else if u is right of x, shrink sa to x, x becomes the new lower bound
                else:
                    if self.split_verbose:
                        printf("\n     New x bounded from [%8.2f, %8.2f] to [%8.2f, %8.2f]", sa, sb, x, sb)

                    sa = x

                v, fv = w, fw
                w, fw = x, fx
                x, fx = u, fu
            # Else if f(u) is not a lower value than f(x)
            else:
                # Shrink the bounds, do the opposite of previously
                if u < x:
                    sa = u
                else:
                    sb = u

                if fu <= fw or w == x:
                    v, fv = w, fw
                    w, fw = u, fu
                elif fu <= fv or v == x or v == w:
                    v, fv = u, fu

        returns.x, returns.fx = x, -fx

        return returns


cdef class BestSplitter(BaseDenseSplitter):
    """Splitter for finding the best split."""
    def __reduce__(self):
        return (BestSplitter, (self.criterion,
                               self.max_features,
                               self.min_samples_leaf,
                               self.min_weight_leaf,
                               self.random_state,
                               self.presort,
                               # Extra args
                               self.split_finder_code,
                               self.split_verbose,
                               self.alpha_g_split), self.__getstate__())

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best split on node samples[start:end].
        n_constant_features : array-like, initialized as [0] in DepthFirstTreeBuilder.build()

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t n_samples = end - start

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        # Feature values initialized as list of 0
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef INT32_t* X_idx_sorted = self.X_idx_sorted_ptr
        cdef SIZE_t* sample_mask = self.sample_mask

        # # Best left and right branch 10 g for tensor basis criterion.
        # # Pointers to the memory address of g_node in _reconstructAnisotropyTensor(),
        # # changing with g_node's memory address
        # cdef double* g_node_lptr
        # cdef double* g_node_rptr
        # # Pointers to another memory address of 10, not affected by changing g_node's memory address,
        # # to "save progress"
        # cdef double* g_l = <double*> calloc(10, sizeof(double))
        # cdef double* g_r = <double*> calloc(10, sizeof(double))

        # SplitRecord struct contains feature, pos, threshold, improvement, impurity_left/right
        cdef SplitRecord best, current
        cdef double current_proxy_improvement = -INFINITY
        # Reject negative proxy improvement so that there'll be no node having predictions worse than 0 prediction
        cdef double best_proxy_improvement = 0.  # -INFINITY
        # Define a BrentResults struct to catch multiple returns of _brentSplitFinder()
        cdef BrentResults brent_results

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t tmp
        cdef SIZE_t p
        cdef SIZE_t feature_idx_offset
        cdef SIZE_t feature_offset
        cdef SIZE_t i
        cdef SIZE_t j
        # # Tensor basis criterion related iterators to avoid any unintentional overwrite of other iterators
        # cdef SIZE_t isorted, iunsorted, ibasis

        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end
        cdef double best_pos, pos_double
        cdef double brent_start, brent_end
        # # (Unused) Brent optimization absolute and relative tolerance
        # cdef double xtol = 1e-8
        # cdef double rtol = 1e-7
        # If using "1000" split_scheme, pstep is capped to 1000.
        # Else if using "auto" split_scheme, pstep is 1 when n_samples <= 1000
        cdef SIZE_t pstep = <SIZE_t> max(1, nearbyint((end - start)/1000.)) if self.split_finder_code == 1000 else 1
        if self.split_verbose:
            if self.split_finder_code == 0:
                printf("\n    Using Brent optimization to find the best split of each node... ")
            elif self.split_finder_code == 1000:
                printf('\n    Using brute force capped to 1000 splits to find the best split of each node... ')
            elif self.split_finder_code == 10000 and n_samples > 1000:
                printf('\n    Using Brent optimization to find the best split of each node... ')
            else:
                printf('\n    Using brute force to find the best split of each node... ')

        # Initialize "best" SplitRecord incl. its best.pos to end
        _init_split(&best, end)

        # If presort enabled, all samples lie in start - end index will be masked 1,
        # effectively picking up indices in current node
        if self.presort == 1:
            for p in range(start, end):
                sample_mask[samples[p]] = 1

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        #  n_features-      0+
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                #     0 += 1      const (<= n_features)
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 #     0 += 1                 0+                   0+
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.
            # features[:n_drawn_constants ~ n_known_constants ~ n_total_constants ~ n_features]
            #                                                                   ^f_i^
            # Initially,       0                   0                   0            n_features

            # Draw a feature at random in [n_drawn_constants, f_i - n_found_constants).
            # Initially, f_j in [0, f_i), and f_j < f_i always
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            # If f_j is a known constant feature, swap features[f_j <-> n_drawn_constants].
            # Recall features is a list of [0, 1, 2, ..., n_features - 1]
            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1
            # Else if f_j is not a known constant feature (could still be constant, just not known yet)
            else:
                # Shift f_j by n_found_constants
                # so f_j is effectively drawn from [n_total_constants, f_i).
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # Current feature index
                # f_j in the interval [n_total_constants, f_i[
                current.feature = features[f_j]
                if self.split_verbose:
                    printf("\n    Current feature is %d ", f_j)

                # Sort samples along that feature; either by utilizing
                # presorting, or by copying the values into an array and
                # sorting the array in a manner which utilizes the cache more
                # effectively.
                if self.presort == 1:
                    # Sorted sample index at this node, looped from start to end
                    # (only when X[j] lies in samples[start:end] at this node, via brute search)
                    p = start
                    # X_idx_sorted_stride = n_samples due to Fortran-contiguous conversion when presort == 1.
                    # Thus feature index offset = n_samples * current feature index,
                    # synonymous with column (feature) number in X_idx_sorted 2D array
                    feature_idx_offset = self.X_idx_sorted_stride * current.feature

                    for i in range(self.n_total_samples):
                        # Original unsorted sample index at current feature.
                        # X_idx_sorted is a pointer to a list
                        j = X_idx_sorted[i + feature_idx_offset]
                        # If such sample lies in samples[start:end] in this node
                        if sample_mask[j] == 1:
                            # Sample index at this node starts at j, then the next time this condition fulfills,
                            # the next sample index at this node starts at the new j
                            samples[p] = j
                            # Since j is the original unsorted sample index,
                            # get corresponding X value at j for current feature
                            Xf[p] = self.X[j, current.feature]
                            p += 1
                else:
                    for i in range(start, end):
                        Xf[i] = self.X[samples[i], current.feature]

                    sort(Xf + start, samples + start, end - start)

                # If sorted feature values Xf are all constant, found constant feature + 1
                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                # While going through every feature in sampled max_features features,
                # if currently non-constant feature
                else:
                    f_i -= 1
                    # Let feature[f_i] be current feature
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Evaluate all splits
                    # Set sum_left to 0, sum_right to sum_total, and pos to start, for every feature
                    self.criterion.reset()
                    p = start

                    # Go through every sample at current node
                    # If split_finder is "brent", then use Brent optimization to find the best split
                    if self.split_finder_code == 0 or \
                            (self.split_finder_code == 10000 and n_samples > 1000):
                        current.pos = p
                        # TODO: not skipping constant feature values atm
                        # Before starting Brent optimization,
                        # ensure left and right bin have at least min_samples_leaf samples.
                        # Recall split is [start, pos - 1] and [pos, end - 1].
                        # Later threshold is between pos - 1 and pos, i.e. split before pos
                        brent_start, brent_end = p + min_samples_leaf, end - min_samples_leaf
                        # Reject if min_samples_leaf is not guaranteed
                        if brent_start <= brent_end:
                            # If only 1 split possible, e.g. 0, 1, | 2, 3, min_samples_split = 2,
                            # then bypass brent optimization and calculate impurity improvement
                            # for [start, current.pos) and [current.pos, end)
                            if brent_start == brent_end:
                                current.pos = <SIZE_t> brent_start
                                brent_results.x = <double> current.pos
                                # Equivalent to current_proxy_improvement.
                                # As long as it > 0, it is an improvement and the split should be done
                                brent_results.fx = self.criterion.proxy_impurity_improvement_pipeline(
                                        brent_results.x,
                                        min_samples_leaf,
                                        min_weight_leaf,
                                        self.alpha_g_split)
                                # If impurity improvement is non-positive, the split is bad, reset it to worst
                                if brent_results.fx <= 0.: brent_results.fx = -INFINITY
                            else:
                                # Brent optimization to find best split and corresponding pseudo impurity improvement
                                brent_results = self._brentSplitFinder(brent_start, brent_end)

                            # Current proxy improvement is this current feature
                            # and is not necessarily the best among all features
                            current_proxy_improvement = brent_results.fx
                            # For a feature, if improvement is even larger than previous feature, save it
                            if current_proxy_improvement > best_proxy_improvement:
                                best_proxy_improvement = current_proxy_improvement
                                # Update current.pos to best split of current feature,
                                # which will be inherited by best.
                                # nearbyint() doesn't raise FE_INEXACT exception,
                                # unlike rint()
                                current.pos = <SIZE_t> nearbyint(brent_results.x)
                                # Split value
                                # sum of halves is used to avoid infinite value
                                current.threshold = Xf[current.pos - 1] / 2.0 + Xf[current.pos] / 2.0
                                if ((current.threshold == Xf[current.pos]) or
                                    (current.threshold == INFINITY) or
                                    (current.threshold == -INFINITY)):
                                    # Making sure current split value isn't +-INFINITY anymore
                                    current.threshold = Xf[current.pos - 1]

                                # Best is the final record of the best split
                                best = current  # copy

                            # # After best split is found, calculate 10 g again using best split location
                            # # for both left and right branch.
                            # # Has to be in the feature loop since samples are changing every non-constant feature
                            # if self.tb is not None:
                            #     # Recalculate left and right g_node using best split position
                            #     self.criterion._reconstructAnisotropyTensor(start, best.pos)
                            #     memcpy(g_l, g_node_lptr)
                            #     # Return is a pointer to MSE.g_node which is another pointer
                            #     # to the memory address of 10 g_node
                            #     g_node_lptr = self.bestTensorBasisCoefficients(start, best.pos)
                            #     # Since g_node is calculated again for right branch
                            #     # which changes conten g_node is pointing to,
                            #     # copy memory to a new memory block pointed by g_l.
                            #     # memcpy(dest, src) doesn't alter content dest is pointing to
                            #     # even if src's content is changed later
                            #     memcpy(g_l, g_node_lptr, 10*sizeof(double))
                            #     # Repeat for the right branch
                            #     g_node_rptr = self.bestTensorBasisCoefficients(best.pos, end)
                            #     memcpy(g_r, g_node_rptr, 10*sizeof(double))
                            #     # Change y(g)'s content at corresponding points, i.e.
                            #     # change whatever is in left branch to g_l, vice versa.
                            #     # Go through all sorted points from start to end of this node
                            #     for isorted in range(start, end):
                            #         # Get the original unsorted index
                            #         iunsorted = samples[isorted]
                            #         # For each sample in this node, go through every tensor basis
                            #         for ibasis in range(10):
                            #             # If the sorted index is left to best.pos, then use best g of left branch;
                            #             # vice versa
                            #             if p < best.pos:
                            #                 self.y[iunsorted, ibasis] = g_l[ibasis]
                            #             else:
                            #                 self.y[iunsorted, ibasis] = g_r[ibasis]


                    # Else if split_finder is "brute" or "1000" or "auto" (but <= 1000 samples at this node)
                    else:
                        while p < end:
                            # Skip constant feature values
                            while (p + 1 < end and
                                   Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                                p += 1

                            # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                            #                    X[samples[p], current.feature])
                            # Skip samples to speed up finding best split if split_finder is '1000'
                            p += pstep
                            # (p >= end) or (X[samples[p], current.feature] >
                            #                X[samples[p - 1], current.feature])

                            if p < end:
                                current.pos = p

                                # # Reject if min_samples_leaf is not guaranteed
                                # if (((current.pos - start) < min_samples_leaf) or
                                #         ((end - current.pos) < min_samples_leaf)):
                                #     continue
                                #
                                # # Derive and update sum_left, sum_right from current.pos and sum_total
                                # # and set Criterion.pos to current.pos
                                # self.criterion.update(current.pos)
                                #
                                # # Reject if min_weight_leaf is not satisfied
                                # if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                #         (self.criterion.weighted_n_right < min_weight_leaf)):
                                #     continue
                                #
                                # # For default MSE
                                # # Having derived sum_left and sum_right, calculate interim pseudo impurity improvement as
                                # # sum_left(y)^2/n_left + sum_right(y)^2/n_right, higher is better.
                                # # Basically mean(y_left)^2 + mean(y_right)^2, higher is better
                                # #
                                # # For tensor basis MSE, impurity improvement becomes (sum_left + sum_right) due to a
                                # # slight definition change of sum_*
                                # current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                                # Cast current.pos to double since proxy_impurity_improvement_pipeline()
                                # takes double split location to accomodate Brent optimization split finder
                                pos_double = <double> current.pos
                                current_proxy_improvement = self.criterion.proxy_impurity_improvement_pipeline(
                                        pos_double,
                                        min_samples_leaf,
                                        min_weight_leaf,
                                        self.alpha_g_split)
                                # If only 1 split possible, e.g. 0, 1, | 2, 3, min_samples_leaf = 2,
                                # then if impurity improvement <= 0, reject split
                                if start + min_samples_leaf == end - min_samples_leaf:
                                    if current_proxy_improvement <= 0.: current_proxy_improvement = -INFINITY

                                if current_proxy_improvement > best_proxy_improvement:
                                    best_proxy_improvement = current_proxy_improvement
                                    # Split value
                                    # sum of halves is used to avoid infinite value
                                    current.threshold = Xf[p - 1] / 2.0 + Xf[p] / 2.0
                                    if ((current.threshold == Xf[p]) or
                                        (current.threshold == INFINITY) or
                                        (current.threshold == -INFINITY)):
                                        # Making sure current split value isn't +-INFINITY anymore
                                        current.threshold = Xf[p - 1]

                                    best = current  # copy

                                    # # After best split for a feature is found,
                                    # # calculate 10 g again using best split location for both left and right branch.
                                    # # Has to be in the feature loop since samples are changing every non-constant feature
                                    # if self.tb is not None:
                                    #     g_node_lptr = self.bestTensorBasisCoefficients(start, current.pos)
                                    #     memcpy(g_l, g_node_lptr, 10*sizeof(double))
                                    #     g_node_rptr = self.bestTensorBasisCoefficients(current.pos, end)
                                    #     memcpy(g_r, g_node_rptr, 10*sizeof(double))
                                    #     for isorted in range(start, end):
                                    #         iunsorted = samples[isorted]
                                    #         for ibasis in range(10):
                                    #             if p < best.pos:
                                    #                 self.y[iunsorted, ibasis] = g_l[ibasis]
                                    #             else:
                                    #                 self.y[iunsorted, ibasis] = g_r[ibasis]


        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if self.split_verbose:
            printf("\n    Best split is %d with feature %d ", best.pos, best.feature)

        # This check is because when there were no split cuz of min_samples_leaf or min_weight_leaf,
        # best.pos has been initialized to end
        if best.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                if self.X[samples[p], best.feature] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

            self.criterion.reset()
            # Now that best.pos is found recalculate sum_left and sum_right based on best.pos
            self.criterion.update(best.pos)
            # Calculate actual impurity improvement based on best.pos
            best.improvement = self.criterion.impurity_improvement(impurity)
            # Verbose on best split position and impurity improvement
            if self.split_verbose:
                printf("\n    Impurity improved %8.8f, split at %d ", best.improvement, best.pos)

            # Calculate children impurity based on best.pos
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)

        # Reset sample mask
        if self.presort == 1:
            for p in range(start, end):
                sample_mask[samples[p]] = 0

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants

        # # Free temporary pointer to left and right branch's g
        # if self.tb is not None and self.bij is not None:
        #     free(g_l)
        #     free(g_r)

        return 0


# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    if n == 0:
      return
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples,
        SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples,
                    SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1


cdef class RandomSplitter(BaseDenseSplitter):
    """Splitter for finding the best random split."""
    def __reduce__(self):
        return (RandomSplitter, (self.criterion,
                                 self.max_features,
                                 self.min_samples_leaf,
                                 self.min_weight_leaf,
                                 self.random_state,
                                 self.presort), self.__getstate__())

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best random split on node samples[start:end]

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Draw random splits and pick the best
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        cdef double current_proxy_improvement = - INFINITY
        cdef double best_proxy_improvement = - INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t p
        cdef SIZE_t tmp
        cdef SIZE_t feature_stride
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef SIZE_t n_visited_features = 0
        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end

        _init_split(&best, end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):
            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]

                # Find min, max
                min_feature_value = self.X[samples[start], current.feature]
                max_feature_value = min_feature_value
                Xf[start] = min_feature_value

                for p in range(start + 1, end):
                    current_feature_value = self.X[samples[p], current.feature]
                    Xf[p] = current_feature_value

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Draw a random threshold
                    current.threshold = rand_uniform(min_feature_value,
                                                     max_feature_value,
                                                     random_state)

                    if current.threshold == max_feature_value:
                        current.threshold = min_feature_value

                    # Partition
                    partition_end = end
                    p = start
                    while p < partition_end:
                        current_feature_value = Xf[p]
                        if current_feature_value <= current.threshold:
                            p += 1
                        else:
                            partition_end -= 1

                            Xf[p] = Xf[partition_end]
                            Xf[partition_end] = current_feature_value

                            tmp = samples[partition_end]
                            samples[partition_end] = samples[p]
                            samples[p] = tmp

                    current.pos = partition_end

                    # Reject if min_samples_leaf is not guaranteed
                    if (((current.pos - start) < min_samples_leaf) or
                            ((end - current.pos) < min_samples_leaf)):
                        continue

                    # Evaluate split
                    self.criterion.reset()
                    self.criterion.update(current.pos)

                    # Reject if min_weight_leaf is not satisfied
                    if ((self.criterion.weighted_n_left < min_weight_leaf) or
                            (self.criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            if current.feature != best.feature:
                partition_end = end
                p = start

                while p < partition_end:
                    if self.X[samples[p], best.feature] <= best.threshold:
                        p += 1

                    else:
                        partition_end -= 1

                        tmp = samples[partition_end]
                        samples[partition_end] = samples[p]
                        samples[p] = tmp


            self.criterion.reset()
            self.criterion.update(best.pos)
            best.improvement = self.criterion.impurity_improvement(impurity)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0


cdef class BaseSparseSplitter(Splitter):
    # The sparse splitter works only with csc sparse matrix format
    cdef DTYPE_t* X_data
    cdef INT32_t* X_indices
    cdef INT32_t* X_indptr

    cdef SIZE_t n_total_samples

    cdef SIZE_t* index_to_samples
    cdef SIZE_t* sorted_samples

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state, bint presort,
                  # Extra kwargs
                  int split_finder_code=1,
                  bint split_verbose=0,
                  double alpha_g_split=0.):
        # Parent __cinit__ is automatically called

        self.X_data = NULL
        self.X_indices = NULL
        self.X_indptr = NULL

        self.n_total_samples = 0

        self.index_to_samples = NULL
        self.sorted_samples = NULL

    def __dealloc__(self):
        """Deallocate memory."""
        free(self.index_to_samples)
        free(self.sorted_samples)

    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  np.ndarray X_idx_sorted=None,
                  DOUBLE_t[:, :, ::1] tb=None) except -1:
        """Initialize the splitter.
        For tensor basis criterion, tensor basis tb of shape (n_samples, n_outputs, n_bases) needs to be supplied,
        and y is anisotropy tensor of shape (n_samples, n_outputs).

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Call parent init
        # Also provide arg of tb and bij even if they're None
        # X_idx_sorted has no effect in Splitter.init()
        Splitter.init(self, X, y, sample_weight, X_idx_sorted, tb)

        if not isinstance(X, csc_matrix):
            raise ValueError("X should be in csc format")

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t n_samples = self.n_samples

        # Initialize X
        cdef np.ndarray[dtype=DTYPE_t, ndim=1] data = X.data
        cdef np.ndarray[dtype=INT32_t, ndim=1] indices = X.indices
        cdef np.ndarray[dtype=INT32_t, ndim=1] indptr = X.indptr
        cdef SIZE_t n_total_samples = X.shape[0]

        self.X_data = <DTYPE_t*> data.data
        self.X_indices = <INT32_t*> indices.data
        self.X_indptr = <INT32_t*> indptr.data
        self.n_total_samples = n_total_samples

        # Initialize auxiliary array used to perform split
        safe_realloc(&self.index_to_samples, n_total_samples)
        safe_realloc(&self.sorted_samples, n_samples)

        cdef SIZE_t* index_to_samples = self.index_to_samples
        cdef SIZE_t p
        for p in range(n_total_samples):
            index_to_samples[p] = -1

        for p in range(n_samples):
            index_to_samples[samples[p]] = p
        return 0

    cdef inline SIZE_t _partition(self, double threshold,
                                  SIZE_t end_negative, SIZE_t start_positive,
                                  SIZE_t zero_pos) nogil:
        """Partition samples[start:end] based on threshold."""

        cdef double value
        cdef SIZE_t partition_end
        cdef SIZE_t p

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t* index_to_samples = self.index_to_samples

        if threshold < 0.:
            p = self.start
            partition_end = end_negative
        elif threshold > 0.:
            p = start_positive
            partition_end = self.end
        else:
            # Data are already split
            return zero_pos

        while p < partition_end:
            value = Xf[p]

            if value <= threshold:
                p += 1

            else:
                partition_end -= 1

                Xf[p] = Xf[partition_end]
                Xf[partition_end] = value
                sparse_swap(index_to_samples, samples, p, partition_end)

        return partition_end

    cdef inline void extract_nnz(self, SIZE_t feature,
                                 SIZE_t* end_negative, SIZE_t* start_positive,
                                 bint* is_samples_sorted) nogil:
        """Extract and partition values for a given feature.

        The extracted values are partitioned between negative values
        Xf[start:end_negative[0]] and positive values Xf[start_positive[0]:end].
        The samples and index_to_samples are modified according to this
        partition.

        The extraction corresponds to the intersection between the arrays
        X_indices[indptr_start:indptr_end] and samples[start:end].
        This is done efficiently using either an index_to_samples based approach
        or binary search based approach.

        Parameters
        ----------
        feature : SIZE_t,
            Index of the feature we want to extract non zero value.


        end_negative, start_positive : SIZE_t*, SIZE_t*,
            Return extracted non zero values in self.samples[start:end] where
            negative values are in self.feature_values[start:end_negative[0]]
            and positive values are in
            self.feature_values[start_positive[0]:end].

        is_samples_sorted : bint*,
            If is_samples_sorted, then self.sorted_samples[start:end] will be
            the sorted version of self.samples[start:end].

        """
        cdef SIZE_t indptr_start = self.X_indptr[feature],
        cdef SIZE_t indptr_end = self.X_indptr[feature + 1]
        cdef SIZE_t n_indices = <SIZE_t>(indptr_end - indptr_start)
        cdef SIZE_t n_samples = self.end - self.start

        # Use binary search if n_samples * log(n_indices) <
        # n_indices and index_to_samples approach otherwise.
        # O(n_samples * log(n_indices)) is the running time of binary
        # search and O(n_indices) is the running time of index_to_samples
        # approach.
        if ((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
                n_samples * log(n_indices) < EXTRACT_NNZ_SWITCH * n_indices):
            extract_nnz_binary_search(self.X_indices, self.X_data,
                                      indptr_start, indptr_end,
                                      self.samples, self.start, self.end,
                                      self.index_to_samples,
                                      self.feature_values,
                                      end_negative, start_positive,
                                      self.sorted_samples, is_samples_sorted)

        # Using an index to samples  technique to extract non zero values
        # index_to_samples is a mapping from X_indices to samples
        else:
            extract_nnz_index_to_samples(self.X_indices, self.X_data,
                                         indptr_start, indptr_end,
                                         self.samples, self.start, self.end,
                                         self.index_to_samples,
                                         self.feature_values,
                                         end_negative, start_positive)


cdef int compare_SIZE_t(const void* a, const void* b) nogil:
    """Comparison function for sort."""
    return <int>((<SIZE_t*>a)[0] - (<SIZE_t*>b)[0])


cdef inline void binary_search(INT32_t* sorted_array,
                               INT32_t start, INT32_t end,
                               SIZE_t value, SIZE_t* index,
                               INT32_t* new_start) nogil:
    """Return the index of value in the sorted array.

    If not found, return -1. new_start is the last pivot + 1
    """
    cdef INT32_t pivot
    index[0] = -1
    while start < end:
        pivot = start + (end - start) / 2

        if sorted_array[pivot] == value:
            index[0] = pivot
            start = pivot + 1
            break

        if sorted_array[pivot] < value:
            start = pivot + 1
        else:
            end = pivot
    new_start[0] = start


cdef inline void extract_nnz_index_to_samples(INT32_t* X_indices,
                                              DTYPE_t* X_data,
                                              INT32_t indptr_start,
                                              INT32_t indptr_end,
                                              SIZE_t* samples,
                                              SIZE_t start,
                                              SIZE_t end,
                                              SIZE_t* index_to_samples,
                                              DTYPE_t* Xf,
                                              SIZE_t* end_negative,
                                              SIZE_t* start_positive) nogil:
    """Extract and partition values for a feature using index_to_samples.

    Complexity is O(indptr_end - indptr_start).
    """
    cdef INT32_t k
    cdef SIZE_t index
    cdef SIZE_t end_negative_ = start
    cdef SIZE_t start_positive_ = end

    for k in range(indptr_start, indptr_end):
        if start <= index_to_samples[X_indices[k]] < end:
            if X_data[k] > 0:
                start_positive_ -= 1
                Xf[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)


            elif X_data[k] < 0:
                Xf[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void extract_nnz_binary_search(INT32_t* X_indices,
                                           DTYPE_t* X_data,
                                           INT32_t indptr_start,
                                           INT32_t indptr_end,
                                           SIZE_t* samples,
                                           SIZE_t start,
                                           SIZE_t end,
                                           SIZE_t* index_to_samples,
                                           DTYPE_t* Xf,
                                           SIZE_t* end_negative,
                                           SIZE_t* start_positive,
                                           SIZE_t* sorted_samples,
                                           bint* is_samples_sorted) nogil:
    """Extract and partition values for a given feature using binary search.

    If n_samples = end - start and n_indices = indptr_end - indptr_start,
    the complexity is

        O((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
          n_samples * log(n_indices)).
    """
    cdef SIZE_t n_samples

    if not is_samples_sorted[0]:
        n_samples = end - start
        memcpy(sorted_samples + start, samples + start,
               n_samples * sizeof(SIZE_t))
        qsort(sorted_samples + start, n_samples, sizeof(SIZE_t),
              compare_SIZE_t)
        is_samples_sorted[0] = 1

    while (indptr_start < indptr_end and
           sorted_samples[start] > X_indices[indptr_start]):
        indptr_start += 1

    while (indptr_start < indptr_end and
           sorted_samples[end - 1] < X_indices[indptr_end - 1]):
        indptr_end -= 1

    cdef SIZE_t p = start
    cdef SIZE_t index
    cdef SIZE_t k
    cdef SIZE_t end_negative_ = start
    cdef SIZE_t start_positive_ = end

    while (p < end and indptr_start < indptr_end):
        # Find index of sorted_samples[p] in X_indices
        binary_search(X_indices, indptr_start, indptr_end,
                      sorted_samples[p], &k, &indptr_start)

        if k != -1:
             # If k != -1, we have found a non zero value

            if X_data[k] > 0:
                start_positive_ -= 1
                Xf[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)


            elif X_data[k] < 0:
                Xf[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1
        p += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void sparse_swap(SIZE_t* index_to_samples, SIZE_t* samples,
                             SIZE_t pos_1, SIZE_t pos_2) nogil:
    """Swap sample pos_1 and pos_2 preserving sparse invariant."""
    samples[pos_1], samples[pos_2] =  samples[pos_2], samples[pos_1]
    index_to_samples[samples[pos_1]] = pos_1
    index_to_samples[samples[pos_2]] = pos_2


cdef class BestSparseSplitter(BaseSparseSplitter):
    """Splitter for finding the best split, using the sparse data."""

    def __reduce__(self):
        return (BestSparseSplitter, (self.criterion,
                                     self.max_features,
                                     self.min_samples_leaf,
                                     self.min_weight_leaf,
                                     self.random_state,
                                     self.presort), self.__getstate__())

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best split on node samples[start:end], using sparse features

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef INT32_t* X_indices = self.X_indices
        cdef INT32_t* X_indptr = self.X_indptr
        cdef DTYPE_t* X_data = self.X_data

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t* sorted_samples = self.sorted_samples
        cdef SIZE_t* index_to_samples = self.index_to_samples
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        _init_split(&best, end)
        cdef double current_proxy_improvement = - INFINITY
        cdef double best_proxy_improvement = - INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p, tmp
        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value

        cdef SIZE_t p_next
        cdef SIZE_t p_prev
        cdef bint is_samples_sorted = 0  # indicate if sorted_samples is
                                         # initialized

        # We assume implicitly that end_positive = end and
        # start_negative = start
        cdef SIZE_t start_positive
        cdef SIZE_t end_negative

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]
                self.extract_nnz(current.feature,
                                 &end_negative, &start_positive,
                                 &is_samples_sorted)

                # Sort the positive and negative parts of `Xf`
                sort(Xf + start, samples + start, end_negative - start)
                sort(Xf + start_positive, samples + start_positive,
                     end - start_positive)

                # Update index_to_samples to take into account the sort
                for p in range(start, end_negative):
                    index_to_samples[samples[p]] = p
                for p in range(start_positive, end):
                    index_to_samples[samples[p]] = p

                # Add one or two zeros in Xf, if there is any
                if end_negative < start_positive:
                    start_positive -= 1
                    Xf[start_positive] = 0.

                    if end_negative != start_positive:
                        Xf[end_negative] = 0.
                        end_negative += 1

                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Evaluate all splits
                    self.criterion.reset()
                    p = start
                    # TODO: brent optimization?
                    while p < end:
                        if p + 1 != end_negative:
                            p_next = p + 1
                        else:
                            p_next = start_positive

                        while (p_next < end and
                               Xf[p_next] <= Xf[p] + FEATURE_THRESHOLD):
                            p = p_next
                            if p + 1 != end_negative:
                                p_next = p + 1
                            else:
                                p_next = start_positive


                        # (p_next >= end) or (X[samples[p_next], current.feature] >
                        #                     X[samples[p], current.feature])
                        p_prev = p
                        p = p_next
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p_prev], current.feature])


                        if p < end:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < min_samples_leaf) or
                                    ((end - current.pos) < min_samples_leaf)):
                                continue

                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not satisfied
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                    (self.criterion.weighted_n_right < min_weight_leaf)):
                                continue

                            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                            if current_proxy_improvement > best_proxy_improvement:
                                best_proxy_improvement = current_proxy_improvement
                                # sum of halves used to avoid infinite values
                                current.threshold = Xf[p_prev] / 2.0 + Xf[p] / 2.0

                                if ((current.threshold == Xf[p]) or
                                    (current.threshold == INFINITY) or
                                    (current.threshold == -INFINITY)):
                                    current.threshold = Xf[p_prev]

                                best = current

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            self.extract_nnz(best.feature, &end_negative, &start_positive,
                             &is_samples_sorted)

            self._partition(best.threshold, end_negative, start_positive,
                            best.pos)

            self.criterion.reset()
            self.criterion.update(best.pos)
            best.improvement = self.criterion.impurity_improvement(impurity)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0


cdef class RandomSparseSplitter(BaseSparseSplitter):
    """Splitter for finding a random split, using the sparse data."""

    def __reduce__(self):
        return (RandomSparseSplitter, (self.criterion,
                                       self.max_features,
                                       self.min_samples_leaf,
                                       self.min_weight_leaf,
                                       self.random_state,
                                       self.presort), self.__getstate__())

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find a random split on node samples[start:end], using sparse features

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef INT32_t* X_indices = self.X_indices
        cdef INT32_t* X_indptr = self.X_indptr
        cdef DTYPE_t* X_data = self.X_data

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t* sorted_samples = self.sorted_samples
        cdef SIZE_t* index_to_samples = self.index_to_samples
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        _init_split(&best, end)
        cdef double current_proxy_improvement = - INFINITY
        cdef double best_proxy_improvement = - INFINITY

        cdef DTYPE_t current_feature_value

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p, tmp
        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef SIZE_t partition_end

        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value

        cdef bint is_samples_sorted = 0  # indicate that sorted_samples is
                                         # inititialized

        # We assume implicitly that end_positive = end and
        # start_negative = start
        cdef SIZE_t start_positive
        cdef SIZE_t end_negative

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]

                self.extract_nnz(current.feature,
                                 &end_negative, &start_positive,
                                 &is_samples_sorted)

                # Add one or two zeros in Xf, if there is any
                if end_negative < start_positive:
                    start_positive -= 1
                    Xf[start_positive] = 0.

                    if end_negative != start_positive:
                        Xf[end_negative] = 0.
                        end_negative += 1

                # Find min, max in Xf[start:end_negative]
                min_feature_value = Xf[start]
                max_feature_value = min_feature_value

                for p in range(start, end_negative):
                    current_feature_value = Xf[p]

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                # Update min, max given Xf[start_positive:end]
                for p in range(start_positive, end):
                    current_feature_value = Xf[p]

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Draw a random threshold
                    current.threshold = rand_uniform(min_feature_value,
                                                     max_feature_value,
                                                     random_state)

                    if current.threshold == max_feature_value:
                        current.threshold = min_feature_value

                    # Partition
                    current.pos = self._partition(current.threshold,
                                                  end_negative,
                                                  start_positive,
                                                  start_positive +
                                                  (Xf[start_positive] == 0.))

                    # Reject if min_samples_leaf is not guaranteed
                    if (((current.pos - start) < min_samples_leaf) or
                            ((end - current.pos) < min_samples_leaf)):
                        continue

                    # Evaluate split
                    self.criterion.reset()
                    self.criterion.update(current.pos)

                    # Reject if min_weight_leaf is not satisfied
                    if ((self.criterion.weighted_n_left < min_weight_leaf) or
                            (self.criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        current.improvement = self.criterion.impurity_improvement(impurity)

                        self.criterion.children_impurity(&current.impurity_left,
                                                         &current.impurity_right)
                        best = current

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            if current.feature != best.feature:
                self.extract_nnz(best.feature, &end_negative, &start_positive,
                                 &is_samples_sorted)

                self._partition(best.threshold, end_negative, start_positive,
                                best.pos)

            self.criterion.reset()
            self.criterion.update(best.pos)
            best.improvement = self.criterion.impurity_improvement(impurity)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0
