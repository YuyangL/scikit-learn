"""Losses and corresponding default initial estimators for gradient boosting
decision trees.
"""

from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from scipy.special import expit

from ..tree._tree import TREE_LEAF
from ..utils.fixes import logsumexp
from ..utils.stats import _weighted_percentile
from ..dummy import DummyClassifier
from ..dummy import DummyRegressor
from warnings import warn


class LossFunction(metaclass=ABCMeta):
    """Abstract base class for various loss functions.

    Parameters
    ----------
    n_classes : int
        Number of classes.

    Attributes
    ----------
    K : int
        The number of regression trees to be induced;
        1 for regression and binary classification;
        ``n_classes`` for multi-class classification.
    """

    is_multi_class = False

    def __init__(self, n_classes):
        self.K = n_classes

    def init_estimator(self):
        """Default ``init`` estimator for loss function. """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the loss.

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            True labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves).

        sample_weight : 1d array, shape (n_samples,), optional
            Sample weights.
        """

    @abstractmethod
    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the negative gradient.

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            The target labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """

    def update_terminal_regions(self, tree, X, y, residual, raw_predictions,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0,
                                # Extra kwarg for bij prediction
                                tb=None,
                                bij_novelty=None):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.

        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : 2d array, shape (n, m)
            The data array.
        y : 1d array, shape (n,)
            The target labels.
        residual : 1d array, shape (n,)
            The residuals (usually the negative gradient).
        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : 1d array, shape (n,)
            The weight of each sample.
        sample_mask : 1d array, shape (n,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default=0
            The index of the estimator being updated.
        """

        # Recall last D is useless Tree.max_n_classes and is 1
        prediction = tree.predict(X,
                                  # Extra kwarg
                                  tb=tb)[..., 0]
        # Manually reset bij novelty values outside [-1/2*3, 2/3*3] with lenient margin multiplier 3
        if tb is not None and bij_novelty in ('excl', 'exclude', 'reset'):
            if bij_novelty in ('excl', 'exclude'): warn('\nNaN bij novelties are set to 0, i.e. no prediction')
            mask = np.zeros(prediction.shape[1])
            prediction_mask = raw_predictions + learning_rate*prediction
            for i in range(prediction.shape[0]):
                if max(prediction_mask[i]) > 2. or min(prediction_mask[i]) < -1.5: prediction[i] = mask

        # update predictions
        # Inplace update only works with += / *= / /= /-= and not for number or string
        if len(y.shape) == 1:
            raw_predictions[:, k] += learning_rate*prediction.ravel()
        # Else if multioutputs, recall raw_predictions had shape (n_samples, n_outputs)
        # and prediction has the same shape
        else:
            raw_predictions += learning_rate*prediction

        del prediction

    @abstractmethod
    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        """Template method for updating terminal regions (i.e., leaves)."""

    @abstractmethod
    def get_init_raw_predictions(self, X, estimator):
        """Return the initial raw predictions.

        Parameters
        ----------
        X : 2d array, shape (n_samples, n_features)
            The data array.
        estimator : estimator instance
            The estimator to use to compute the predictions.

        Returns
        -------
        raw_predictions : 2d array, shape (n_samples, K)
            The initial raw predictions. K is equal to 1 for binary
            classification and regression, and equal to the number of classes
            for multiclass classification. ``raw_predictions`` is casted
            into float64.
        """
        pass


class RegressionLossFunction(LossFunction, metaclass=ABCMeta):
    """Base class for regression loss functions.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    """
    def __init__(self, n_classes):
        if n_classes != 1:
            raise ValueError("``n_classes`` must be 1 for regression but "
                             "was %r" % n_classes)
        super().__init__(n_classes)

    def check_init_estimator(self, estimator):
        """Make sure estimator has the required fit and predict methods.

        Parameters
        ----------
        estimator : estimator instance
            The init estimator to check.
        """
        if not (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')):
            raise ValueError(
                "The init parameter must be a valid estimator and "
                "support both fit and predict."
            )

    def get_init_raw_predictions(self, X, estimator):
        predictions = estimator.predict(X)
        return predictions.reshape(-1, 1).astype(np.float64)


class LeastSquaresError(RegressionLossFunction):
    """Loss function for least squares (LS) estimation.
    Terminal regions do not need to be updated for least squares.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    """

    def init_estimator(self):
        return DummyRegressor(strategy='mean')

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the least squares loss.

        Parameters
        ----------
        y : 1/2d array, shape (n_samples,) or (n_samples, n_outputs)
            True labels.

        raw_predictions : 2d array, shape (n_samples, K) or (n_samples, n_outputs)
            The raw_predictions (i.e. values from the tree leaves).

        sample_weight : 1d array, shape (n_samples,), optional
            Sample weights.
        """
        if len(y.shape) == 1:
            if sample_weight is None:
                return np.mean((y - raw_predictions.ravel()) ** 2)
            else:
                return (1 / sample_weight.sum() * np.sum(
                    sample_weight * ((y - raw_predictions.ravel()) ** 2)))

        # Else if multioutputs, take the Frobenius norm of the error along outputs (columns)
        else:
            if sample_weight is None:
                return np.mean(np.linalg.norm(y - raw_predictions, axis=1)**2.)
            else:
                return (1. / sample_weight.sum() * np.sum(
                    sample_weight *
                    (np.linalg.norm(y - raw_predictions, axis=1) ** 2.)))

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the negative gradient.

        Parameters
        ----------
        y : 1/2d array, shape (n_samples,) or (n_samples, n_outputs)
            The target labels.

        raw_predictions : 1/2d array, shape (n_samples,) or (n_samples, n_outputs)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        if len(y.shape) == 1:
            return y - raw_predictions.ravel()
        # Else if multioutputs, return a n_outputs residual
        else:
            return y - raw_predictions

    def update_terminal_regions(self, tree, X, y, residual, raw_predictions,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0,
                                # Extra kwarg for bij prediction
                                tb=None,
                                bij_novelty=None):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.

        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : 2d array, shape (n, m)
            The data array.
        y : 1d array, shape (n,)
            The target labels.
        residual : 1d array, shape (n,)
            The residuals (usually the negative gradient).
        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : 1d array, shape (n,)
            The weight of each sample.
        sample_mask : 1d array, shape (n,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default=0
            The index of the estimator being updated.
        """

        # Recall last D is useless Tree.max_n_classes and is 1
        prediction = tree.predict(X,
                                  # Extra kwarg
                                  tb=tb)[..., 0]
        # Manually remove/reset bij novelty values outside [-1/2*3, 2/3*3] with lenient margin multiplier 3
        if tb is not None and bij_novelty in ('excl', 'exclude', 'reset'):
            # Mask containing values depending on either removal or reset
            if bij_novelty in ('excl', 'exclude'): warn('\nNaN bij novelties are set to 0, i.e. no prediction')
            mask = np.zeros(prediction.shape[0])
            # Go through every sample and reset the whole sample that exceeds the bound
            for i in range(prediction.shape[0]):
                if max(prediction[i]) > 2. or min(prediction[i]) < -1.5: prediction[i] = mask

        # update predictions
        # Inplace update only works with += / *= / /= /-= and not for number or string
        if len(y.shape) == 1:
            raw_predictions[:, k] += learning_rate * prediction.ravel()
        # Else if multioutputs, recall raw_predictions had shape (n_samples, n_outputs)
        # and prediction has the same shape
        else:
            raw_predictions += learning_rate*prediction

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        pass


class LeastAbsoluteError(RegressionLossFunction):
    """Loss function for least absolute deviation (LAD) regression.

    Parameters
    ----------
    n_classes : int
        Number of classes
    """
    def init_estimator(self):
        return DummyRegressor(strategy='quantile', quantile=.5)

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the least absolute error.

        Parameters
        ----------
        y : array, shape (n_samples,)
            True labels.

        raw_predictions : array, shape (n_samples, K)
            The raw_predictions (i.e. values from the tree leaves).

        sample_weight : 1d array, shape (n_samples,), optional
            Sample weights.
        """
        if sample_weight is None:
            return np.abs(y - raw_predictions.ravel()).mean()
        else:
            return (1 / sample_weight.sum() * np.sum(
                sample_weight * np.abs(y - raw_predictions.ravel())))

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the negative gradient.

        1.0 if y - raw_predictions > 0.0 else -1.0

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            The target labels.

        raw_predictions : array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        raw_predictions = raw_predictions.ravel()
        return 2 * (y - raw_predictions > 0) - 1

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        """LAD updates terminal regions to median estimates."""
        terminal_region = np.where(terminal_regions == leaf)[0]
        sample_weight = sample_weight.take(terminal_region, axis=0)
        diff = (y.take(terminal_region, axis=0) -
                raw_predictions.take(terminal_region, axis=0))
        tree.value[leaf, 0, 0] = _weighted_percentile(diff, sample_weight,
                                                      percentile=50)

class HuberLossFunction(RegressionLossFunction):
    """Huber loss function for robust regression.

    M-Regression proposed in Friedman 2001.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    Parameters
    ----------
    n_classes : int
        Number of classes.

    alpha : float, default=0.9
        Percentile at which to extract score.
    """

    def __init__(self, n_classes, alpha=0.9):
        super().__init__(n_classes)
        self.alpha = alpha
        self.gamma = None

    def init_estimator(self):
        return DummyRegressor(strategy='quantile', quantile=.5)

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the Huber loss.

        Parameters
        ----------
        y : 1/2d array, shape (n_samples,) or (n_samples, n_outputs)
            True labels.

        raw_predictions : 2d array, shape (n_samples, K) or (n_samples, n_outputs)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : 1d array, shape (n_samples,), optional
            Sample weights.
        """
        if len(y.shape) == 1:
            raw_predictions = raw_predictions.ravel()

        diff = y - raw_predictions
        gamma = self.gamma
        if gamma is None:
            if sample_weight is None:
                gamma = np.percentile(np.abs(diff), self.alpha * 100, axis=0)
            else:
                if len(y.shape) == 1:
                    gamma = _weighted_percentile(np.abs(diff), sample_weight,
                                             self.alpha * 100)
                # Else if multioutputs
                else:
                    gamma = np.empty(y.shape[1])
                    # _weighted_percentile() sort array along 1 axis. Thus retrieve gamma 1 output at a time
                    for i in range(y.shape[1]):
                        gamma[i] = _weighted_percentile(np.abs(diff[:, i]), sample_weight,
                                                     self.alpha*100)

        if len(y.shape) == 1:
            gamma_mask = np.abs(diff) <= gamma
            if sample_weight is None:
                sq_loss = np.sum(0.5*diff[gamma_mask]**2)
                lin_loss = np.sum(gamma*(np.abs(diff[~gamma_mask]) -
                                         gamma/2))
                loss = (sq_loss + lin_loss)/y.shape[0]
            else:
                sq_loss = np.sum(0.5*sample_weight[gamma_mask]*
                                 diff[gamma_mask]**2)
                lin_loss = np.sum(gamma*sample_weight[~gamma_mask]*
                                  (np.abs(diff[~gamma_mask]) - gamma/2))
                loss = (sq_loss + lin_loss)/sample_weight.sum()

        # Else if multioutputs
        else:
            # Again, create mask 1 output at a time
            gamma_mask = np.empty_like(y, dtype=np.bool)
            for i in range(y.shape[1]):
                gamma_mask[:, i] = np.abs(diff[:, i]) <= gamma[i]

            if sample_weight is None:
                sq_loss = np.sum(0.5 * diff[gamma_mask] ** 2)
                lin_loss = np.sum(gamma * (np.abs(diff[~gamma_mask]) -
                                           gamma / 2))
                # Output-averaged loss
                loss = (sq_loss + lin_loss) / y.shape[0]/y.shape[1]
            # Else if sample_weight is provided
            else:
                # sample_weight has shape (n_samples,) while gamma_mask has shape (n_samples, n_outputs),
                # thus aggregate 1 output at a time
                sq_loss, lin_loss, loss = 0., 0., 0.
                for i in range(y.shape[1]):
                    sq_loss += np.sum(0.5 * sample_weight[gamma_mask[:, i]] *
                                     diff[gamma_mask[:, i], i] ** 2)
                    lin_loss += np.sum(gamma[i] * sample_weight[~gamma_mask[:, i]] *
                                      (np.abs(diff[~gamma_mask[:, i], i]) - gamma[i] / 2))
                    loss += (sq_loss + lin_loss) / sample_weight.sum()

                # Output-averaged loss
                loss /= y.shape[1]

        del gamma_mask
        return loss

    def negative_gradient(self, y, raw_predictions, sample_weight=None,
                          **kargs):
        """Compute the negative gradient.

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            The target labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.

        sample_weight : 1d array, shape (n_samples,), optional
            Sample weights.
        """
        if len(y.shape) == 1:
            raw_predictions = raw_predictions.ravel()

        diff = y - raw_predictions
        if len(y.shape) == 1:
            if sample_weight is None:
                gamma = np.percentile(np.abs(diff), self.alpha * 100)
            else:
                gamma = _weighted_percentile(np.abs(diff), sample_weight,
                                             self.alpha * 100)

            gamma_mask = np.abs(diff) <= gamma
            residual = np.empty((y.shape[0],), dtype=np.float64)
            residual[gamma_mask] = diff[gamma_mask]
            residual[~gamma_mask] = gamma * np.sign(diff[~gamma_mask])
            self.gamma = gamma
        else:
            # gamma is a list equal number of outputs
            if sample_weight is None:
                gamma = np.percentile(np.abs(diff), self.alpha*100, axis=0)
            else:
                # _weighted_percentile() sort array along 1 axis. Thus retrieve gamma 1 output at a time
                gamma = np.empty(y.shape[1])
                for i in range(y.shape[1]):
                    gamma[i] = _weighted_percentile(np.abs(diff[:, i]), sample_weight,
                                                 self.alpha*100)

            # gamma_mask is also multioutputs, thus shape (n_samples, n_outputs)
            gamma_mask = np.empty_like(y, dtype=np.bool)
            # Again, create mask 1 output at a time
            for i in range(y.shape[1]):
                gamma_mask[:, i] = np.abs(diff[:, i]) <= gamma[i]

            residual = np.empty_like(y, dtype=np.float64)
            residual[gamma_mask] = diff[gamma_mask]
            # gamma[n_outputs]*diff[n_samples x n_outputs] is element-wise product.
            # However, casting a mask will trim the array thus the above can't work.
            # Therefore, go through every output
            for i in range(y.shape[1]):
                residual[~gamma_mask[:, i], i] = gamma[i]*np.sign(diff[~gamma_mask[:, i], i])

            # self.gamma is a list equalling number of outputs
            self.gamma = gamma

        return residual

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight,
                                # Extra kwarg
                                tb=None):
        terminal_region = np.where(terminal_regions == leaf)[0]
        # axis=0 means take terminal_region row and all columns at that row if nD array
        sample_weight = sample_weight.take(terminal_region, axis=0)
        gamma = self.gamma
        # print(y.shape)
        # print(terminal_region.shape)
        # print(raw_predictions.shape)
        diff = (y.take(terminal_region, axis=0)
                - raw_predictions.take(terminal_region, axis=0))
        # Take terminal region indices of Tij too it provided
        if tb is not None:
            tb = tb.take(terminal_region, axis=0)
            # Multiple samples taken, which is most likely the case,
            # reshape Tij from shape (n_samples, n_outputs, n_bases) to (n_samples*n_outputs, n_bases),
            # so that g can be found via g = Tij^T*bij
            if len(tb.shape) == 3: tb = tb.reshape((-1, tb.shape[2]))
            bij_residual = np.empty_like(diff)

        # If single output
        if len(y.shape) == 1:
            median = _weighted_percentile(diff, sample_weight, percentile=50)
        # Else if multioutputs
        else:
            # Calculate median for every output and 1 by 1
            median = np.empty(y.shape[1])
            for i in range(y.shape[1]):
                median[i] = _weighted_percentile(diff[:, i], sample_weight, percentile=50)

        # This is an array of shape (n_samples, n_outputs) if multioutputs,
        # otherwise of shape (n_samples,)
        diff_minus_median = diff - median
        if len(y.shape) == 1:
            # Recall whenever Tree.value is called, value has been wrapped to a 3D array
            # of shape (n_samples, n_outputs, max_n_classes), where max_n_classes = 1 for regression
            tree.value[leaf, 0] = median + np.mean(
                np.sign(diff_minus_median) *
                np.minimum(np.abs(diff_minus_median), gamma))
        # Else if multioutputs
        else:
            if tb is None:
                # Go through every output
                for i in range(y.shape[1]):
                    tree.value[leaf, i] = median[i] + np.mean(
                            np.sign(diff_minus_median[:, i])*
                            np.minimum(np.abs(diff_minus_median[:, i]), gamma[i]))
            else:
                for i in range(y.shape[1]):
                    bij_residual[:, i] = median[i] + np.mean(
                            np.sign(diff_minus_median[:, i])*
                            np.minimum(np.abs(diff_minus_median[:, i]), gamma[i]))

                # Least squares to find g and inplace update Tree.value at this leaf
                # res = np.linalg.lstsq(tb, bij_residual.ravel(), rcond=None)[0]
                # print(res.shape)
                # print(tree.value[leaf].shape)
                tree.value[leaf, :, 0] = np.linalg.lstsq(tb, bij_residual.ravel(), rcond=None)[0]
                del bij_residual


class QuantileLossFunction(RegressionLossFunction):
    """Loss function for quantile regression.

    Quantile regression allows to estimate the percentiles
    of the conditional distribution of the target.

    Parameters
    ----------
    n_classes : int
        Number of classes.

    alpha : float, optional (default = 0.9)
        The percentile.
    """
    def __init__(self, n_classes, alpha=0.9):
        super().__init__(n_classes)
        self.alpha = alpha
        self.percentile = alpha * 100

    def init_estimator(self):
        return DummyRegressor(strategy='quantile', quantile=self.alpha)

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the Quantile loss.

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            True labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : 1d array, shape (n_samples,), optional
            Sample weights.
        """
        raw_predictions = raw_predictions.ravel()
        diff = y - raw_predictions
        alpha = self.alpha

        mask = y > raw_predictions
        if sample_weight is None:
            loss = (alpha * diff[mask].sum() -
                    (1 - alpha) * diff[~mask].sum()) / y.shape[0]
        else:
            loss = ((alpha * np.sum(sample_weight[mask] * diff[mask]) -
                    (1 - alpha) * np.sum(sample_weight[~mask] *
                                         diff[~mask])) / sample_weight.sum())
        return loss

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the negative gradient.

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            The target labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw_predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        alpha = self.alpha
        raw_predictions = raw_predictions.ravel()
        mask = y > raw_predictions
        return (alpha * mask) - ((1 - alpha) * ~mask)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        terminal_region = np.where(terminal_regions == leaf)[0]
        diff = (y.take(terminal_region, axis=0)
                - raw_predictions.take(terminal_region, axis=0))
        sample_weight = sample_weight.take(terminal_region, axis=0)

        val = _weighted_percentile(diff, sample_weight, self.percentile)
        tree.value[leaf, 0] = val


class ClassificationLossFunction(LossFunction, metaclass=ABCMeta):
    """Base class for classification loss functions. """

    def _raw_prediction_to_proba(self, raw_predictions):
        """Template method to convert raw predictions into probabilities.

        Parameters
        ----------
        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        Returns
        -------
        probas : 2d array, shape (n_samples, K)
            The predicted probabilities.
        """

    @abstractmethod
    def _raw_prediction_to_decision(self, raw_predictions):
        """Template method to convert raw predictions to decisions.

        Parameters
        ----------
        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        Returns
        -------
        encoded_predictions : 2d array, shape (n_samples, K)
            The predicted encoded labels.
        """

    def check_init_estimator(self, estimator):
        """Make sure estimator has fit and predict_proba methods.

        Parameters
        ----------
        estimator : estimator instance
            The init estimator to check.
        """
        if not (hasattr(estimator, 'fit') and
                hasattr(estimator, 'predict_proba')):
            raise ValueError(
                "The init parameter must be a valid estimator "
                "and support both fit and predict_proba."
            )


class BinomialDeviance(ClassificationLossFunction):
    """Binomial deviance loss function for binary classification.

    Binary classification is a special case; here, we only need to
    fit one tree instead of ``n_classes`` trees.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    """
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes; got {1:d} class(es)"
                             .format(self.__class__.__name__, n_classes))
        # we only need to fit one tree for binary clf.
        super().__init__(n_classes=1)

    def init_estimator(self):
        # return the most common class, taking into account the samples
        # weights
        return DummyClassifier(strategy='prior')

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the deviance (= 2 * negative log-likelihood).

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            True labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : 1d array , shape (n_samples,), optional
            Sample weights.
        """
        # logaddexp(0, v) == log(1.0 + exp(v))
        raw_predictions = raw_predictions.ravel()
        if sample_weight is None:
            return -2 * np.mean((y * raw_predictions) -
                                np.logaddexp(0, raw_predictions))
        else:
            return (-2 / sample_weight.sum() * np.sum(
                sample_weight * ((y * raw_predictions) -
                                 np.logaddexp(0, raw_predictions))))

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the residual (= negative gradient).

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            True labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw_predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        return y - expit(raw_predictions.ravel())

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        """Make a single Newton-Raphson step.

        our node estimate is given by:

            sum(w * (y - prob)) / sum(w * prob * (1 - prob))

        we take advantage that: y - prob = residual
        """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)
        denominator = np.sum(sample_weight *
                             (y - residual) * (1 - y + residual))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _raw_prediction_to_proba(self, raw_predictions):
        proba = np.ones((raw_predictions.shape[0], 2), dtype=np.float64)
        proba[:, 1] = expit(raw_predictions.ravel())
        proba[:, 0] -= proba[:, 1]
        return proba

    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)
        return np.argmax(proba, axis=1)

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        proba_pos_class = probas[:, 1]
        eps = np.finfo(np.float32).eps
        proba_pos_class = np.clip(proba_pos_class, eps, 1 - eps)
        # log(x / (1 - x)) is the inverse of the sigmoid (expit) function
        raw_predictions = np.log(proba_pos_class / (1 - proba_pos_class))
        return raw_predictions.reshape(-1, 1).astype(np.float64)


class MultinomialDeviance(ClassificationLossFunction):
    """Multinomial deviance loss function for multi-class classification.

    For multi-class classification we need to fit ``n_classes`` trees at
    each stage.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    """

    is_multi_class = True

    def __init__(self, n_classes):
        if n_classes < 3:
            raise ValueError("{0:s} requires more than 2 classes.".format(
                self.__class__.__name__))
        super().__init__(n_classes)

    def init_estimator(self):
        return DummyClassifier(strategy='prior')

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the Multinomial deviance.

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            True labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : 1d array, shape (n_samples,), optional
            Sample weights.
        """
        # create one-hot label encoding
        Y = np.zeros((y.shape[0], self.K), dtype=np.float64)
        for k in range(self.K):
            Y[:, k] = y == k

        if sample_weight is None:
            return np.sum(-1 * (Y * raw_predictions).sum(axis=1) +
                          logsumexp(raw_predictions, axis=1))
        else:
            return np.sum(
                -1 * sample_weight * (Y * raw_predictions).sum(axis=1) +
                logsumexp(raw_predictions, axis=1))

    def negative_gradient(self, y, raw_predictions, k=0, **kwargs):
        """Compute negative gradient for the ``k``-th class.

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            The target labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw_predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.

        k : int, optional default=0
            The index of the class.
        """
        return y - np.nan_to_num(np.exp(raw_predictions[:, k] -
                                        logsumexp(raw_predictions, axis=1)))

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        """Make a single Newton-Raphson step. """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)
        numerator *= (self.K - 1) / self.K

        denominator = np.sum(sample_weight * (y - residual) *
                             (1 - y + residual))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _raw_prediction_to_proba(self, raw_predictions):
        return np.nan_to_num(
            np.exp(raw_predictions -
                   (logsumexp(raw_predictions, axis=1)[:, np.newaxis])))

    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)
        return np.argmax(proba, axis=1)

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        eps = np.finfo(np.float32).eps
        probas = np.clip(probas, eps, 1 - eps)
        raw_predictions = np.log(probas).astype(np.float64)
        return raw_predictions


class ExponentialLoss(ClassificationLossFunction):
    """Exponential loss function for binary classification.

    Same loss as AdaBoost.

    References
    ----------
    Greg Ridgeway, Generalized Boosted Models: A guide to the gbm package, 2007

    Parameters
    ----------
    n_classes : int
        Number of classes.
    """
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes; got {1:d} class(es)"
                             .format(self.__class__.__name__, n_classes))
        # we only need to fit one tree for binary clf.
        super().__init__(n_classes=1)

    def init_estimator(self):
        return DummyClassifier(strategy='prior')

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the exponential loss

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            True labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : 1d array, shape (n_samples,), optional
            Sample weights.
        """
        raw_predictions = raw_predictions.ravel()
        if sample_weight is None:
            return np.mean(np.exp(-(2. * y - 1.) * raw_predictions))
        else:
            return (1.0 / sample_weight.sum() * np.sum(
                sample_weight * np.exp(-(2 * y - 1) * raw_predictions)))

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the residual (= negative gradient).

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            True labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        y_ = -(2. * y - 1.)
        return y_ * np.exp(y_ * raw_predictions.ravel())

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        terminal_region = np.where(terminal_regions == leaf)[0]
        raw_predictions = raw_predictions.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        y_ = 2. * y - 1.

        numerator = np.sum(y_ * sample_weight * np.exp(-y_ * raw_predictions))
        denominator = np.sum(sample_weight * np.exp(-y_ * raw_predictions))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _raw_prediction_to_proba(self, raw_predictions):
        proba = np.ones((raw_predictions.shape[0], 2), dtype=np.float64)
        proba[:, 1] = expit(2.0 * raw_predictions.ravel())
        proba[:, 0] -= proba[:, 1]
        return proba

    def _raw_prediction_to_decision(self, raw_predictions):
        return (raw_predictions.ravel() >= 0).astype(np.int)

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        proba_pos_class = probas[:, 1]
        eps = np.finfo(np.float32).eps
        proba_pos_class = np.clip(proba_pos_class, eps, 1 - eps)
        # according to The Elements of Statistical Learning sec. 10.5, the
        # minimizer of the exponential loss is .5 * log odds ratio. So this is
        # the equivalent to .5 * binomial_deviance.get_init_raw_predictions()
        raw_predictions = .5 * np.log(proba_pos_class / (1 - proba_pos_class))
        return raw_predictions.reshape(-1, 1).astype(np.float64)


LOSS_FUNCTIONS = {
    'ls': LeastSquaresError,
    'lad': LeastAbsoluteError,
    'huber': HuberLossFunction,
    'quantile': QuantileLossFunction,
    'deviance': None,  # for both, multinomial and binomial
    'exponential': ExponentialLoss,
}
