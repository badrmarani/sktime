# -*- coding: utf-8 -*-
"""Auto-Regressive Neural Network (AR-Net)."""

__author__ = ["Badr-Eddine Marani"]
__all__ = ["ARNetwork"]

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import check_random_state
from tensorflow import keras

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.deep_learning.base import BaseDeepForecaster
from sktime.networks.arnet import _ARNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class ARNetwork(BaseDeepForecaster):
    """Auto-Regressive Neural Network (AR-Net), as described in [1].

    Parameters
    ----------
    n_epochs: int, default = 2000
        the number of epochs to train the model
    batch_size: int, default = 16
        the number of samples per gradient update.
    random_state: int or None, default=None
        Seed for random number generation.
    verbose: boolean, default = False
        whether to output extra information
    loss: string, default="mean_squared_error"
        fit parameter for the keras model
    optimizer: keras.optimizers object, default `None`.
        when `None`, internally uses `keras.optimizers.Adam(0.01)`

    References
    ----------
    .. [1] Network originally defined in:
    @misc{triebe2019arnet,
        title={AR-Net: A simple Auto-Regressive Neural Network for time-series},
        author={Oskar Triebe and Nikolay Laptev and Ram Rajagopal},
        year={2019},
        eprint={1911.12436},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }

    Examples
    --------
    >>> from sktime.forecasting.deep_learning.arnet import ARNetwork
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> arnet = ARNetwork(n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> arnet.fit(y_train)  # doctest: +SKIP
    ARNetwork(...)
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "scitype:y": "univariate",
        "capability:pred_var": True,
        "capability:pred_int": True,
    }

    def __init__(
        self,
        ar_order=1,
        n_layers=1,
        hidden_size=1,
        n_epochs=20,
        batch_size=16,
        callbacks=None,
        verbose=2,
        loss="mean_squared_error",
        metrics=None,
        regularization=0.0,
        random_state=None,
        optimizer=None,
    ) -> None:
        _check_dl_dependencies(severity="error")
        super(ARNetwork, self).__init__()

        # assert (
        #     regularization and n_layers == 1
        # ), "The regularization is not implemented for deeper models."

        self.callbacks = callbacks
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ar_order = ar_order
        self.verbose = verbose
        self.loss = loss
        self.random_state = random_state
        self.optimizer = optimizer
        self.history = None
        self.metrics = metrics
        self.regularization = regularization if regularization else 0
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        ar_network = _ARNetwork(self.n_layers, self.hidden_size)
        (
            self.input_layer,
            self.output_layer,
            self.output_tensor,
        ) = ar_network.build_network(ar_order)
        print(self.input_layer, self.output_layer, self.output_tensor)

    def _arnet_loss_fn(self, y_true, y_pred):
        loss = keras.losses.mean_squared_error(y_true, y_pred)

        reg_fn = tf.zeros(shape=(1,))
        if self.regularization:
            abs_weights = tf.abs(self.output_layer.get_weights())
            reg_fn = tf.divide(2.0, 1.0 + tf.exp(-3.0 * abs_weights ** (1.0 / 3.0))) - 1

        return loss + self.regularization * tf.reduce_mean(reg_fn)

    def build_model(self):
        """Construct a complied, un-trained, keras model that is ready for training.

        Returns
        -------
        output: a compiled Keras model
        """
        tf.random.set_seed(self.random_state)

        metrics = ["mean_squared_error"] if self.metrics is None else self.metrics

        self.optimizer_ = (
            keras.optimizers.SGD(learning_rate=0.01)
            if self.optimizer is None
            else self.optimizer
        )

        model = keras.models.Model(inputs=self.input_layer, outputs=self.output_tensor)
        model.compile(
            loss=self.arnet_loss_fn,
            optimizer=self.optimizer_,
            metrics=metrics,
        )

        return model

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster on the training set `y`.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        if self.callbacks is None:
            self._callbacks = []

        # y = y.values()

        self._y = y

        n_timepoints = y.shape[0]

        _x_train = []
        _y_train = []
        for i in range(self.ar_order, n_timepoints):
            _x_train.append(y[i - self.ar_order : i])
            _y_train.append(y[i])

        _x_train = np.array(_x_train)
        _y_train = np.array(_y_train)
        _x_train = _x_train.reshape(*_x_train.shape, 1)
        _y_train = _y_train.reshape(*_y_train.shape, 1)

        check_random_state(self.random_state)
        self.input_shape = self.ar_order
        self.model_ = self.build_model()
        if self.verbose:
            self.model_.summary()

        self.history = self.model_.fit(
            _x_train,
            _y_train,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self._callbacks,
        )

        return self

    def _predict(self, fh, X=None):
        # n_timepoints = fh.shape[0]
        n_timepoints = 36
        y_pred = []
        for i in range(n_timepoints):
            yp = self._y[-self.ar_order :]
            print("qsdqsd", yp.shape)
            y_pred.append(self.model_.predict(yp))

            np.append(self._y, y_pred[-1])

        y_pred = np.array(y_pred)
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        param1 = {
            "n_epochs": 10,
            "batch_size": 4,
        }

        param2 = {
            "n_epochs": 12,
            "batch_size": 6,
        }

        return [param1, param2]
