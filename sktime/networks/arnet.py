# -*- coding: utf-8 -*-
"""ARNet."""

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class _ARNetwork(BaseDeepNetwork):
    """An implementation of the Auto-Regressive Neural Network
    for time series [1].

    Parameters
    ----------
    n_layers : int, default = 1
        the number of hidden layers
    hidden_size : int, default = None

    Notes
    -----
    Adapted from source code
    https://github.com/ourownstory/AR-Net/blob/master/v0_1/model.py

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
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        n_layers=1,
        hidden_size=None,
    ):
        _check_dl_dependencies(severity="error")
        super(_ARNetwork, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape    : tuple of shape = (series_length (m), n_dimensions (d))
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        from tensorflow import keras

        input_layer = keras.layers.Input(input_shape)
        layer = keras.layers.Flatten()(input_layer)
        if self.hidden_size is None and self.n_layers > 1:
            self.hidden_size = input_shape
        if self.n_layers == 1:
            output_layer = keras.layers.Dense(1)
        else:
            layer = keras.layers.Dense(self.hidden_size, activation="relu")(layer)
            for _ in range(self.n_layers - 2):
                layer = keras.layers.Dense(self.hidden_size, activation="relu")(layer)

            output_layer = keras.layers.Dense(1)

        output_tensor = output_layer(layer)
        return input_layer, output_layer, output_tensor
