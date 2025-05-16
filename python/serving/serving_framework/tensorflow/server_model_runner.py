# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements ModelRunner by forwarding to a TFserving instance.

Relies on the model being served by a TFserving instance running on localhost
unless a stub configured otherwise is provided.
"""

from collections.abc import Mapping, Set

from absl import logging
import grpc
import numpy as np
from typing_extensions import override

# pylint: disable = g-direct-tensorflow-import
# Importing protos should be safe anyway?
from serving.serving_framework import model_runner
from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


_HOSTPORT = "localhost:8500"


class ServerModelRunner(model_runner.ModelRunner):
  """ModelRunner implementation using grpc to TFserving."""

  def __init__(
      self, stub: prediction_service_pb2_grpc.PredictionServiceStub = None
  ):
    """Initializes the instance, with a local connection by default.

    Args:
      stub: A stub to use for the connection. If not provided, a default
        connection to localhost is established. This argument is intended for
        testing.
    """
    if stub is not None:
      self._stub = stub
      return
    credentials = grpc.local_channel_credentials(
        grpc.LocalConnectionType.LOCAL_TCP
    )
    channel = grpc.secure_channel(_HOSTPORT, credentials)
    self._stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  @override
  def run_model_multiple_output(
      self,
      model_input: Mapping[str, np.ndarray] | np.ndarray,
      *,
      model_name: str = "default",
      model_version: int | None = None,
      model_output_keys: Set[str],
  ) -> Mapping[str, np.ndarray]:
    """Runs a model on the given input and returns multiple outputs.

    Args:
      model_input: An array or map of arrays comprising the input tensors for
        the model. A bare array is keyed by "inputs".
      model_name: The name of the model to run.
      model_version: The version of the model to run. Uses default if None.
      model_output_keys: The desired model output keys.

    Returns:
      A mapping of model output keys to tensors.

    Raises:
      KeyError: If any of the model_output_keys are not found in the model
        output.
    """
    # If a bare np.ndarray was passed, it will be passed using the default
    # input key "inputs".
    # If a Mapping was passed, use the keys from that mapping.
    if isinstance(model_input, np.ndarray):
      logging.debug("Handling bare input tensor.")
      input_map = {"inputs": tensor_util.make_tensor_proto(model_input)}
    else:
      logging.debug("Handling input tensor map.")
      input_map = {
          k: tensor_util.make_tensor_proto(v) for k, v in model_input.items()
      }

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    if model_version is not None:
      request.model_spec.version.value = model_version
    for key, data in input_map.items():
      request.inputs[key].CopyFrom(data)

    logging.debug("Calling PredictionService.Predict")
    result = self._stub.Predict(request, timeout=60).outputs
    logging.debug("PredictionService.Predict returned.")
    # Check for expected keys in the result.
    result_keys = set(result.keys())
    missing_keys = model_output_keys - result_keys
    if missing_keys:
      raise KeyError(
          f"Model output keys {missing_keys} not found in model output. "
          f"Available keys: {result_keys}"
      )
    return {k: tensor_util.MakeNdarray(result[k]) for k in model_output_keys}
