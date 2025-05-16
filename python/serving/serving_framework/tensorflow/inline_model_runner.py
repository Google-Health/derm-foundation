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

"""Implements ModelRunner running a provided checkpoint in-process.

Runs the serving_default signature of the provided checkpoint.

Does not implement multi-model support.
"""

from collections.abc import Mapping, Set

import numpy as np
import tensorflow as tf
from typing_extensions import override

from serving.serving_framework import model_runner


class InlineModelRunner(model_runner.ModelRunner):
  """ModelRunner implementation using in-process tensorflow."""

  def __init__(self, model: tf.train.Checkpoint):
    self._model = model

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
        the model.
      model_name: The name of the model to run.
      model_version: The version of the model to run. Uses default if None.
      model_output_keys: The desired model output keys.

    Returns:
      A mapping of model output keys to tensors.
    """
    if model_name != "default" or model_version is not None:
      raise NotImplementedError(
          "InlineModelRunner does not support multiple models."
      )
    del model_name, model_version
    if isinstance(model_input, np.ndarray):
      tensor_input = tf.convert_to_tensor(model_input)
    else:
      tensor_input = {
          k: tf.convert_to_tensor(v) for k, v in model_input.items()
      }

    result = self._model.signatures["serving_default"](tensor_input)
    return {k: result[k].numpy() for k in model_output_keys}
