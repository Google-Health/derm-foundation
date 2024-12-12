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

from unittest import mock

import numpy as np
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized
from serving.serving_framework import inline_model_runner


def tensor_equal(first: tf.Tensor, second: tf.Tensor) -> bool:
  """Equality check for tensors.

  As implemented only confirms that the values are equal at all shared
  addresses.
  It is expected that some tensors of different shape may be misidentified as
  equal.

  Args:
    first: First tensor to compare.
    second: Second tensor to compare.

  Returns:
    True if the tensors are equal, False otherwise.
  """
  return bool(tf.math.reduce_all(tf.equal(first, second)))


class InlineModelRunnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._model = mock.create_autospec(tf.train.Checkpoint, instance=True)
    self._runner = inline_model_runner.InlineModelRunner(model=self._model)

  def test_singleton_input(self):
    # Setup test values.
    input_np = np.array([[0.5] * 3] * 3, dtype=np.float32)
    input_tensor = tf.constant(input_np, shape=(3, 3), dtype=tf.float32)
    output_np = np.ones((3, 2), dtype=np.float32)
    output_tensor = tf.constant(output_np, shape=(3, 2), dtype=tf.float32)
    # Setup mock model.
    model_signature_mock = mock.MagicMock()
    model_signature_mock.return_value = {"output_0": output_tensor}
    self._model.signatures = {"serving_default": model_signature_mock}

    result = self._runner.run_model(input_np)

    self.assertLen(model_signature_mock.call_args_list, 1)
    self.assertTrue(
        tensor_equal(model_signature_mock.call_args[0][0], input_tensor),
        "Tensor passed to model values do not match input.",
    )
    # Missing a check for the shape of the input tensor.
    np.testing.assert_array_equal(result, output_np)

  def test_map_input(self):
    # Setup test values.
    input_np_map = {
        "a": np.array([[0.5] * 3] * 3, dtype=np.float32),
        "b": np.array([[0.25] * 3] * 3, dtype=np.float32),
    }
    input_tensor_map = {
        label: tf.constant(input_np_map[label], shape=(3, 3), dtype=tf.float32)
        for label in input_np_map
    }
    output_np = np.ones((3, 2), dtype=np.float32)
    output_tensor = tf.constant(output_np, shape=(3, 2), dtype=tf.float32)
    # Setup mock model.
    model_signature_mock = mock.MagicMock()
    model_signature_mock.return_value = {"output_0": output_tensor}
    self._model.signatures = {"serving_default": model_signature_mock}

    result = self._runner.run_model(input_np_map)

    self.assertLen(model_signature_mock.call_args_list, 1)
    self.assertSameElements(
        model_signature_mock.call_args[0][0].keys(),
        input_np_map.keys(),
        "Input tensor map keys don't match argument keys.",
    )
    self.assertTrue(
        all([
            tensor_equal(
                model_signature_mock.call_args[0][0][key], input_tensor_map[key]
            )
            for key in input_tensor_map
        ]),
        "Tensor passed to model values do not match input.",
    )
    # Missing a check for the shape of the input tensor.
    np.testing.assert_array_equal(result, output_np)

  def test_batch_singleton_input(self):
    # Setup test values.
    input_nps = [
        np.array([[0.5] * 3] * 3, dtype=np.float32),
        np.array([[0.25] * 3] * 3, dtype=np.float32),
    ]
    input_tensors = [
        tf.constant(input_np, shape=(3, 3), dtype=tf.float32)
        for input_np in input_nps
    ]
    output_nps = [
        np.ones((3, 2), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
    ]
    output_tensors = [
        tf.constant(output_np, shape=(3, 2), dtype=tf.float32)
        for output_np in output_nps
    ]
    # Setup mock model.
    model_signature_mock = mock.MagicMock()
    model_signature_mock.side_effect = [
        {"output_0": output_tensor} for output_tensor in output_tensors
    ]
    self._model.signatures = {"serving_default": model_signature_mock}

    result = self._runner.batch_model(input_nps)

    self.assertLen(model_signature_mock.call_args_list, len(input_nps))
    for input_tensor, call in zip(
        input_tensors, model_signature_mock.call_args_list
    ):
      self.assertTrue(
          tensor_equal(call[0][0], input_tensor),
          "Tensor passed to model values do not match input.",
      )
    # Missing a check for the shape of the input tensor.
    for result_np, output_np in zip(result, output_nps):
      np.testing.assert_array_equal(result_np, output_np)

  def test_batch_map_input(self):
    # Setup test values.
    input_np_maps = [
        {
            "a": np.array([[0.5] * 3] * 3, dtype=np.float32),
            "b": np.array([[0.25] * 3] * 3, dtype=np.float32),
        },
        {
            "a": np.array([[0.25] * 3] * 3, dtype=np.float32),
            "b": np.array([[0.5] * 3] * 3, dtype=np.float32),
        },
    ]
    input_tensor_maps = []
    for input_np_map in input_np_maps:
      input_tensor_maps.append({
          label: tf.constant(
              input_np_map[label], shape=(3, 3), dtype=tf.float32
          )
          for label in input_np_map
      })
    output_nps = [
        np.ones((3, 2), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
    ]
    output_tensors = [
        tf.constant(output_np, shape=(3, 2), dtype=tf.float32)
        for output_np in output_nps
    ]
    # Setup mock model.
    model_signature_mock = mock.MagicMock()
    model_signature_mock.side_effect = [
        {"output_0": output_tensor} for output_tensor in output_tensors
    ]
    self._model.signatures = {"serving_default": model_signature_mock}

    result = self._runner.batch_model(input_np_maps)

    self.assertLen(model_signature_mock.call_args_list, len(input_np_maps))
    for input_tensor_map, call in zip(
        input_tensor_maps, model_signature_mock.call_args_list
    ):
      self.assertTrue(
          all([
              tensor_equal(call[0][0][key], input_tensor_map[key])
              for key in input_tensor_map
          ]),
          "Tensor passed to model values do not match input.",
      )
    # Missing a check for the shape of the input tensor.
    for result_np, output_np in zip(result, output_nps):
      np.testing.assert_array_equal(result_np, output_np)

  def test_keyed_output(self):
    # Setup test values.
    input_np = np.array([[0.5] * 3] * 3, dtype=np.float32)
    input_tensor = tf.constant(input_np, shape=(3, 3), dtype=tf.float32)
    output_np = np.ones((3, 2), dtype=np.float32)
    output_tensor = tf.constant(output_np, shape=(3, 2), dtype=tf.float32)
    surplus_np = np.array([[0.1] * 3] * 2, dtype=np.float32)
    surplus_tensor = tf.constant(surplus_np, shape=(3, 2), dtype=tf.float32)
    # Setup mock model.
    model_signature_mock = mock.MagicMock()
    model_signature_mock.return_value = {
        "output_a": output_tensor,
        "output_b": surplus_tensor,
    }
    self._model.signatures = {"serving_default": model_signature_mock}

    result = self._runner.run_model(input_np, model_output_key="output_a")

    self.assertLen(model_signature_mock.call_args_list, 1)
    self.assertTrue(
        tensor_equal(model_signature_mock.call_args[0][0], input_tensor),
        "Tensor passed to model values do not match input.",
    )
    # Missing a check for the shape of the input tensor.
    np.testing.assert_array_equal(result, output_np)

  def test_multi_output(self):
    # Setup test values.
    input_np = np.array([[0.5] * 3] * 3, dtype=np.float32)
    input_tensor = tf.constant(input_np, shape=(3, 3), dtype=tf.float32)
    output_np = np.ones((3, 2), dtype=np.float32)
    output_tensor = tf.constant(output_np, shape=(3, 2), dtype=tf.float32)
    second_out_np = np.array([[0.7] * 3] * 3, dtype=np.float32)
    second_out_tensor = tf.constant(
        second_out_np, shape=(3, 3), dtype=tf.float32
    )
    surplus_np = np.array([[0.1] * 3] * 2, dtype=np.float32)
    surplus_tensor = tf.constant(surplus_np, shape=(3, 2), dtype=tf.float32)
    # Setup mock model.
    model_signature_mock = mock.MagicMock()
    model_signature_mock.return_value = {
        "output_a": output_tensor,
        "output_b": surplus_tensor,
        "output_c": second_out_tensor,
    }
    self._model.signatures = {"serving_default": model_signature_mock}

    result = self._runner.run_model_multiple_output(
        input_np, model_output_keys={"output_a", "output_c"}
    )

    self.assertLen(model_signature_mock.call_args_list, 1)
    self.assertTrue(
        tensor_equal(model_signature_mock.call_args[0][0], input_tensor),
        "Tensor passed to model values do not match input.",
    )
    # Missing a check for the shape of the input tensor.
    self.assertEqual(result.keys(), {"output_a", "output_c"})
    np.testing.assert_array_equal(result["output_a"], output_np)
    np.testing.assert_array_equal(result["output_c"], second_out_np)

  def test_batch_multi_output(self):
    # Setup test values.
    input_nps = [
        np.array([[0.5] * 3] * 3, dtype=np.float32),
        np.array([[0.25] * 3] * 3, dtype=np.float32),
    ]
    input_tensors = [
        tf.constant(input_np, shape=(3, 3), dtype=tf.float32)
        for input_np in input_nps
    ]
    output_np_a_1 = np.array([[0.5] * 2] * 3, dtype=np.float32)
    output_np_a_2 = np.array([[0.25] * 2] * 3, dtype=np.float32)
    output_np_b_1 = np.array([[0.6] * 2] * 3, dtype=np.float32)
    output_np_b_2 = np.array([[0.2] * 2] * 3, dtype=np.float32)
    output_np_c_1 = np.array([[0.7] * 2] * 3, dtype=np.float32)
    output_np_c_2 = np.array([[0.8] * 2] * 3, dtype=np.float32)
    output_maps = [
        {
            "output_a": tf.constant(
                output_np_a_1, shape=(3, 2), dtype=tf.float32
            ),
            "output_b": tf.constant(
                output_np_b_1, shape=(3, 2), dtype=tf.float32
            ),
            "output_c": tf.constant(
                output_np_c_1, shape=(3, 2), dtype=tf.float32
            ),
        },
        {
            "output_a": tf.constant(
                output_np_a_2, shape=(3, 2), dtype=tf.float32
            ),
            "output_b": tf.constant(
                output_np_b_2, shape=(3, 2), dtype=tf.float32
            ),
            "output_c": tf.constant(
                output_np_c_2, shape=(3, 2), dtype=tf.float32
            ),
        },
    ]
    # Setup mock model.
    model_signature_mock = mock.MagicMock()
    model_signature_mock.side_effect = output_maps
    self._model.signatures = {"serving_default": model_signature_mock}

    result = self._runner.batch_model_multiple_output(
        input_nps, model_output_keys={"output_a", "output_c"}
    )

    self.assertLen(model_signature_mock.call_args_list, len(input_nps))
    for input_tensor, call in zip(
        input_tensors, model_signature_mock.call_args_list
    ):
      self.assertTrue(
          tensor_equal(call[0][0], input_tensor),
          "Tensor passed to model values do not match input.",
      )
    # Missing a check for the shape of the input tensor.
    np.testing.assert_array_equal(result[0]["output_a"], output_np_a_1)
    np.testing.assert_array_equal(result[0]["output_c"], output_np_c_1)
    np.testing.assert_array_equal(result[1]["output_a"], output_np_a_2)
    np.testing.assert_array_equal(result[1]["output_c"], output_np_c_2)
    self.assertNotIn("output_b", result[0])
    self.assertNotIn("output_b", result[1])

  @parameterized.named_parameters(
      ("name_only", "alternate", None),
      ("name_and_version", "alternate", 1),
      ("version_only", "default", 1),
  )
  def test_not_implemented_multiversion(self, name: str, version: int | None):
    with self.assertRaises(NotImplementedError):
      self._runner.run_model(
          model_input=np.array([[0.5] * 3] * 3, dtype=np.float32),
          model_name=name,
          model_version=version,
      )


if __name__ == "__main__":
  absltest.main()
