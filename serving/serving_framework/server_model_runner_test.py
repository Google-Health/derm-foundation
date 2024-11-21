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
from serving_framework import server_model_runner
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


class ServerModelRunnerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self._stub = mock.create_autospec(
        prediction_service_pb2_grpc.PredictionServiceStub, instance=True
    )
    self._stub.Predict = mock.MagicMock()
    self._runner = server_model_runner.ServerModelRunner(stub=self._stub)

  def test_singleton_input(self):
    input_np = np.array([[0.5] * 3] * 3, dtype=np.float32)
    input_proto = predict_pb2.PredictRequest()
    input_proto.inputs["inputs"].CopyFrom(tf.make_tensor_proto(input_np))
    input_proto.model_spec.name = "default"
    output_np = np.ones((3, 2), dtype=np.float32)
    output_proto = predict_pb2.PredictResponse()
    output_proto.outputs["output_0"].CopyFrom(tf.make_tensor_proto(output_np))
    output_proto.model_spec.name = "default"
    self._stub.Predict.return_value = output_proto

    result = self._runner.run_model(input_np)

    self.assertLen(self._stub.Predict.call_args_list, 1)
    self.assertEqual(
        self._stub.Predict.call_args[0][0].SerializeToString(
            deterministic=True
        ),
        input_proto.SerializeToString(deterministic=True),
        "Proto passed to model does not match expectation.",
    )
    np.testing.assert_array_equal(result, output_np)

  def test_map_input(self):
    input_np_map = {
        "a": np.array([[0.5] * 3] * 3, dtype=np.float32),
        "b": np.array([[0.25] * 3] * 3, dtype=np.float32),
    }
    input_proto = predict_pb2.PredictRequest()
    for label in input_np_map:
      input_proto.inputs[label].CopyFrom(
          tf.make_tensor_proto(input_np_map[label])
      )
    input_proto.model_spec.name = "default"
    output_np = np.ones((3, 2), dtype=np.float32)
    output_proto = predict_pb2.PredictResponse()
    output_proto.outputs["output_0"].CopyFrom(tf.make_tensor_proto(output_np))
    output_proto.model_spec.name = "default"
    self._stub.Predict.return_value = output_proto

    result = self._runner.run_model(input_np_map)

    self.assertLen(self._stub.Predict.call_args_list, 1)
    self.assertEqual(
        self._stub.Predict.call_args[0][0].SerializeToString(
            deterministic=True
        ),
        input_proto.SerializeToString(deterministic=True),
        "Tensor passed to model values do not match input.",
    )
    np.testing.assert_array_equal(result, output_np)

  def test_batch_singleton_input(self):
    input_nps = [
        np.array([[0.5] * 3] * 3, dtype=np.float32),
        np.array([[0.25] * 3] * 3, dtype=np.float32),
    ]
    input_protos = []
    for input_np in input_nps:
      input_proto = predict_pb2.PredictRequest()
      input_proto.inputs["inputs"].CopyFrom(tf.make_tensor_proto(input_np))
      input_proto.model_spec.name = "default"
      input_protos.append(input_proto)
    output_nps = [
        np.ones((3, 2), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
    ]
    output_protos = []
    for output_np in output_nps:
      output_proto = predict_pb2.PredictResponse()
      output_proto.outputs["output_0"].CopyFrom(tf.make_tensor_proto(output_np))
      output_proto.model_spec.name = "default"
      output_protos.append(output_proto)
    self._stub.Predict.side_effect = output_protos

    result = self._runner.batch_model(input_nps)

    self.assertLen(self._stub.Predict.call_args_list, len(input_nps))
    for input_proto, call in zip(
        input_protos, self._stub.Predict.call_args_list
    ):
      self.assertEqual(
          call[0][0].SerializeToString(deterministic=True),
          input_proto.SerializeToString(deterministic=True),
          "Tensor passed to model values do not match input.",
      )
    for result_np, output_np in zip(result, output_nps):
      np.testing.assert_array_equal(result_np, output_np)

  def test_batch_map_input(self):
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
    input_protos = []
    for input_np_map in input_np_maps:
      input_proto = predict_pb2.PredictRequest()
      for label in input_np_map:
        input_proto.inputs[label].CopyFrom(
            tf.make_tensor_proto(input_np_map[label])
        )
      input_proto.model_spec.name = "default"
      input_protos.append(input_proto)
    output_nps = [
        np.ones((3, 2), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
    ]
    output_protos = []
    for output_np in output_nps:
      output_proto = predict_pb2.PredictResponse()
      output_proto.outputs["output_0"].CopyFrom(tf.make_tensor_proto(output_np))
      output_proto.model_spec.name = "default"
      output_protos.append(output_proto)
    self._stub.Predict.side_effect = output_protos

    result = self._runner.batch_model(input_np_maps)

    self.assertLen(self._stub.Predict.call_args_list, len(input_np_maps))
    for input_proto, call in zip(
        input_protos, self._stub.Predict.call_args_list
    ):
      self.assertEqual(
          call[0][0].SerializeToString(deterministic=True),
          input_proto.SerializeToString(deterministic=True),
          "Tensor passed to model values do not match input.",
      )
    for result_np, output_np in zip(result, output_nps):
      np.testing.assert_array_equal(result_np, output_np)

  def test_keyed_output(self):
    # Set up test values.
    input_np = np.array([[0.5] * 3] * 3, dtype=np.float32)
    input_proto = predict_pb2.PredictRequest()
    input_proto.inputs["inputs"].CopyFrom(tf.make_tensor_proto(input_np))
    input_proto.model_spec.name = "default"
    output_np = np.ones((3, 2), dtype=np.float32)
    surplus_np = np.array([[0.1] * 3] * 2, dtype=np.float32)
    output_proto = predict_pb2.PredictResponse()
    output_proto.outputs["output_a"].CopyFrom(tf.make_tensor_proto(output_np))
    output_proto.outputs["output_b"].CopyFrom(tf.make_tensor_proto(surplus_np))
    output_proto.model_spec.name = "default"
    self._stub.Predict.return_value = output_proto

    result = self._runner.run_model(input_np, model_output_key="output_a")

    self.assertLen(self._stub.Predict.call_args_list, 1)
    self.assertEqual(
        self._stub.Predict.call_args[0][0].SerializeToString(
            deterministic=True
        ),
        input_proto.SerializeToString(deterministic=True),
        "Proto passed to model does not match expectation.",
    )
    np.testing.assert_array_equal(result, output_np)

  def test_miskeyed_output(self):
    # Set up test values.
    input_np = np.array([[0.5] * 3] * 3, dtype=np.float32)
    input_proto = predict_pb2.PredictRequest()
    input_proto.inputs["inputs"].CopyFrom(tf.make_tensor_proto(input_np))
    input_proto.model_spec.name = "default"
    output_np = np.ones((3, 2), dtype=np.float32)
    surplus_np = np.array([[0.1] * 3] * 2, dtype=np.float32)
    output_proto = predict_pb2.PredictResponse()
    output_proto.outputs["output_a"].CopyFrom(tf.make_tensor_proto(output_np))
    output_proto.outputs["output_b"].CopyFrom(tf.make_tensor_proto(surplus_np))
    output_proto.model_spec.name = "default"
    self._stub.Predict.return_value = output_proto

    with self.assertRaises(KeyError):
      _ = self._runner.run_model(input_np, model_output_key="output_c")

  def test_multi_output(self):
    # Set up test values.
    input_np = np.array([[0.5] * 3] * 3, dtype=np.float32)
    input_proto = predict_pb2.PredictRequest()
    input_proto.inputs["inputs"].CopyFrom(tf.make_tensor_proto(input_np))
    input_proto.model_spec.name = "default"
    output_np = np.ones((3, 2), dtype=np.float32)
    second_out_np = np.array([[0.7] * 3] * 3, dtype=np.float32)
    surplus_np = np.array([[0.1] * 3] * 2, dtype=np.float32)
    output_proto = predict_pb2.PredictResponse()
    output_proto.outputs["output_a"].CopyFrom(tf.make_tensor_proto(output_np))
    output_proto.outputs["output_b"].CopyFrom(tf.make_tensor_proto(surplus_np))
    output_proto.outputs["output_c"].CopyFrom(
        tf.make_tensor_proto(second_out_np)
    )
    output_proto.model_spec.name = "default"
    self._stub.Predict.return_value = output_proto

    result = self._runner.run_model_multiple_output(
        input_np, model_output_keys={"output_a", "output_c"}
    )

    self.assertLen(self._stub.Predict.call_args_list, 1)
    self.assertEqual(
        self._stub.Predict.call_args[0][0].SerializeToString(
            deterministic=True
        ),
        input_proto.SerializeToString(deterministic=True),
        "Proto passed to model does not match expectation.",
    )
    self.assertEqual(result.keys(), {"output_a", "output_c"})
    np.testing.assert_array_equal(result["output_a"], output_np)
    np.testing.assert_array_equal(result["output_c"], second_out_np)

  def test_batch_multi_output(self):
    input_nps = [
        np.array([[0.5] * 3] * 3, dtype=np.float32),
        np.array([[0.25] * 3] * 3, dtype=np.float32),
    ]
    input_protos = []
    for input_np in input_nps:
      input_proto = predict_pb2.PredictRequest()
      input_proto.inputs["inputs"].CopyFrom(tf.make_tensor_proto(input_np))
      input_proto.model_spec.name = "default"
      input_protos.append(input_proto)
    output_nps_a = [
        np.ones((3, 2), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
    ]
    output_nps_b = [
        np.array([[0.6] * 2] * 3, dtype=np.float32),
        np.array([[0.2] * 2] * 3, dtype=np.float32),
    ]
    output_nps_c = [
        np.array([[0.7] * 2] * 3, dtype=np.float32),
        np.array([[0.8] * 2] * 3, dtype=np.float32),
    ]
    output_protos = []
    for output_np in zip(output_nps_a, output_nps_b, output_nps_c):
      output_proto = predict_pb2.PredictResponse()
      output_proto.outputs["output_a"].CopyFrom(
          tf.make_tensor_proto(output_np[0])
      )
      output_proto.outputs["output_b"].CopyFrom(
          tf.make_tensor_proto(output_np[1])
      )
      output_proto.outputs["output_c"].CopyFrom(
          tf.make_tensor_proto(output_np[2])
      )
      output_proto.model_spec.name = "default"
      output_protos.append(output_proto)
    self._stub.Predict.side_effect = output_protos

    result = self._runner.batch_model_multiple_output(
        input_nps, model_output_keys={"output_a", "output_c"}
    )

    self.assertLen(self._stub.Predict.call_args_list, len(input_nps))
    for input_proto, call in zip(
        input_protos, self._stub.Predict.call_args_list
    ):
      self.assertEqual(
          call[0][0].SerializeToString(deterministic=True),
          input_proto.SerializeToString(deterministic=True),
          "Tensor passed to model values do not match input.",
      )
    for result_map, *output_nps in zip(result, output_nps_a, output_nps_c):
      np.testing.assert_array_equal(result_map["output_a"], output_nps[0])
      np.testing.assert_array_equal(result_map["output_c"], output_nps[1])
      self.assertNotIn("output_b", result_map)

  def test_model_name_specification(self):
    input_np = np.array([[0.5] * 3] * 3, dtype=np.float32)
    input_proto = predict_pb2.PredictRequest()
    input_proto.inputs["inputs"].CopyFrom(tf.make_tensor_proto(input_np))
    input_proto.model_spec.name = "alternative"
    output_np = np.ones((3, 2), dtype=np.float32)
    output_proto = predict_pb2.PredictResponse()
    output_proto.outputs["output_0"].CopyFrom(tf.make_tensor_proto(output_np))
    output_proto.model_spec.name = "alternative"
    self._stub.Predict.return_value = output_proto

    _ = self._runner.run_model(input_np, model_name="alternative")

    self.assertLen(self._stub.Predict.call_args_list, 1)
    self.assertEqual(
        self._stub.Predict.call_args[0][0].SerializeToString(
            deterministic=True
        ),
        input_proto.SerializeToString(deterministic=True),
        "Proto passed to model does not match expectation.",
    )

  def test_model_version_specification(self):
    input_np = np.array([[0.5] * 3] * 3, dtype=np.float32)
    input_proto = predict_pb2.PredictRequest()
    input_proto.inputs["inputs"].CopyFrom(tf.make_tensor_proto(input_np))
    input_proto.model_spec.name = "default"
    input_proto.model_spec.version.value = 5
    output_np = np.ones((3, 2), dtype=np.float32)
    output_proto = predict_pb2.PredictResponse()
    output_proto.outputs["output_0"].CopyFrom(tf.make_tensor_proto(output_np))
    output_proto.model_spec.name = "alternative"
    self._stub.Predict.return_value = output_proto

    _ = self._runner.run_model(input_np, model_version=5)

    self.assertLen(self._stub.Predict.call_args_list, 1)
    self.assertEqual(
        self._stub.Predict.call_args[0][0].SerializeToString(
            deterministic=True
        ),
        input_proto.SerializeToString(deterministic=True),
        "Proto passed to model does not match expectation.",
    )


if __name__ == "__main__":
  absltest.main()
