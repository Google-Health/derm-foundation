#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Derm foundation model predictor."""

from unittest import mock

import numpy as np
import tensorflow as tf

from absl.testing import absltest
from data_processing import data_processing_lib
from serving.serving_framework import model_runner
from health_foundations.derm_foundation.serving import predictor


@mock.patch.object(model_runner, "ModelRunner", autospec=True)
class PredictorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._request_instance = {
        "instances": [{
            "gcs_uri": "gs://bucket/file.png",
            "bearer_token": "my_token",
        }]
    }

  @mock.patch.object(
      data_processing_lib,
      "process_image_bytes_to_tf_example",
      autospec=True,
      return_value=tf.train.Example(),
  )
  @mock.patch.object(
      data_processing_lib,
      "retrieve_file_bytes_from_gcs",
      autospec=True,
      return_value=b"asdf",
  )
  def test_predict_model_called_with_correct_input(
      self, unused_mock_retrieve, unused_mock_process, mock_model_runner
  ):
    predictor.Predictor().predict(
        request=self._request_instance,
        model=mock_model_runner,
    )
    mock_model_runner.run_model.assert_called_once_with(
        model_input=np.array([tf.train.Example().SerializeToString()]),
        model_output_key="embedding",
    )

  @mock.patch.object(predictor.Predictor, "_get_model_input", autospec=True)
  def test_predict_returns_correct_response(
      self, unused_mock_model_input, mock_model_runner
  ):
    mock_model_runner.run_model.return_value = np.array([[[1, 2, 3]]])
    response = predictor.Predictor().predict(
        request=self._request_instance,
        model=mock_model_runner,
    )
    self.assertEqual(response, {"predictions": [{"embedding": [[1, 2, 3]]}]})

  @mock.patch.object(
      data_processing_lib,
      "retrieve_file_bytes_from_gcs",
      autospec=True,
      side_effect=ValueError("some retrieval error"),
  )
  def test_predict_retrieve_image_data_failure_returns_error_response(
      self, unused_retrieve, mock_model_runner
  ):
    response = predictor.Predictor().predict(
        request=self._request_instance,
        model=mock_model_runner,
    )
    self.assertEqual(
        response["predictions"][0]["error"]["description"],
        "Failed to get prediction for instance. Reason: Failed to retrieve data"
        " from request instance.",
    )

  @mock.patch.object(
      data_processing_lib,
      "process_image_bytes_to_tf_example",
      autospec=True,
      side_effect=ValueError("some processing error"),
  )
  @mock.patch.object(
      data_processing_lib,
      "retrieve_file_bytes_from_gcs",
      autospec=True,
      return_value=b"asdf",
  )
  def test_predict_process_image_to_tf_example_failure_returns_error_response(
      self, unused_mock_retrieve, unused_mock_process, mock_model_runner
  ):
    response = predictor.Predictor().predict(
        request=self._request_instance,
        model=mock_model_runner,
    )
    self.assertEqual(
        response["predictions"][0]["error"]["description"],
        "Failed to get prediction for instance. Reason: Failed to process image"
        " to TF example.",
    )

  def test_predict_without_image_input_returns_error_response(
      self, mock_model_runner
  ):
    response = predictor.Predictor().predict(
        request={"instances": [{}]},
        model=mock_model_runner,
    )
    self.assertEqual(
        response["predictions"][0]["error"]["description"],
        "Failed to get prediction for instance. Reason: Missing required"
        " `input_bytes` or `gcs_uri` key in request instance.",
    )

  def test_predict_with_multiple_image_inputs_returns_error_response(
      self, mock_model_runner
  ):
    response = predictor.Predictor().predict(
        request={
            "instances": [{
                "gcs_uri": "gs://bucket/file.dcm",
                "input_bytes": "c29tZV9ieXRlcw==",
            }]
        },
        model=mock_model_runner,
    )
    self.assertEqual(
        response["predictions"][0]["error"]["description"],
        "Failed to get prediction for instance. Reason: Request has more than"
        " one image input. Must specify either `input_bytes` or `gcs_uri`.",
    )

  @mock.patch.object(
      data_processing_lib, "process_image_bytes_to_tf_example", autospec=True
  )
  @mock.patch.object(
      data_processing_lib,
      "retrieve_file_bytes_from_gcs",
      autospec=True,
      side_effect=[KeyError("some retrieval error"), b"asdf"],
  )
  def test_predict_with_multiple_request_instances_returns_correct_response(
      self, unused_mock_retrieve, unused_mock_process, mock_model_runner
  ):
    mock_model_runner.run_model.return_value = np.array([[[1, 2, 3]]])
    response = predictor.Predictor().predict(
        request={
            "instances": [
                {
                    "gcs_uri": "gs://bucket/file1.png",
                    "bearer_token": "my_token",
                },
                {
                    "gcs_uri": "gs://bucket/file2.png",
                    "bearer_token": "my_token",
                },
            ]
        },
        model=mock_model_runner,
    )
    self.assertEqual(
        response["predictions"][0]["error"]["description"],
        "Failed to get prediction for instance. Reason: Failed to retrieve data"
        " from request instance.",
    )
    self.assertEqual(response["predictions"][1]["embedding"], [[1, 2, 3]])

  @mock.patch.object(
      data_processing_lib,
      "process_image_bytes_to_tf_example",
      autospec=True,
      return_value=tf.train.Example(),
  )
  def test_predict_with_input_bytes_process_called_with_correct_input(
      self, mock_process, mock_model_runner
  ):
    mock_model_runner.run_model.return_value = np.array([[[1, 2, 3]]])
    predictor.Predictor().predict(
        request={
            "instances": [
                {
                    "input_bytes": "c29tZV9ieXRlcw==",
                },
            ]
        },
        model=mock_model_runner,
    )
    mock_process.assert_called_once_with(b"some_bytes")


if __name__ == "__main__":
  absltest.main()
