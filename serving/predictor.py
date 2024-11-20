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

"""Derm foundation model predictor.

Prepares model input, calls the model, and post-processes the output into the
final response.
"""

import base64
from typing import Any

from absl import logging
from google.oauth2 import credentials
import numpy as np

from data_processing import data_processing_lib
from prediction_container import model_runner


_INPUT_BYTES_KEY = 'input_bytes'
_GCS_KEY = 'gcs_uri'
_BEARER_TOKEN_KEY = 'bearer_token'


# TODO(b/372747494): Improve error handling and client-facing messaging.
class _PredictorError(Exception):
  """Exception for known predictor errors."""

  def __init__(self, client_message: str):
    super().__init__()
    self.client_message = client_message


class Predictor:
  """A predictor for getting embeddings from the Derm Foundation model."""

  def _get_image_bytes(self, instance: dict[str, Any]) -> bytes:
    """Gets the image bytes from a single instance."""
    if _INPUT_BYTES_KEY in instance and _GCS_KEY in instance:
      raise _PredictorError(
          'Request has more than one image input. Must specify either'
          ' `input_bytes` or `gcs_uri`.'
      )

    if _INPUT_BYTES_KEY in instance:
      return base64.b64decode(instance[_INPUT_BYTES_KEY])

    if _GCS_KEY not in instance:
      raise _PredictorError(
          'Missing required `input_bytes` or `gcs_uri` key in request instance.'
      )

    creds = (
        credentials.Credentials(token=instance[_BEARER_TOKEN_KEY])
        if _BEARER_TOKEN_KEY in instance
        else None
    )
    gcs_uri = instance[_GCS_KEY]
    logging.info('Retrieving file bytes from GCS: %s', gcs_uri)
    return data_processing_lib.retrieve_file_bytes_from_gcs(gcs_uri, creds)

  def _get_model_input(self, instance: dict[str, Any]) -> np.ndarray:
    """Gets the model input for a single instance."""
    try:
      image_bytes = self._get_image_bytes(instance)
    except _PredictorError as e:
      raise e
    except Exception as e:
      raise _PredictorError(
          'Failed to retrieve data from request instance.'
      ) from e
    logging.info('Retrieved image bytes.')
    try:
      example = data_processing_lib.process_image_bytes_to_tf_example(
          image_bytes
      )
    except Exception as e:
      raise _PredictorError('Failed to process image to TF example.') from e
    logging.info('Processed image to TF example.')
    return np.array([example.SerializeToString()])

  def _prepare_response(self, predictions: np.ndarray) -> dict[str, Any]:
    """Prepares the response json for the client."""
    return {'embedding': predictions.tolist()}

  def predict(
      self,
      request: dict[str, Any],
      model: model_runner.ModelRunner,
  ) -> dict[str, Any]:
    """Runs model inference on the request instances.

    Args:
      request: The parsed request json to process.
      model: The model runner object to use to call the model.

    Returns:
      The response json which will be returned to the client through the
      Vertex endpoint API.
    """
    predictions: list[dict[str, Any]] = []
    for instance in request['instances']:
      try:
        model_input = self._get_model_input(instance)
        embedding = model.run_model(
            model_input=model_input, model_output_key='embedding'
        )
        logging.info('Ran inference on model.')
      except _PredictorError as e:
        logging.exception('Failed to get prediction for instance.')
        response = {
            'error': {
                'description': (
                    'Failed to get prediction for instance. Reason:'
                    f' {e.client_message}'
                )
            }
        }
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Catch-all for any other exceptions that haven't been caught and
        # converted to _PredictorError.
        logging.exception('Failed to get prediction for instance: %s', e)
        response = {
            'error': {
                'description': 'Internal error getting prediction for instance.'
            }
        }
      else:
        response = self._prepare_response(embedding)
        logging.info('Prepared response.')
      predictions.append(response)
    return {'predictions': predictions}
