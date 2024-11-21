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

"""Executable to carry out custom handling of the prediction request.

A subprocess which handles a piped-in request json and returns the response
json body to stdout. Depends on a local TensorFlow Serving instance to serve
the model.
"""

from collections.abc import Sequence
import json
import sys
from typing import Any

from absl import app
from absl import logging

from serving_framework import server_model_runner
import predictor


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  model_runner = server_model_runner.ServerModelRunner()
  predictor_instance = predictor.Predictor()

  logging.info('Starting prediction executor loop.')
  while True:
    logging.debug('Waiting for request.')
    try:
      request_str = sys.stdin.readline()
      request: dict[str, Any] = json.loads(request_str)
    except EOFError:
      logging.warning('EOF on input, exiting.')
      exit(1)
    except json.JSONDecodeError:
      logging.exception('Failed to parse request JSON, exiting.')
      exit(1)
    logging.info('Received request.')
    result_json = predictor_instance.predict(request, model_runner)
    logging.debug('Returning result from executor.')
    try:
      json.dump(result_json, sys.stdout)
      sys.stdout.write('\n')
      sys.stdout.flush()
    except BrokenPipeError:
      logging.warning('Output pipe broken, exiting.')
      exit(1)
    logging.info('Finished handling request.')


if __name__ == '__main__':
  app.run(main)
