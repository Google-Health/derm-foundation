#!/bin/bash
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

# This script launches the Derm foundation prediction container.
# It copies the model from Google Cloud Storage (GCS) to a local directory
# and launches the TensorFlow Model Server and the Derm foundation server.
# The script waits for any process to exit and then exits with the status of
# the process that exited first.
set -e

# Port for the TensorFlow Model Server REST API.
export MODEL_REST_PORT=8600

# Local directory to store the model.
export LOCAL_MODEL_PATH=/model/default

echo "Prediction container start, launching model server"

# Copy model from Google Cloud Storage (GCS) to local directory
mkdir -p "$LOCAL_MODEL_PATH/1"

# TODO(b/379159076): Remove gcloud
gcloud storage cp "$AIP_STORAGE_URI/*" "$LOCAL_MODEL_PATH/1" --recursive

/usr/bin/tensorflow_model_server \
    --port=8500 \
    --rest_api_port="$MODEL_REST_PORT" \
    --model_name=default \
    --model_base_path="$LOCAL_MODEL_PATH" \
    --xla_cpu_compilation_enabled=true &

echo "Launching front end"

/server-env/bin/python3.12 -m serving.server_gunicorn --alsologtostderr \
    --verbosity=1 &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
