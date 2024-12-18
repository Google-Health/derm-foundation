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
#
# This is used to build a Docker image that includes the necessary dependencies
# for running the Derm Foundation as a microservice.

FROM tensorflow/serving:2.18.0

COPY ./python/data_processing /data_processing
COPY ./python/serving /serving
COPY ./LICENSE /LICENSE
RUN chmod a+x /serving/entrypoint.sh

# Install python3.12 and git
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y git python3.12 python3.12-venv

# Install dependencies
RUN python3.12 -m venv /server-env && \
    /server-env/bin/python3.12 -m pip install --upgrade pip && \
    /server-env/bin/python3.12 -m pip install -r /serving/requirements.txt

# TODO(b/379159076) - Remove gcloud
# Install gcloud CLI
RUN apt-get install -y apt-transport-https curl gnupg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get install google-cloud-cli -y

# Clone python-certifi to meet MPL 2.0 License source code mirroring terms
RUN git clone https://github.com/certifi/python-certifi.git

# Clone python-frozendict to meet LGPL 3.0 License source code mirroring terms
RUN git clone https://github.com/Marco-Sulla/python-frozendict.git

ENTRYPOINT ["/serving/entrypoint.sh"]
