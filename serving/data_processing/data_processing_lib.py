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

"""Methods to retrieve and process data into model inputs for prediction."""

import io

from absl import logging
from google.cloud import storage
from google.oauth2 import credentials
import numpy as np
from PIL import Image
import pydicom
import tensorflow as tf

# TODO(b/373943872): Use EZ-WSI based module.
from hcls_imaging_ml_toolkit import dicom_path
from data_processing import dicom_client
from data_processing import image_utils

_DICOMWEB_URI_PREFIX = 'https://healthcare.googleapis.com/v1/'

_IMAGE_KEY = 'image/encoded'
_IMAGE_FORMAT = 'image/format'


def retrieve_file_bytes_from_gcs(
    gcs_uri: str, creds: credentials.Credentials | None
) -> bytes:
  """Retrieves bytes from a GCS file.

  Args:
    gcs_uri: Location of the file in GCS, in the format
      `gs://bucket-name/path/to/file`.
    creds: (Optional) Credentials to use to access the data. If none are
      provided, the Application Default Credentials will be used.

  Returns:
    The bytes stored in the file.
  """
  storage_client = storage.Client(credentials=creds)
  blob = storage.blob.Blob.from_string(gcs_uri, client=storage_client)
  return blob.download_as_bytes()


def retrieve_dicom_from_gcs(
    gcs_uri: str, creds: credentials.Credentials | None
) -> pydicom.Dataset:
  """Retrieves bytes from a GCS file, and converts it to a pydicom.Dataset.

  Args:
    gcs_uri: Location of the file in GCS, in the format
      `gs://bucket-name/path/to/file`.
    creds: (Optional) Credentials to use to access the data. If none are
      provided, the Application Default Credentials will be used.

  Returns:
    The file, converted to a pydicom.Dataset.
  """
  return pydicom.dcmread(
      io.BytesIO(retrieve_file_bytes_from_gcs(gcs_uri, creds))
  )


def retrieve_instance_from_dicom_store(
    dicomweb_uri: str, creds: credentials.Credentials | None
) -> pydicom.Dataset:
  """Retrieves an instance from DICOM store.

  Args:
    dicomweb_uri: DICOMweb path specifying a DICOM instance.
    creds: (Optional) Credentials to use to access the data. If none are
      provided, the Application Default Credentials will be used.

  Returns:
    A pydicom.Dataset containing the DICOM instance data.
  """
  dicom_path_str = dicomweb_uri[len(_DICOMWEB_URI_PREFIX) :]
  path = dicom_path.FromString(dicom_path_str, dicom_path.Type.INSTANCE)
  dicom_store = f'{path.location}/{path.dataset_id}/{path.store_id}'
  dicomweb_client = dicom_client.DicomWebStatefulClient(
      path.project_id, dicom_store, input_credentials=creds
  )
  return dicomweb_client.get_pydicom(
      path.study_uid, path.series_uid, path.instance_uid
  )


def process_image_bytes_to_tf_example(image_bytes: bytes) -> tf.train.Example:
  """Creates a tf.train.Example from encoded image bytes."""
  image_feature = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[image_bytes])
  )
  return tf.train.Example(
      features=tf.train.Features(feature={_IMAGE_KEY: image_feature})
  )


def _process_xray_image_array_to_tf_example(
    image_array: np.ndarray,
) -> tf.train.Example:
  """Creates a tf.train.Example from an X-ray image array."""
  pixel_array = image_utils.shift_to_unsigned(image_array)
  pixel_array = image_utils.rescale_dynamic_range(pixel_array)
  png_bytes = image_utils.encode_png(pixel_array.astype(np.uint16))
  example = process_image_bytes_to_tf_example(png_bytes)
  example.features.feature[_IMAGE_FORMAT].bytes_list.value[:] = [b'png']
  return example


def _apply_pydicom_prep(ds: pydicom.Dataset) -> np.ndarray:
  """Applies data handling from pydicom."""
  arr = ds.pixel_array
  pixel_array = pydicom.pixels.processing.apply_modality_lut(arr, ds)
  if 'WindowWidth' in ds and 'WindowCenter' in ds:
    pixel_array = image_utils.window(
        pixel_array, ds.WindowCenter, ds.WindowWidth
    )
  if ds.PhotometricInterpretation == 'MONOCHROME1':
    pixel_array = np.max(pixel_array) - pixel_array
  return pixel_array


def process_xray_image_to_tf_example(
    image: bytes | pydicom.Dataset,
) -> tf.train.Example:
  """Processes X-ray image data into a tf.train.Example.

  Args:
    image: The image data to process, either as image bytes or a
      pydicom.Dataset.

  Returns:
    A tf.train.Example containing the processed image.
  """
  if isinstance(image, pydicom.Dataset):
    logging.info('Processing DICOM instance to tf.train.Example.')
    image_array = _apply_pydicom_prep(image)
  else:
    logging.info('Processing image bytes to tf.train.Example.')
    image_array = np.asarray(Image.open(io.BytesIO(image)).convert('L'))
  return _process_xray_image_array_to_tf_example(image_array)
