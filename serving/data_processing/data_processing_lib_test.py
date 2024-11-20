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

"""Tests for data processing library."""

import io
import os
from unittest import mock

from google.cloud import storage
import numpy as np
import png
import pydicom

from absl.testing import absltest
from absl.testing import parameterized
from data_processing import data_processing_lib
from data_processing import dicom_client


_TEST_DICOM_PATH = 'serving/data_processing/testdata/fake.dcm'
_TEST_PNG_PATH = 'serving/data_processing/testdata/fake.png'


def _get_test_png_bytes():
  with open(_TEST_PNG_PATH, 'rb') as f:
    return f.read()


def _get_test_dicom_bytes():
  with open(_TEST_DICOM_PATH, 'rb') as f:
    return f.read()


def _get_test_pydicom():
  return pydicom.dcmread(io.BytesIO(_get_test_dicom_bytes()))


class MockBlob:

  @classmethod
  def from_string(cls, uri, client):
    del uri, client
    return cls()

  def download_as_bytes(self):
    return b'some_bytes'


class MockBlobWithDicom:

  @classmethod
  def from_string(cls, uri, client):
    del uri, client
    return cls()

  def download_as_bytes(self):
    return _get_test_dicom_bytes()


class MockDicomWebClient:

  def __init__(self, project, dicom_store, input_credentials):
    pass

  def get_pydicom(self, study_uid, series_uid, instance_uid):
    del study_uid, series_uid, instance_uid
    return pydicom.Dataset()


class DataProcessingLibTest(parameterized.TestCase):

  @mock.patch.object(storage.blob, 'Blob', MockBlob)
  @mock.patch.object(storage, 'Client', autospec=True)
  def test_retrieve_file_bytes_from_gcs_succeeds(
      self, unused_mock_storage_client
  ):
    retrieved_data = data_processing_lib.retrieve_file_bytes_from_gcs(
        'gs://bucket/file.png', None  # No credentials
    )
    self.assertEqual(retrieved_data, b'some_bytes')

  @mock.patch.object(storage.blob, 'Blob', MockBlobWithDicom)
  @mock.patch.object(storage, 'Client', autospec=True)
  def test_retrieve_dicom_from_gcs_succeeds(self, unused_mock_storage_client):
    retrieved_data = data_processing_lib.retrieve_dicom_from_gcs(
        'gs://bucket/file.dcm', None  # No credentials
    )
    self.assertEqual(retrieved_data, _get_test_pydicom())

  @mock.patch.object(dicom_client, 'DicomWebStatefulClient', MockDicomWebClient)
  @mock.patch.object(
      pydicom, 'dcmread', autospec=True, return_value=pydicom.Dataset()
  )
  def test_retrieve_instance_from_dicom_store_succeeds(
      self,
      unused_mock_dcmread,
  ):
    dicomweb_uri = (
        'https://healthcare.googleapis.com/v1/'
        'projects/a/locations/b/datasets/c/dicomStores/d/dicomWeb/'
        'studies/1/series/2/instances/3'
    )
    retrieved_data = data_processing_lib.retrieve_instance_from_dicom_store(
        dicomweb_uri, None  # no credentials
    )
    self.assertEqual(retrieved_data, _get_test_pydicom())

  @parameterized.named_parameters(
      dict(
          testcase_name='image_bytes',
          image=_get_test_png_bytes(),
      ),
      dict(
          testcase_name='dicom',
          image=_get_test_pydicom(),
      ),
  )
  def test_process_image_to_tf_example_succeeds(
      self,
      image,
  ):
    example = data_processing_lib.process_xray_image_to_tf_example(image)
    f_dict = example.features.feature
    if 'image/format' in f_dict:
      self.assertEqual(f_dict['image/format'].bytes_list.value[:], [b'png'])
    _, _, raw_image, meta = png.Reader(
        io.BytesIO(f_dict['image/encoded'].bytes_list.value[:][0])
    ).asDirect()
    image_2d = np.vstack(list(map(np.uint16, raw_image)))
    self.assertEqual(
        meta,
        {
            'alpha': False,
            'bitdepth': 16,
            'greyscale': True,
            'interlace': 0,
            'planes': 1,
            'size': (1024, 1024),
        },
    )
    self.assertAlmostEqual(np.average(image_2d), 75.4108, places=3)


if __name__ == '__main__':
  absltest.main()
