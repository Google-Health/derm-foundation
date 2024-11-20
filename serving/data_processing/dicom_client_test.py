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

"""Tests for dicom_client."""

from unittest import mock

from google.auth import credentials
import pydicom

from absl.testing import absltest
from hcls_imaging_ml_toolkit import dicom_web
from data_processing import dicom_client

_PYDICOM_MAJOR_VERSION = int((pydicom.__version__).split('.')[0])


class TestBearerCredentials(credentials.Credentials):

  def __init__(self, bearer_token):
    super().__init__()
    self.token = bearer_token

  def refresh(self, request):
    pass


class DicomWebStatefulClientTest(absltest.TestCase):

  @mock.patch.object(
      dicom_web.DicomWebClientImpl,
      'QidoRs',
      return_value=[{
          '0020000E': {
              'Value': ['1.2.840.113654'],
              'vr': 'UI'
          }
      }],
      autospec=True)
  def test_dicom_store(self, _):
    # Shorten the page size to test pagination parameters.
    dicom_client._API_QUERY_PAGE_SIZE = 5
    dcs = dicom_client.DicomWebStatefulClient('TEST', 'UP/LOC/DS',
                                              TestBearerCredentials('test'))
    return_uids_single = list(dcs.get_series_uids('study1'))
    self.assertCountEqual(return_uids_single, ['1.2.840.113654'])

  def test_get_series_dicoms(self):
    dcs = dicom_client.DicomWebStatefulClient('TEST', 'UP/LOC/DS',
                                              TestBearerCredentials('test'))
    with mock.patch.object(
        dcs, 'get_instance_uids', autospec=True) as mock_uid_call:
      with mock.patch.object(
          dcs, 'get_pydicom', autospec=True) as mock_dicom_call:
        mock_uid_call.return_value = ['3.3.3', '3.3.4']
        mock_dicom_call.return_value = _make_pydicom()
        dicoms = dcs.get_series_pydicoms('1.1.1', '2.2.2')
        mock_uid_call.assert_called_once_with('1.1.1', '2.2.2')
        self.assertLen(dicoms, 2)
        self.assertEqual(mock_dicom_call.call_count, 2)
        self.assertEqual(mock_dicom_call.call_args_list[0],
                         mock.call('1.1.1', '2.2.2', '3.3.3'))

  def test_get_study_dicoms(self):
    dcs = dicom_client.DicomWebStatefulClient('TEST', 'UP/LOC/DS',
                                              TestBearerCredentials('test'))
    with mock.patch.object(
        dcs, 'get_series_and_instance_uids', autospec=True) as mock_uid_call:
      with mock.patch.object(
          dcs, 'get_pydicom', autospec=True) as mock_dicom_call:
        mock_uid_call.return_value = [('2.2', '2.3'), ('3.3', '3.4')]
        mock_dicom_call.return_value = _make_pydicom()
        dicoms = dcs.get_study_pydicoms('1.1')
        mock_uid_call.assert_called_once_with('1.1')
        self.assertLen(dicoms, 2)
        self.assertEqual(mock_dicom_call.call_count, 2)
        self.assertEqual(mock_dicom_call.call_args_list[0],
                         mock.call('1.1', '2.2', '2.3'))
        self.assertEqual(mock_dicom_call.call_args_list[1],
                         mock.call('1.1', '3.3', '3.4'))


def _make_pydicom():
  ds = pydicom.Dataset()
  ds.StudyInstanceUID = '1.1'
  ds.SeriesInstanceUID = '2.2'
  ds.SOPInstanceUID = '3.3'
  if _PYDICOM_MAJOR_VERSION <= 2:
    ds.is_implicit_VR = False
  ds.file_meta = pydicom.dataset.FileMetaDataset()
  return ds


if __name__ == '__main__':
  absltest.main()
