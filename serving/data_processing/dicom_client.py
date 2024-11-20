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

"""Stateful DICOMWeb client with credential management for TF example creation."""

import io
from typing import Any, Dict, Iterator, List, Optional, Tuple

from google.auth import credentials
import pydicom

# TODO(b/373943872): Use EZ-WSI based modules.
from hcls_imaging_ml_toolkit import dicom_path
from hcls_imaging_ml_toolkit import dicom_web


_OFFSET_LIMIT_TEMPLATE = '?offset={offset}&limit={limit}'
_API_QUERY_PAGE_SIZE = 200
_STUDY_UID_TAG = '0020000D'  # (0008, 0018)
_SERIES_UID_TAG = '0020000E'  # (0020, 000E)
_SOP_INSTANCE_UID_TAG = '00080018'  # (0008, 0018)


class DicomWebStatefulClient:
  """Stateful DICOMWeb client for tf example creation."""

  def __init__(
      self,
      project: str,
      dicom_store: str,
      input_credentials: Optional[credentials.Credentials] = None,
  ):
    """Initialized the class with the given store and credentials.

    Args:
      project: The ID of the project where the DICOM store is located.
      dicom_store: The DICOM store to point to for operations. It is in the
        format of location/dataset/store.
      input_credentials: [Optional] Input credentials for access. If none are
        provided then Application Default Credentials are used.
    """
    dicom_store_components = dicom_store.split('/')
    if len(dicom_store_components) != 3:
      raise ValueError(
          'Illegal target_dicom_store parameter: {}'.format(dicom_store)
      )
    location, dataset, store = dicom_store_components
    self._base_path = dicom_path.Path(project, location, dataset, store)
    self._client = dicom_web.DicomWebClientImpl(input_credentials)

  def _construct_url(
      self,
      study_uid: Optional[str] = None,
      series_uid: Optional[str] = None,
      instance_uid: Optional[str] = None,
  ) -> str:
    """Construct request URL for various API operations."""
    url = dicom_web.PathToUrl(
        dicom_path.FromPath(
            self._base_path,
            study_uid=study_uid,
            series_uid=series_uid,
            instance_uid=instance_uid,
        )
    )
    # The URL returned from dicom_web.PathToUrl() will be one of the following:
    # https://.../dicomStores/<dicom_store_id>/dicomWeb/studies/<study_uid>/series/<series_uid>/instances/<instance_uid>
    # https://.../dicomStores/<dicom_store_id>/dicomWeb/studies/<study_uid>/series/<series_uid>
    # https://.../dicomStores/<dicom_store_id>/dicomWeb/studies/<study_uid>
    # https://.../dicomStores/<dicom_store_id>
    # depending on whether study_uid, series_uid, or instance_uid are given.
    if study_uid and series_uid and instance_uid:
      return url
    if study_uid and series_uid:
      return url + '/instances'
    if study_uid:
      return url + '/series'
    return url + '/dicomWeb/studies'

  def query_dicoms(
      self, study_uid: Optional[str] = None, series_uid: Optional[str] = None
  ) -> Iterator[Dict[str, Any]]:
    """Yields raw DICOM tags from set DICOM store and any given UIDs.

    Args:
        study_uid: [Optional] The Study UID of the DICOMs to query.
        series_uid: [Optional] The Series UID of the DICOMs to query. If this is
          specified, a study_uid MUST be given.

    Yields:
        A dictionary with data associated with dicom(s) dependant on the depth
        of the query.

    Raises:
      ValueError: If a series_uid is given without a study_uid.
    """
    offset = 0
    limit = _API_QUERY_PAGE_SIZE
    done = False

    if series_uid and not study_uid:
      raise ValueError('Series UID given without Study UID!')

    while not done:
      qido_url = self._construct_url(study_uid, series_uid)
      qido_url += _OFFSET_LIMIT_TEMPLATE.format(limit=limit, offset=offset)
      parsed_content = self._client.QidoRs(qido_url)
      for a_dict in parsed_content:
        yield a_dict
      done = len(parsed_content) < limit
      offset += limit

  def _query_dicoms_uid(
      self, study_uid: Optional[str] = None, series_uid: Optional[str] = None
  ) -> Iterator[str]:
    """Yield study or series UID values for set DICOM store.

    Args:
      study_uid: study UID of the DICOMs of interest.
      series_uid: series UID of the DICOMs of interest.

    Yields:
      Study UID values (if no study series or study UID given) OR
      Series UID values (if only study UID given) OR
      Instance UID value if both study and series UID given.
    """
    parsed_content = self.query_dicoms(study_uid, series_uid)
    if series_uid:
      for a_dict in parsed_content:
        if _SOP_INSTANCE_UID_TAG in a_dict:
          yield a_dict[_SOP_INSTANCE_UID_TAG]['Value'][0]
    elif study_uid:
      for a_dict in parsed_content:
        if _SERIES_UID_TAG in a_dict:
          yield a_dict[_SERIES_UID_TAG]['Value'][0]
    else:
      for a_dict in parsed_content:
        if _STUDY_UID_TAG in a_dict:
          yield a_dict[_STUDY_UID_TAG]['Value'][0]

  def get_study_uids(self) -> Iterator[str]:
    return self._query_dicoms_uid()

  def get_series_uids(self, study_uid: str) -> Iterator[str]:
    return self._query_dicoms_uid(study_uid)

  def get_instance_uids(self, study_uid: str, series_uid: str) -> Iterator[str]:
    return self._query_dicoms_uid(study_uid, series_uid)

  def get_series_and_instance_uids(
      self, study_uid: str
  ) -> List[Tuple[str, str]]:
    series_instances = []
    series_uids = list(self.get_series_uids(study_uid))
    for series_uid in series_uids:
      instance_uids = list(self.get_instance_uids(study_uid, series_uid))
      for instance_uid in instance_uids:
        series_instances.append((series_uid, instance_uid))
    return series_instances

  def get_pydicom(
      self,
      study_uid: Optional[str],
      series_uid: Optional[str],
      instance_uid: Optional[str],
  ) -> pydicom.dataset.FileDataset:
    """Download the given DICOM to a pydicom."""
    download_url = self._construct_url(study_uid, series_uid, instance_uid)
    dicom_byte_data = self._client.WadoRs(download_url)
    return pydicom.dcmread(
        io.BytesIO(dicom_byte_data), stop_before_pixels=False
    )

  def get_series_pydicoms(
      self, study_uid: str, series_uid: str
  ) -> List[pydicom.dataset.FileDataset]:
    """Download all DICOMs for a given series within a study."""
    instances = self.get_instance_uids(study_uid, series_uid)
    return [self.get_pydicom(study_uid, series_uid, inst) for inst in instances]

  def get_study_pydicoms(
      self, study_uid: str
  ) -> List[pydicom.dataset.FileDataset]:
    """Download all DICOMs for a given study."""
    series_instances = self.get_series_and_instance_uids(study_uid)
    return [self.get_pydicom(study_uid, *sinst) for sinst in series_instances]
