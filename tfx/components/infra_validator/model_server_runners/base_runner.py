# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Module for shared interface of every model server runners."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six
from typing import Text


class BaseModelServerRunner(six.with_metaclass(abc.ABCMeta, object)):
  """Shared interface of all model server runners."""

  @abc.abstractmethod
  def __repr__(self) -> Text:
    pass

  @abc.abstractmethod
  def GetEndpoint(self) -> Text:
    """Get an endpoint to the model server to connect to.

    Endpoint will be available after `WaitUntilRun()` is called.

    Raises:

    """
    pass

  @abc.abstractmethod
  def Start(self) -> None:
    """Start the model server in non-blocking manner."""
    pass

  @abc.abstractmethod
  def WaitUntilRunning(self, deadline: float) -> None:
    """Wait until model availability from model server is determined.

    Args:
      deadline: A deadline time in UTC timestamp (in seconds).
    Returns:
      Whether the model is available or not.
    """
    pass

  @abc.abstractmethod
  def Stop(self) -> None:
    """Stop the model server in blocking manner."""
    pass
