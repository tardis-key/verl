# Copyright (c) 2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import stat


class Constant(object):
    # params
    DATA_MAP = "data_map"
    DATA_TYPE = "data_type"
    PROFILER_TYPE = "profiler_type"
    RANK_LIST = "rank_list"
    RANK_ID = "rank_id"
    PROFILER_DATA_PATH = "profiler_data_path"

    # dir name
    SINGLE_OUTPUT = "ASCEND_PROFILER_OUTPUT"

    # file suffix
    PT_PROF_SUFFIX = "ascend_pt"

    # result files type
    TEXT = "text"
    DB = "db"

    # Unit Conversion
    US_TO_MS = 1000
    NS_TO_US = 1000

    # profiler info
    PROFILER_INFO_HEAD = 'profiler_info_'
    PROFILER_INFO_EXTENSION = '.json'
    PROFILER_METADATA_JSON = 'profiler_metadata.json'

    # file authority
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP
    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC