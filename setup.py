# Copyright 2022 The Symanto Research Team Authors.
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
from typing import Dict

from setuptools import find_packages, setup

VERSION: Dict[str, str] = {}
with open("symanto_fsb/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open("requirements.txt", encoding="utf-8") as req_fp:
    install_requires = req_fp.readlines()

setup(
    version=VERSION["VERSION"],
    name="symanto-fsb",
    description="Symanto Few-Shot Learning Benchmark",
    author="Symanto Research GmbH & Co. KG",
    author_email="thomas.mueller@symanto.com",
    entry_points={"console_scripts": ["symanto-fsb=symanto_fsb.cli:app"]},
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    python_requires=">=3.6.0",
    zip_safe=False,
)
