#!/usr/bin/env python
# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

import sys
import importlib
import importlib.util
from pathlib import Path

package_name = 'foresight'

version_file = Path(__file__).parent.joinpath(package_name, 'version.py')
spec = importlib.util.spec_from_file_location('{}.version'.format(package_name), version_file)
package_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_version)
sys.modules[spec.name] = package_version

try:
    import torch
except ImportError:
    print('PyTorch not found! please install torch/torchvision before proceeding to install the foresight package.')
    exit(1)

class build_maybe_inplace(build_py):
    def run(self):
        global package_version
        package_version = importlib.reload(package_version)
        _dist_file = version_file.parent.joinpath('_dist_info.py')
        assert not _dist_file.exists()
        _dist_file.write_text('\n'.join(map(lambda attr_name: attr_name+' = '+repr(getattr(package_version, attr_name)), package_version.__all__)) + '\n')
        return super().run()


setup(name='foresight',
      version=package_version.version,
      description='Zero-cost proxies for NAS - predicting accuracy without training.',
      author='SAIC-Cambridge, On-Device Team, Automated ML Group',
      author_email='on.device@samsung.com',
      url='https://github.com/mohsaied/zero-cost-nas',
      download_url='https://github.com/mohsaied/zero-cost-nas',
      python_requires='>=3.6.0',
      setup_requires=[
          'git-python'
      ],
      install_requires=[
          'git-python',
          'h5py>=2.10.0',
          'jupyter>=1.0.0',
          'matplotlib>=3.2.1',
          'nas-bench-201==2.0',
          'numpy>=1.18.4',
          'prettytable>=2.0.0',
          'pytorch-ignite>=0.3.0',
          'pytorchcv>=0.0.58',
          'scikit-learn>=0.23.2',
          'scipy>=1.4.1',
          'tqdm>=4.46.0'
      ],
      packages=find_packages(where='.', include=[ 'foresight', 'foresight.*' ]),
      package_dir={ '': '.' },
      cmdclass={
          'build_py': build_maybe_inplace
      }
)
