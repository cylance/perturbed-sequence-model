"""
distutils/setuptools install script.
"""
import re
from setuptools import find_packages, setup

VERSION_RE = re.compile(r"""([0-9dev.]+)""")
EDITABLE_INSTALL_RE = re.compile(r"""^-e\s+(.+)""")


def get_version():
    with open("VERSION", "r") as fh:
        init = fh.read().strip()
    return VERSION_RE.search(init).group(1)


def requirement_specification(requirements_line):
    """Extract the requirement specification from a line of a pip requirements.txt

  Usually the line is already a requirement specification compatible with
  setuptools and nothing needs to be done, but sometimes it'll be of the form
  '-e <path>' to tell pip to install a local copy of a dependency in editable
  mode for cross-package development; in this case, the last component of the
  path is passed to setuptools as the requirement (version can't be specified).
  """
    editable_requirement = EDITABLE_INSTALL_RE.match(requirements_line)
    if editable_requirement is None:
        return requirements_line.strip()
    else:
        path = editable_requirement.group(1)
        return path.split("/")[-1]


def get_requirements():
    with open("requirements.txt", "r") as f:
        lines = f.read().splitlines()
        return [requirement_specification(l) for l in lines]


setup(name='local_perturbations',
      version=get_version(),
      description='Local Perturbations Model for Personalized Sequence Learning',
      author='Michael T. Wojnowicz and Xuan Zhao',
      author_email='mwojnowicz@cylance.com, xzhao@cylance.com',
      license='Cylance Inc.',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      include_package_data=True,
      package_data={'local_perturbations': ['VERSION', 'requirements.txt']},
      install_requires=get_requirements(),
      )
