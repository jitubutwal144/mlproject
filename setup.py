from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(filename: str) -> List[str]:
    # this function will return list of requirements
    requirements = []
    with open(filename) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

setup(
  name='mlproject',
  version='0.1',
  author='Jitendra Kasaudhan',
  author_email='jiten.ktm+git@gmail.com',
  packages=find_packages(),
  license='MIT',
  install_requires=get_requirements('requirements.txt')
)
