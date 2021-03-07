
from setuptools import setup


setup(name='sign_recognizer',
      description='Python side-project which recognize sign (French Sign Language) with mediapipe and tensorflow',
      author='Dahmani Selim',
      url="https://github.com/maveist/sign_recognizer",
      version='0.0.1',
      python_requires='>=3.*',
      install_requires=[req.replace("\n", "") for req in open('requirements.txt')]
      )
