from setuptools import setup, find_packages

with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
  name='torchnca',
  version='0.1.0',
  description='Neighbourhood Components Analysis in PyTorch.',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/kevinzakka/nca',
  author='Kevin Zakka',
  author_email='kevinarmandzakka@gmail.com',
  license='MIT',
  classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
  ],
  keywords='ai metric learning nearest neighbours dimensionality reduction',
  packages=find_packages(exclude=['examples']),
  install_requires=[
    'numpy>=1.0.0,<2.0.0',
    'torch>=1.0.0,<=1.4.0',
  ],
  python_requires='>=3.5',
)
