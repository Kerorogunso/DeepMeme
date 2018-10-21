import os
from setuptools import setup, find_packages


version = "0.1.0"

requirements_file = 'requirements.txt'
with open(requirements_file) as f:
  install_requires = f.read().splitlines()

setup(name="deepmeme",
      packages=find_packages(),
      version=version,
      description="Package for accessing the clustering tools",
      classifiers=[
          'Development Status :: Beta',
          'Environment :: Web/Desktop Environment',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
      ],  
      author='Albert Chung',
      author_email='albertswchung@gmail.com',
      license='',
      python_requires='>=3',
      install_requires=install_requires,
)