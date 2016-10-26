from distutils.core import setup
import numpy 

DESCRIPTION = "Implements Demixed Principal Components Analysis"
LONG_DESCRIPTION = DESCRIPTION
NAME = "dPCA"
AUTHOR = "Machens Lab"
AUTHOR_EMAIL = "christian.machens@neuro.fchampalimaud.org"
MAINTAINER = "Alex Williams"
MAINTAINER_EMAIL = "alex.h.willia@gmail.com"
DOWNLOAD_URL = 'https://github.com/ahwillia/dPCA'
LICENSE = 'MIT'
VERSION = '0.1'

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=DOWNLOAD_URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['dPCA'],
      package_data={}
     )

from Cython.Build import cythonize

setup(
    ext_modules = cythonize("dPCA/nan_shuffle.pyx"),
    include_dirs = [numpy.get_include()]
)
