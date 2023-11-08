from __future__ import division, absolute_import, print_function
import os

from setuptools import setup, Extension
import shutil

import numpy as np
numpy_lib = os.path.split(np.__file__)[0]
numpy_include = os.path.join(numpy_lib, 'core/include')

ext = Extension("ctw",
                sources=["ctw.pyx",  "cppctw.cpp"],
                libraries=["m"],
                language="c++",
                extra_link_args=["-std=c++11"],
                include_dirs=[numpy_include],
               )

setup(name="ctw",
      version="0.0.1",
      author="Kieran Murphy",
      description="Implementation of infinite depth context tree weighting (CTW)"
      " for the explicit purpose of estimating entropy rate of a symbolic sequence (though you could do other stuff too)",
      ext_modules=[ext])

if __name__ == '__main__':
  import ctw
  ctw.estimate_entropy([1, 0, 0, 1], 2)



# import os

# # I don't know how you want to build your extension or your file structure, 
# # so removing the build stuff.

# your_modulename=Extension('_extensionname',
#                           sources=['path/to/extension.cpp', 'more/file/paths'],
#                           language='c'
#                          )


# # set-up script
# setup(
#       name=DISTNAME
#     , version=FULLVERSION
#     , description= DESCRIPTION
#     , author= AUTHOR
#     , author_email= EMAIL
#     , maintainer= AUTHOR
#     , maintainer_email= EMAIL
#     , long_description=LONG_DESCRIPTION
#     , ext_modules=[your_modulename]
#     , packages=['package_py']
#     )