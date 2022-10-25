import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "boost",
        ["./src/boost/evaluation.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

if __name__ == '__main__':
    setup(
        name="Calculate True Positive Count",
        ext_modules=cythonize(extensions, language_level="3"),
    )
