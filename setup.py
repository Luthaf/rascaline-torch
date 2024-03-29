import glob
import os
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel

from distutils.command.install import install  # isort: skip

ROOT = os.path.realpath(os.path.dirname(__file__))

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")


class universal_wheel(bdist_wheel):
    # When building the wheel, the `wheel` package assumes that if we have a
    # binary extension then we are linking to `libpython.so`; and thus the wheel
    # is only usable with a single python version. This is not the case for
    # here, and the wheel will be compatible with any Python >=3.6. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


class cmake_ext(build_ext):
    """
    Build the native library using cmake
    """

    def run(self):
        source_dir = ROOT
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(ROOT, "build", "cmake-install")

        try:
            os.mkdir(build_dir)
        except OSError:
            pass

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_FOR_PYTHON:BOOL=ON",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        subprocess.run(
            ["cmake", "-S", source_dir, "-B", build_dir, *cmake_options], check=True
        )
        subprocess.run(["cmake", "--build", build_dir, "--parallel"], check=True)
        subprocess.run(
            ["cmake", "--build", build_dir, "--target", "install"], check=True
        )

        files = glob.glob(os.path.join(install_dir, "lib", "*"))
        assert len(files) == 1
        libfile = files[0]

        output = os.path.join(
            ROOT, self.build_lib, "rascaline_torch", "_rascaline_torch.so"
        )
        self.copy_file(libfile, output)


setup(
    ext_modules=[
        # only declare the extension, it is built & copied as required by cmake
        # in the build_ext command
        Extension(name="rascaline_torch", sources=[]),
    ],
    cmdclass={
        "build_ext": cmake_ext,
        "bdist_wheel": universal_wheel,
        "install": install,
    },
)
