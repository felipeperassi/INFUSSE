import sys
from distutils.core import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

setup(
    name='INFUSSE',
    version='1.0',
    author='Kevin Michalewicz',
    author_email='k.michalewicz22@imperial.ac.uk',
    description='Node regression framework with Large Language Models and Graph Convolutional Networks.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['infusse', 'infusse.dataset', 'infusse.model', 'infusse.utils'],
    install_requires=['biopython', 'logomaker', 'matplotlib', 'numpy', 'pandas', 'requests', 'scikit-learn', 'scipy', 'torch', 'torch_geometric', 'transformers'],
    cmdclass={'test': PyTest}
)
