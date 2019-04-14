from setuptools import setup, find_packages

setup(
    name='sequence_sensei',
    version='0.0.1',
    author='Nikoleta Glynatsi, Vince Knight',
    author_email=('glynatsine@cardiff.ac.uk'),
    packages=find_packages('src'),
    package_dir={"": "src"},
    description='A library for performing a genetic algorithm on sequences.',
)
