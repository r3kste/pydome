from setuptools import setup, find_packages

setup(
    name="pydome",
    version="0.1.0",
    author='Anton "r3kste" Beny',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "scipy"],
)
