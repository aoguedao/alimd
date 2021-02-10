from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="alimd",
    version="0.1.0",
    description="An approach to local influence with massive data",
    long_description=readme,
    author="Alonso Ogueda",
    author_email="alonso.ogueda@gmail.com",
    url="https://github.com/aoguedao/alimd",
    license=license,
    packages=find_packages(exclude=("tests", "docs"))
)