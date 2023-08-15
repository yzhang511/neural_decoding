from setuptools import setup

with open("requirements.txt") as f:
    require = [x.strip() for x in f.readlines()]

setup(
    name="side_info_decoding",
    version="0.1",
    packages=["side_info_decoding"],
    install_requires=require,
)

