from setuptools import setup

# This call to setup() does all the work
setup(
    name="pockpy",
    version="0.2.0",
    description="Package dedicated to closed orbit analysis.",
    author="Joel D. Andersson",
    author_email="joel.daniel.andersson@cern.ch",
    license="GPLv3",
    packages=["pockpy"],
    install_requires=["numpy", "pandas", "cpymad", "cvxpy", "tfs-pandas",
                      "pyyaml"],
)

