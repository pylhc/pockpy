# POCKPy

---

POCKPy is a Python 3 package for closed orbit analysis in HL-LHC. It is based on a linear treatment where the closed orbit perturbation of correctors and element errors is computed analytically and stored inside response matrices. The framework relies on input in the form of either MAD-X `.tfs` files from `TWISS` calls, or direct provision of a MAD-X script defining HL-LHC.

POCKPy was developed at CERN, with support from the HL-LHC project, as part of a Master's thesis available [here](http://lup.lub.lu.se/student-papers/record/8998721). It is distributed under the [GPLv3](LICENSES/GPLv3.txt) license, with the exception of code relying on `cpymad` interaction which relies in part on the [MAD-X](LICENSES/MADX_LICENSE.txt) license

## Requirements

The following packages are necessary to install POCKPy:

* `pandas`
* `numpy`
* `cvxpy`
* `cpymad`
* `tfs-pandas`
* `pyyaml`

The packages `sphinx` and `sphinx_rtd_theme` are necessary to compile the documentation, but not to install and run POCKPy.

## Installation

To install POCKPy, clone the repository and run `pip install .` in the root directory. To uninstall, run `pip uninstall pockpy`.

## Quick start

For example usage, see the Jupyter notebooks in the `example` folder.

## API Reference

A reference with some minimal examples is available [here](https://jodander.github.io/pockpy/). In the current form of the repository, the documentation is updated by running `make html` in the `build/docs` directory and copying the output in `build/docs/output/html` to `docs`.
