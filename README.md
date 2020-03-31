# POCKPy

---

POCKPy is a Python 3 package for closed orbit analysis in HL-LHC. It is based on a linear treatment where the closed orbit perturbation of correctors and element errors is computed analytically and stored inside response matrices. The framework relies on input in the form of either MAD-X `.tfs` files from `TWISS` calls, or direct provision of a MAD-X script defining HL-LHC.

POCKPy was developed at CERN, with support from the HL-LHC project, as part of a Master's thesis available [here](http://lup.lub.lu.se/student-papers/record/8998721). It is distributed under the [GPLv3](LICENSES/GPLv3.txt) license, with the exception of code relying on `cpymad` interaction which relies in part on the [MAD-X](LICENSES/MADX_LICENSE.txt) license

**NOTE:** Still undergoing further development!

## Table of contents

 * [Requirements](#requirements)
 * [Installation](#installation)
 * [Quick start](#quick-start)

## Requirements

The following packages are necessary to install POCKPy:

* `pandas`
* `numpy`
* `cvxpy`
* `cpymad`
* `tfs-pandas`
* `pyyaml`

## Installation

To install POCKPy, clone the repository and run `pip install` in the root directory. To uninstall, run `pip uninstall pockpy`.

## Quick start

For example usage, see the Jupyter notebooks in the `example` folder.

## API Reference

A Sphinx-based HTML API reference can be built by running `make` in the `doc` folder, granted Sphinx is available (`pip install sphinx sphinx_rtd_theme`). This GitLab repository should be configured as to put this reference in GitLab Pages for this repository, but it could be that this functionality is not available yet on gitlab.cern.ch, as it does not show up.
