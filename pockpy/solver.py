""" Main module for all analysis and user interaction with POCKPy. """

import os
import warnings
import re
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.linalg
import cvxpy as cp

import pockpy.core_io as core_io
import pockpy.config as config
import pockpy.utils as utils
from pockpy.matrix_builder import MatrixBuilder

class Solver:
    """ Class containing all the main routines for orbit correction analysis.

    **Initialize from TFS files**::

        >>> tfs_filename_dict = {
        ...     'B1' : tfs_filename_b1,
        ...     'B2' : tfs_filename_b2
        ... }
        >>> solver = Solver(tfs_filename_dict=tfs_filename_dict)

    **Initialize from MAD-X file (with sliced quadrupoles)**::

        >>> madx_filename = '/afs/directory/somewhere/test.madx'
        >>> makethin_li = [
        ...     {
        ...         'pattern'='^MQ.*',
        ...         'slice' : 10
        ...     }
        ... ] # Slice all quadrupoles in 10 slices
        >>> solver = Solver(madx_filename=madx_filename,
        ...                 makethin_option_li=makethin_li)

    """


    def __init__(self,
                 madx_filename=None,
                 makethin_option_li=None,
                 tfs_filename_dict=None):

        if tfs_filename_dict is not None:
            twiss_table, summ_table = core_io.tfs_to_tables(tfs_filename_dict)
        elif madx_filename is not None:
            twiss_table, summ_table = core_io.madx_to_tables(madx_filename,
                                                             makethin_option_li)
        else:
            twiss_table, summ_table = None, None

        if twiss_table is not None:
            self.twiss_table = twiss_table.copy()

        self.summ_table = summ_table
        self.parent_twiss_table = twiss_table
        self.twiss_table = twiss_table

        self._offset_mapping = None
        self._orbit_is_wrt_reference_orbit = True
        self._reference_dict = None

        # Initialize variables to be set later on a ``build_matrices`` call.
        self.error_response_matrix = None
        self.corrector_response_matrix = None
        self.error_response_matrix = None
        self.error_table = None

    def _get_all_unique_element_names(self):
        """
        Returns all unique element names across all beams.

        Returns:
            List of element names.
        """
        return utils.remove_duplicates(
            self.twiss_table.index.get_level_values(1))

    def _get_reference_dict(self):
        reference_dict = {}
        element_names = self._get_all_unique_element_names()
        for col in self.error_response_matrix.loc[:,
                (['DIPOLE', 'QUADRUPOLE'], slice(None), 'DX')]:
            element_type, element_name, __ = col
            if element_name in config.CONNECTED_ELEMENTS:
                shifted_elements = config.CONNECTED_ELEMENTS[element_name]
            else:
                shifted_elements = [element_name]

            # Also take into account that sliced quadrupoles must be shifted
            # as well if necessary
            sliced_elements_to_shift = []
            for e in shifted_elements:
                sliced_elements_to_shift += [
                    x for x in element_names if re.search('^'+e, x)
                ]
            shifted_elements += sliced_elements_to_shift
            shifted_elements = utils.remove_duplicates(shifted_elements)
            reference_dict[element_name] = shifted_elements

        return reference_dict

    def _switch_reference(self, incr):
        if self._reference_dict is None:
            self._reference_dict = self._get_reference_dict()
        for error_element, offset_elements in self._reference_dict.items():
            self.error_response_matrix.loc[
                (slice(None), 'X', offset_elements),
                (slice(None), error_element, 'DX')
            ] += incr
            self.error_response_matrix.loc[
                (slice(None), 'Y', offset_elements),
                (slice(None), error_element, 'DY')
            ] += incr

    def give_orbit_wrt_element_center(self):
        """
        Makes all orbit values measured at quadrupoles or dipoles offset
        by their misalignment.

        **NOTE**: This also applies to sliced optics, where offsetting a quadrupole
        also changes the reference of all sliced quadrupoles belonging to it.
        Elements that are considered connected as per
        :py:data:`config.CONNECTED_ELEMENTS` will also be affected.

        To instead make all orbit values provided with respect to the ideal
        orbit :py:func:`give_orbit_wrt_reference_orbit`'.
        """
        if not self._orbit_is_wrt_reference_orbit:
            print('Orbit is already given wrt to element centers..')
            return

        self._switch_reference(-1.0)
        self._orbit_is_wrt_reference_orbit = False

    def give_orbit_wrt_reference_orbit(self):
        """
        Makes all orbit values given with respect to the ideal orbit.

        To instead make all orbit values at dipoles and quadrupoles provided
        with respect to their center, call
        :py:func:`give_orbit_wrt_reference_orbit`'.
        """
        if self._orbit_is_wrt_reference_orbit:
            print('Orbit is already wrt to reference orbit..')
            return

        self._switch_reference(1.0)
        self._orbit_is_wrt_reference_orbit = True

    def _change_offsets(self, bpm_mapping, incr):
        for bpm, element in bpm_mapping.items():
            try:
                self.error_response_matrix.loc[
                    (slice(None), 'X', bpm), (slice(None), element, 'DX')] += incr
                self.error_response_matrix.loc[
                    (slice(None), 'Y', bpm), (slice(None), element, 'DY')] += incr
            except KeyError:
                raise ValueError(f'The mapping {bpm} : {element} failed!')

    def attach_bpms(self, bpm_mapping):
        """ Attaches BPMs to either quadrupoles or dipoles.

        This function makes it such that when the element assigned to a
        given BPM is transversally misaligned, so is the BPM.
        This misalignment of the BPM is not reflected in the error table.

        In an example::

            >>> bpm_mapping = {
            ...     'BPM1' : 'MQ1'
            ... }
            >>> solver.attach_bpms(bpm_mapping)
            >>> errors = {
            ...     'QUADRUPOLE' : {
            ...         '^MQ1$' : {
            ...             'DX' : 1
            ...         }
            ...     }
            ... }
            >>> solver.add_errors(errors)
            >>> solver.error_table.at['QUADRUPOLE', 'MQ1', 'DX']
            ... 1.0
            >>> solver.error_table.at['BPM', 'BPM1', 'DX']
            ... 0.0

        Any addtional error added to an attached BPM is independent of said
        attached element.

        Args:
            bpm_mapping(dict): Dict mapping from a BPM (list of which can be
                retrieved by calling :py:func:`get_bpm_list`) to an orbit
                perturbation source (list of which found by calling
                :py:func:`get_orbit_perturbation_element_list`). BPMs
                not included in the mapping are assumed to be detached
                from all other elements.
        Raises:
            :py:exc:`ValueError`: If any individual mapping is incorrect.
        """

        # Remove the previous mapping..
        self.detach_all_bpms()

        # Apply the new one
        self._change_offsets_v2(bpm_mapping, -1.0)
        self._offset_mapping = bpm_mapping

    def detach_all_bpms(self):
        """ Detaches all BPMs. See :py:func:`attach_bpms` for more info. """
        if self._offset_mapping is not None:
            self._change_offsets_v2(self._offset_mapping, 1.0)
            self._offset_mapping = None

    def get_free_bpms(self):
        """  Returns a list of all BPMs that are not attached to any source of
        orbit perturbation. """
        bpm_li = self.get_bpm_list()
        if self._offset_mapping is None:
            return bpm_li
        else:
            return [x for x in bpm_li if x not in self._offset_mapping]

    def build_matrices(self,
                       section=None,
                       keep_all_by_default=True,
                       keyword_li=None,
                       pattern_li=None,
                       concatenate_elements=True):
        """ Builds the response matrices and error table.

        .. note:: This function must be run before any analysis is performed.

        **Example**::

            >>> solver = Solver(madx_filename='/home/nice_machine.madx')
            >>> section = {
            ...     'LHCB1' : ('MQ.25L5.B1', 'MQ.25R5.B1'),
            ...     'LHCB2' : ('MQ.25L5.B2', 'MQ.25R5.B2')
            ... }
            >>> solver.build_matrices(
            ...     section=section,
            ...     keep_all_by_default=True, # Use all elements..
            ...     keyword_li=['DRIFT'], # .. but exclude all drift spaces..
            ...     pattern_li=['^IP[1-8'] # .. and all IPs!
            ... )
            >>> solver.build_matrices(
            ...     section=section,
            ...     keep_all_by_default=False, # Only include the essentials..
            ...     keyword_li=['SEXTUPOLE'], # .. but also add all sextupoles..
            ...     pattern_li=['^IP[1-8'] # .. and all IPs!
            ... )

        Args:
            section(dict): Dict of the form::

                    section = {
                        beam : (end, start),
                        ...
                    }

                where beam is the beam name and (end, start) correspond to
                element names which define the inclusive end and start of
                the new section to be considered.
            keep_all_by_default(bool): If True, defaults to keeping all
                elements for the active section, whereby :py:data:`keyword_li`
                and :py:data:`pattern_li` are used to exclude elements.
                Conversely, if False, only elements capable of inducing
                orbit perturbation are kept, where instead :py:data:`keyword_li`
                and :py:data:`pattern_li` are used to include additional
                elements beyond the minimal ones.
            keyword_li(list): A list of keywords to match the 'KEYWORD' column
                in the Twiss table. Allows for exclusion or inclusion depending
                on the value of :py:data:`keep_all_by_default`.
            pattern_li(list): A list of patterns to match the element names
                in the Twiss table. Allows for exclusion or inclusion depending
                on the value of :py:data:`keep_all_by_default`.
            concatenate_elements(bool): If True, merges error sources according
                to the :py:attr:`CONNECTED_ELEMENTS` setting in the
                config file.
        """

        matrix_builder = MatrixBuilder(self.parent_twiss_table, self.summ_table)
        matrix_builder.build_matrices(
            section=section,
            keep_all_by_default=keep_all_by_default,
            keyword_li=keyword_li,
            pattern_li=pattern_li,
            concatenate_elements=concatenate_elements,
        )
        self.twiss_table = matrix_builder.twiss_table
        self.error_table = matrix_builder.error_table
        self.corrector_response_matrix = matrix_builder.corrector_response_matrix
        self.error_response_matrix = matrix_builder.error_response_matrix

    def _compute_weighted_pseudoinverse(self, n_x, n_y=None, correct_at=None,
                                        bpm_scaling=None,
                                        corrector_scaling=None):
        """ Computes a weighted pseudoinverse of the corrector response matrix
        evaluated at all BPMs.

        Args:
            correct_at(dict): Dict of the form:
                ::

                    correct_at = {
                        beam1 : {
                            bpm1 : 'X',
                            bpm2 : 'XY',
                            ...
                        },
                        beam2: {
                            bpm3 : 'Y',
                            ...
                        },
                        ...
                    }

                where each 3-tuple formed by taking two successive keys and
                their respective value forms a row to correct for.
            n_x(int) : Number of singular values in the horizontal
                plane for the correction.
            n_y(int) : Number of singular values in the vertical
                plane for the correction. Defaults to
                :py:data:`n_x` if not specified.
            bpm_scaling(pandas.Series) : Scaling of the BPMs.
                A valid Series can be retrieved by calling
                :py:func:`get_bpm_reading_series`. Weights have to be in the
                range [0,1) where a higher value emphasizes the entry for
                the correction.
            corrector_scaling(pandas.Series): Scaling of the correctors.
                A valid Series can be retrieved by calling
                :py:func:`get_corrector_series`. Weights have to be in the
                range [0,1) where a higher value emphasizes the entry for
                the correction.
        Returns:
            A DataFrame containing the weighted pseudoinverse of the
            corrector response matrix.
        Raises:
            :py:exc:`ValueError`: On incorrect input.
        """

        # Define response matrices evaluated at all BPMs
        RMc, __ = self._get_response_matrices_at_bpms()

        if corrector_scaling is None:
            corrector_scaling = self.get_corrector_series()
            corrector_scaling[:] = 1.0
        else:
            # Make sure that the input corrector scaling is valid.
            if not corrector_scaling.index.equals(RMc.columns):
                raise ValueError(
                    "corrector_scaling index does not match columns in RMc.")
            elif ((corrector_scaling < 0.0).any()
                    or (corrector_scaling > 1.0).any()):
                raise ValueError(
                    "Entry outside allowed range found in corrector_scaling.")

        if bpm_scaling is None:
            bpm_scaling = self.get_bpm_reading_series()
        else:
            # Make sure that the input BPM scaling is valid.
            if not bpm_scaling.index.equals(RMc.index):
                raise ValueError(
                    "bpm_scaling index does not match index of RMc.")
            elif (bpm_scaling < 0.0).any() or (bpm_scaling > 1.0).any():
                raise ValueError(
                    "Entry outside allowed range found in bpm_scaling.")
            bpm_scaling = bpm_scaling.copy() # We do not want to alter the input

        if correct_at is not None:
            # Transform correct_at data structure to index format
            correct_at_idx = self._correct_at_to_idx(correct_at)

            # Make sure that all entries to be corrected are valid.
            if not all([k in bpm_scaling.index for k in correct_at_idx]):
                raise ValueError(
                        "correct_at has entries not found in the index of RMc.")

            bpm_scaling.loc[correct_at_idx] += 1.0

        # If the provided scalings have no non-zero values, then weight equally.
        if bpm_scaling.sum() == 0.0:
            bpm_scaling[:] = 1.0
        if corrector_scaling.sum() == 0.0:
            corrector_scaling[:] = 1.0

        # Convoluted, but faster than naively defining the scaling as diagonal
        # matrices and performing matrix multiplication.
        # Equivalent to: D_corr @ pinv(D_bpm @ RMc @ D_corr) @ D_bpm
        temp = utils.pinv_per_dim(
            corrector_scaling.to_numpy() * RMc * bpm_scaling.to_numpy()[:, None],
            n_x=n_x,
            n_y=n_y)
        return bpm_scaling.to_numpy() * temp * corrector_scaling.to_numpy()[:, None]



    def compute_correction_matrices(self, n_x, n_y=None, correct_at=None,
                                    bpm_scaling=None,
                                    corrector_scaling=None):
        """ Computes two DataFrames which map from errors to corrector strengths
        and errors to residual closed orbit.

        **Example**::

            >>> bs = solver.get_bpm_reading_series()
            >>> bs[:] = 1e-3 # Weight all BPMs equally..
            >>> bs.at['B1', 'X', 'BPM3'] = 1e-1 # except this important one for B1 in X
            >>> cs = solver.get_corrector_series()
            >>> cs[:] = 1 # Weight all correctors equally..
            >>> cs['C5'] = 0.0 # except for C5 which we do not want to use
            >>> correct_at = {
            ...     'B1' : {
            ...         'BPM7' : 'XY'
            ...     }
            ... } # Correct beam 'B1' agressively in both planes at BPM 'BPM7'
            >>> A, B = solver.compute_correction_matrices(
            ...     n_x=10, # Use 10 singular values in the horizontal plane..
            ...     n_y=20, # .. and 20 in the vertical plane.
            ...     correct_at=correct_at,
            ...     bpm_scaling=bs,
            ...     corrector_scaling=cs
            ... )
            >>> A @ solver.error_table # Gives you corrector strength of correction
            >>> B @ solver.error_table # Gives you residual orbit of correction

        Args:
            correct_at(dict): Dict of the form:
                ::

                    correct_at = {
                        beam1 : {
                            bpm1 : 'X',
                            bpm2 : 'XY',
                            ...
                        },
                        beam2: {
                            bpm3 : 'Y',
                            ...
                        },
                        ...
                    }

                where each 3-tuple formed by taking two successive keys and
                their respective value forms a row to correct for.
            n_x(int) : Number of singular values in the horizontal
                plane for the correction.
            n_y(int) : Number of singular values in the vertical
                plane for the correction. Defaults to
                :py:data:`n_x` if not specified.
            bpm_scaling(pandas.Series) : Scaling of the BPMs.
                A valid Series can be retrieved by calling
                :py:func:`get_bpm_reading_series`. Weights have to be in the
                range [0,1) where a higher value emphasizes the entry for
                the correction.
            corrector_scaling(pandas.Series): Scaling of the correctors.
                A valid Series can be retrieved by calling
                :py:func:`get_corrector_series`. Weights have to be in the
                range [0,1) where a higher value emphasizes the entry for
                the correction.
        Returns:
            Two DataFrames mapping from errors to corrector strength and errors
            to residual closed orbit.
        Raises:
            :py:exc:`ValueError`: On incorrect input.
        """

        correction_matrix = self._compute_weighted_pseudoinverse(
                n_x=n_x,
                n_y=n_y,
                correct_at=correct_at,
                bpm_scaling=bpm_scaling,
                corrector_scaling=corrector_scaling)

        RMc, RMe = self._get_response_matrices_at_bpms()

        A = -correction_matrix @ RMe
        B = self.error_response_matrix + self.corrector_response_matrix @ A
        return A, B

    def _get_all_elements_in_range(self, beam, start, end):
        """ Returns all elements in a given range for a given beam.

        Args:
            beam(str): Beam to check for.
            start(str): Element defining the inclusive start of the range.
            end(str): Element defining the inclusive end  of the range.
        Returns:
            List of all elements in the given interval.
        """

        idx_start = self.twiss_table.loc[beam].index.get_loc(start)
        idx_end = self.twiss_table.loc[beam].index.get_loc(end)
        if idx_start <= idx_end:
            elements = list(
                self.twiss_table.loc[beam].iloc[idx_start:idx_end+1].index)
        else:
            elements = list(
                self.twiss_table.loc[beam].iloc[idx_start:].index)
            elements += list(
                self.twiss_table.loc[beam].iloc[:idx_end+1].index)

        return elements


    def add_errors(self, errors, overwrite=False, element_range=None):
        """ Adds machine errors to the error table.

        Args:
            errors(dict): A dict of the form::

                    errors = {
                        'DIPOLE' : {
                            '^MBXF' : {
                                'DX' : 1e-3,
                                'DPSI : 1e-6
                            },
                            '^MBRD' : {
                                'DKR0' : 1e-3
                            }
                        },
                        ...
                    }

                The first key is one of the possible error types, the second
                key is a regular expression matching elements fitting into said
                category of error types and the innermost dict maps from
                applicable errors to their numerical value.

                **NOTE**: Elements that get matched multiple times in one call
                will be incremented each time if :py:data:`overwrite` is False,
                if it :py:data:`overwrite` is True then the final
                value will be arbitrary.
            overwrite(bool): Decides whether the call overwrites
                entries inthe  error_table or if they are incremented.
            element_range(dict): Dictionary of form {beam : (start, end)} which
                specifies the inclusive range over which errors are to be added.
                If not specified, all elements will be considered.
        """

        # First define the range of elements affected.
        error_elements = set(self.get_error_element_list())

        # Only consider the elements within the given range, if provided
        if element_range is not None:
            elements_in_range = set()
            for beam, tup in element_range.items():
                elements_in_range = elements_in_range.union(
                    self._get_all_elements_in_range(beam, *tup))

            # Handle if elements have been concatenated
            for err_elmt_name, elmt_li in config.CONNECTED_ELEMENTS.items():
                if len(elements_in_range.intersection(elmt_li)) > 0:
                    elements_in_range.add(err_elmt_name)

            error_elements = error_elements.intersection(elements_in_range)

        for error_type, pattern_dict in errors.items():
            for pattern, error_dict in pattern_dict.items():
                elmts = [x for x in error_elements if re.search(pattern, x)]
                for err_type, err_val in error_dict.items():
                    if overwrite:
                        self.error_table.loc[
                            error_type, elmts, err_type] = err_val
                    else:
                        # Convoluted, but the naive solution w/ +=
                        # does not work for pandas 0.25.
                        self.error_table.loc[
                            error_type, elmts, err_type] = self.error_table.loc[
                            error_type, elmts, err_type].to_numpy() + err_val

    def reset_errors(self):
        """ Removes all errors from the error table."""
        self.error_table[:] = 0

    def get_bpm_list(self):
        """ Returns a list of all BPMs in the Twiss table. """
        return utils.remove_duplicates(
            self.error_table.loc['BPM'].index.get_level_values(0))

    def get_orbit_perturbation_element_list(self):
        """ Returns a list of all elements able to induce closed orbit
        perturbation. Equivalent to all element with non-zero dipole or
        quadrupole field."""
        return utils.remove_duplicates(
            self.error_table.loc[['QUADRUPOLE', 'DIPOLE']].index.get_level_values(1))

    def get_error_element_list(self):
        """ Returns a list of all elements capable of having an error of
        some kind. """
        return utils.remove_duplicates(
            self.error_table.index.get_level_values(1))

    def _get_response_matrices_at_bpms(self):
        """ Returns the response matrices evaluated at all BPMs, for X and Y.

        Returns the error and corrector response matrices evaluated at all BPMs
        stored in self.twiss_table, for X and Y. Function used internally when
        the response matrices are to simulate the real life scenarios of error
        correction, as the closed orbit of the beam can only be measured at
        BPMs, and only for horizontal/vertical position.

        Returns:
            Corrector and error response matrices evaluated at all BPMs for
            X and Y.
        """

        bpm_li = self.get_bpm_list()
        RMc = self.corrector_response_matrix.loc[
            (slice(None), ['X', 'Y'], bpm_li), :]
        RMe = self.error_response_matrix.loc[
            (slice(None), ['X', 'Y'], bpm_li), :]
        return RMc, RMe

    def get_bpm_reading_series(self):
        """ Returns a Series where each entry is for a BPM in X or Y. """
        RMc, __ = self._get_response_matrices_at_bpms()
        return pd.Series(0.0, index=RMc.index)

    def get_corrector_series(self):
        """ Returns a Series where each entry is a corrector. """
        return pd.Series(0.0, index=self.corrector_response_matrix.columns)

    def linear_correction(self, n_x, n_y=None, correct_at=None, mode='normal',
                          bpm_scaling=None, corrector_scaling=None):
        """ Performs a linear correction of the current errors.

        Performs a pseudoinverse-based correction of the current errors and
        returns the original perturbation, the corrector strength used for the
        correction and the final residual.

        Args:
            correct_at(dict): Dict of the form:
                ::

                    correct_at = {
                        beam1 : {
                            bpm1 : 'X',
                            bpm2 : 'XY',
                            ...
                        },
                        beam2: {
                            bpm3 : 'Y',
                            ...
                        },
                        ...
                    }

                where each 3-tuple formed by taking two successive keys and
                their respective value forms a row to correct for.
            mode(str): Specifies how the errors stored in
                :py:attr:`Solver.error_table` are to be interpreted. Mode can
                assume any of the following three values:

                ``normal``
                    Correction is performed on the errors stored in
                    :py:attr:`error_table` where they assume to represent a specific
                    machine.

                ``worst-case``
                    Correction assumes each error is described by a
                    zero-mean uniform distribution with boundaries given by
                    stored errors (in absolute), and then returns the worst-case
                    perturbation, correction strength and residual
                    (in absolute value) for the given correction strategy, per
                    entry in the output.

                ``rms``
                    Correction assumes each error is described by a
                    zero-mean uniform distribution with boundaries given by
                    stored errors (in absolute), and then returns the RMS values
                    for the perturbation, corrector strength and residual.
            n_x(int) : Number of singular values in the horizontal
                plane for the correction.
            n_y(int) : Number of singular values in the vertical
                plane for the correction. Defaults to
                :py:data:`n_x` if not specified.
            bpm_scaling(pandas.Series) : Scaling of the BPMs.
                A valid Series can be retrieved by calling
                :py:func:`get_bpm_reading_series`. Weights have to be in the
                range [0,1) where a higher value emphasizes the entry for
                the correction.
            corrector_scaling(pandas.Series): Scaling of the correctors.
                A valid Series can be retrieved by calling
                :py:func:`get_corrector_series`. Weights have to be in the
                range [0,1) where a higher value emphasizes the entry for
                the correction.
        Returns;
            Two Series, corresponding to:

            1. Corrector strength used for correction.
            2. Residual orbit post-correction.

        Raises:
            :py:exc:`ValueError`: On incorrect input.
        """

        A, B = self.compute_correction_matrices(
            n_x=n_x,
            n_y=n_y,
            correct_at=correct_at,
            bpm_scaling=bpm_scaling,
            corrector_scaling=corrector_scaling)

        # Computes the corrector strengths, residual and original perturbation
        if mode == 'normal':
            corr_strength = A @ self.error_table
            residual = B @ self.error_table
        elif mode == 'worst-case':
            corr_strength = A.abs() @ self.error_table.abs()
            residual = B.abs() @ self.error_table.abs()
        # NOTE: Returns 1*RMS values
        elif mode == 'rms':
            # As we only want the diagonal elements of the covariance matrices
            # of the corrector strength and residual orbit, we only need to
            # compute the inner product of A and B with themselves, scaled by
            # the appropriate errors.
            # Equivalent to: np.sqrt(np.diag(A.T @ D_err_cov @ A))
            # .. but considerably faster!
            error_cov = (self.error_table ** 2) / 3
            corr_strength = ((A ** 2) * error_cov).sum(axis=1) ** 0.5
            residual = ((B ** 2) * error_cov).sum(axis=1) ** 0.5

        return corr_strength, residual

    # NOTE: The MultiIndex has no guaranteed order. Does not matter for current
    # applications, could matter for future applications.
    def _correct_at_to_idx(self, correct_at):
        """ Constructs a pandas MultiIndex from a 'correct_at' dictionary.

        Internal method used to produce a pandas MultiIndex from a dictionary
        of the form {beam : {element : [dimensions]}}, where the result is a
        three-dimensional MultiIndex of the form (beam, dimension, element).

        Returns:
            A three-dimensional pandas MultiIndex of the form
            (beam, element, dimension).
        """
        li = []
        for beam in correct_at:
            for elmt, dims in correct_at[beam].items():
                for dim in dims:
                    li.append((beam, dim, elmt))
        return pd.MultiIndex.from_tuples(li)

    def _minimize(self, x, f, constraints, cvxpy_options=None):
        """ Solve a convex constrained minimization problem using
        :py:mod:`cvxpy`.

        Args:
            x(cvxpy.Variable) : A cvxpy.Variable object corresponding to the
                variable to optimize over.
            f(cvxpy.Expression) : A convex function defined in cvxpy.
            constraints(list): A list of :py:class:`cvxpy.Expression`
                defining constraints on :py:data:`x`.
            cvxpy_options(dict): A dictionary of inputs to supply to
                :py:func:`cvxpy.Problem.solve` when it is called for the
                minimization. For possible options, see
                https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options
        Returns:
            If succesful, a :py:attribute:`numpy.ndarray` of same dimension as x.
        Raises:
            :py:exc:`RuntimeError`: On unbounded or infeasible optimization
                problem.

        """
        # Define the optimization problem
        prob = cp.Problem(cp.Minimize(f), constraints)

        # Solve the problem
        if cvxpy_options is None:
            cvxpy_options = {'verbose' : False,
                             'abstol_inacc' : 1e-3,
                             'reltol_inacc' : 1e-3,
                             'solver' : cp.ECOS}
        prob.solve(**cvxpy_options)

        # Check what the solver returned
        # See: https://www.cvxpy.org/tutorial/intro/index.html#infeasible-and-unbounded-problems
        if prob.status != cp.OPTIMAL:
            if prob.status == cp.INFEASIBLE:
                raise RuntimeError(
                    'Optimization problem defined is infeasible.')
            elif prob.status == cp.UNBOUNDED:
                raise RuntimeError(
                    'Optimization problem defined is unbounded.')
            else:
                warnings.warn('CVXPY found an inaccurate solution',
                              UserWarning, stacklevel=1)
        return x.value

    def convex_correction(self, orbit_bound, cs_bound, w=0.0, correct_at=None,
                          cvxpy_options=None):
        """ Corrects the errors as a bounded optimzation problem.

        Args:
            correct_at(dict): Dict of the form::

                    correct_at = {
                        beam1 : {
                            bpm1 : 'X',
                            bpm2 : 'XY',
                            ...
                        },
                        beam2: {
                            bpm3 : 'Y',
                            ...
                        },
                        ...
                    }

                where each 3-tuple formed by taking two successive keys and
                their respective value forms a row to correct for.
            orbit_bound(pandas.Series): Bound on the absolute orbit at BPMs.
                A valid Series can be retrieved by calling
                :py:func:`get_bpm_reading_series`.
            cs_bound(pandas.Series): Bound on the absolute corrector strength.
                A valid Series can be retrieved by calling
                :py:func:`get_corrector_series`.
            w(float): A positive scalar the regularizing term
                penalizing overall use of corrector strength. Used in the
                minimization objective as:
                || resdual_orbit ||_2 + w * || corrector_strength ||
            cvxpy_options(dict): A dictionary of inputs to supply to
                :py:func:`cvxpy.Problem.solve` when it is called for the
                minimization. For possible options, see
                https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options
        Returns:
            Two Series, the corrector strength used and the
            residual post-correction.
        Raises:
            :py:exc:`ValueError`: On invalid input.
            :py:exc:`RuntimeError`: On failing optimization.

        """

        # Define the response matrices evaluated only at BPMs
        RMc, RMe = self._get_response_matrices_at_bpms()

        # Define perturbation as seen by BPMs
        pert = RMe @ self.error_table

        # Verify that cs_bound and orbit_bound are valid pandas Series
        if not cs_bound.index.equals(RMc.columns):
            raise ValueError(
                'Index of cs_bound does not match columns of RMc.')
        if not orbit_bound.index.equals(RMc.index):
            raise ValueError(
                'Index of orbit_bound does not match index of RMc.')

        if correct_at is not None:
            correct_at_idx = self._correct_at_to_idx(correct_at)

            if not all([k in RMc.index for k in correct_at_idx]):
                raise ValueError(
                    'Entries in correct_at not found in RMc.')

            orbit_bound = orbit_bound.copy()
            orbit_bound.loc[correct_at_idx] = 0.0

        # Define indices for equality and inequality constraints
        all_idx = RMc.index
        ineq_idx = [x for x in all_idx if orbit_bound[x] > 0.0]
        eq_idx = [x for x in all_idx if x not in ineq_idx]

        # Define the contraints
        RMc_eq = RMc.loc[eq_idx,:].to_numpy()
        RMc_ineq = RMc.loc[ineq_idx,:].to_numpy()
        pert_eq = pert.loc[eq_idx].to_numpy()
        pert_ineq = pert.loc[ineq_idx].to_numpy()

        x = cp.Variable(RMc.shape[1])
        constraints = [
            cp.abs(RMc_ineq @ x + pert_ineq) <= orbit_bound.loc[ineq_idx],
            cp.abs(x) <= cs_bound
        ]

        # If any equality constraints exist, add them
        if len(eq_idx) > 0:
            constraints.append(RMc_eq @ x == - pert_eq)

        # Define objective and solve the problem
        func_to_minimize = cp.norm(
            RMc.to_numpy() @ x + pert, p=2) + w * cp.norm(x, p=2)
        sol = self._minimize(x, func_to_minimize, constraints, cvxpy_options)

        cs = pd.Series(sol, index=RMc.columns)
        res = self.compute_perturbation() + self.corrector_response_matrix @ cs

        return cs, res

    def compute_perturbation(self, errors=None):
        """ Computes the perturbation caused by the errors in the error table.

        Args:
            errors(dict): Errors to :py:func:`add_errors`. This argument is
                optional. If provided the output perturbation will be that
                induced by the errors in :py:data:`errors`.
        Returns:
            A Series for the perturbation evaluated at all elements.
        """
        if errors is None:
            pert = self.error_response_matrix @ self.error_table
        else:
            temp = self.error_table.copy()
            self.reset_errors()
            self.add_errors(errors)
            pert = self.compute_perturbation()
            self.error_table = temp
        return pert

    def knob_implementation(self,
                            cs_bound,
                            c_weight=None,
                            force_parity=None,
                            orbit_regularizer=None,
                            knob_names=None,
                            cvxpy_options=None):
        """ Provides an optimal solution for the implementation of a set of
        knobs.

        **NOTE**: Possible knobs to implement are read from the YAML file
        specified by :py:attr:`config.KNOB_DEFINITIONS_PATH`.

        Args:
            cs_bound(pandas.Series): Bound on the absolute corrector strength.
                A valid Series can be retrieved by calling
                :py:func:`get_corrector_series`.
            c_weight(pandas.Series): Weight for the overall corrector strength
                in the L2-minimization. A valid Series can be retrieved by
                calling :py:func:`get_corrector_series`.
            force_parity(dict): Argument to enforce parity among correctors.
                Argument must take the form of::

                    force_parity = {
                        knob_name1 : [
                            [(p_1, c_1), (p_2, c_2), ...],
                            ...
                        ],
                        ...
                    }

                where every p_i is plus-minus 1, and every c_i is a corrector.
                This argument works as follows: inside each sublist,
                the correctors inside the list will always be powered with the
                same magnitude and with sign given by p_i. For example,::

                    force_parity = {
                        'example' : [
                            [(1, c1), (-1, c2), (1, c3)]
                        ]
                    }

                Here the correctors c1, c2 and c3 are enforced to be powered
                with the same magnitude, where c1 and c3 are to be powered with
                the same sign, and c2 with the opposite sign to c1 and c3, when
                implementing the knob named 'example'.

            orbit_regularizer(dict): Regularizing orbit term for the minimization
                function. Argument must be of the form::

                    orbit_regularizer = {
                        knob_name : (w, row_li, p),
                        ...
                    }

                where each valid term entry in the dictionary adds another term
                to the minimization function as per:

                    min_fun += w*||knob_name_orbit.loc[row_li]||_p

                w is a positive scalar for the added term, row_li a list of indices
                to include in the vector to be minimized and p is the norm to
                be used (np.inf for infinity norm).
            knob_names(list): A list of knobs to be implemented. If not
                provided the method will default to implementing all defined
                knobs.
            cvxpy_options(dict): A dictionary of inputs to supply to
                :py:func:`cvxpy.Problem.solve` when it is called for the
                minimization. For possible options, see
                https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options
            Returns:
                The corrector strength and implemented knob orbits in DataFrames.
        """
        # Knob definition gets parsed from the .yml file provided
        knobs = config.get_knob_definition()

        if knob_names is None:
            knob_names = list(knobs.keys())

        # Nifty to define beforehand.
        RMc = self.corrector_response_matrix
        added_orbits = pd.DataFrame(0.0, index=RMc.index, columns=knob_names)
        knob_rhs_dict = OrderedDict()
        knob_mat_dict = OrderedDict()

        # Load and process all relevant information about the knobs.
        for knob_name in knob_names:
            data = knobs[knob_name]
            knob_def = data['orbit_spec']
            corrector_pattern = data['correctors']
            quad_offsets = data['quadrupole_offsets']

            correctors = [c for c in RMc.columns
                          if re.search(corrector_pattern, c)]

            if len(correctors) == 0:
                raise ValueError(
                        f"Knob {knob_name} matches no correctors.")

            rhs = self._knob_def_to_series(knob_def)

            if not all([k in RMc.index for k in rhs.index]):
                raise ValueError(
                    f"Knob {knob_name} has entries not present in RMc.")

            mat = RMc.loc[rhs.index, correctors]

            if quad_offsets is not None:
                error_dict = {
                    'QUADRUPOLE' : quad_offsets
                }
                added_orbits[knob_name] = self.compute_perturbation(
                    errors=error_dict)

            # Detract perturbation from quadrupole movement if applicable
            rhs[:] -= added_orbits.loc[rhs.index, knob_name]

            knob_rhs_dict[knob_name] = rhs
            knob_mat_dict[knob_name] = mat

        # Knobs values stacked vertically, matrices put into a block matrix.
        knob_rhs = pd.concat(knob_rhs_dict)
        knob_mat = scipy.linalg.block_diag(*knob_mat_dict.values())

        # Define the DataFrame for the equality constraint.
        col_li = []
        for knob_name, df in knob_mat_dict.items():
            col_li += [(knob_name, x) for x in df.columns]
        knob_mat = pd.DataFrame(knob_mat, index=knob_rhs.index,
                                columns=pd.MultiIndex.from_tuples(col_li))

        # Define the DataFrame mapping from allowed correctors per knob to the
        # aggregate use of each corrector.
        corr_mat = pd.DataFrame(0.0, index=RMc.columns,
                                columns=knob_mat.columns)
        for col in corr_mat:
            corr_mat.at[col[1], col] = 1.0

        # If the user manually enforces parity between correctors, then it is
        # taken care of here, and further down post-optimization.
        if force_parity is not None:
            for knob_name, list_of_lists in force_parity.items():
                for parity_li in list_of_lists:
                    p1, c1 = parity_li[0]
                    for p2, c2 in parity_li[1:]:
                        try:
                            knob_mat[knob_name, c1] += p1*p2 * knob_mat[knob_name, c2]
                            knob_mat = knob_mat.drop((knob_name, c2), axis=1)

                            corr_mat.at[c2, (knob_name, c1)] = 1.0
                            corr_mat = corr_mat.drop((knob_name, c2), axis=1)
                        except KeyError:
                            raise ValueError(
                                    "Invalid values for force_parity provided.")

        # Set c_weight to unit vector if not specified
        if c_weight is None:
            c_weight = pd.Series(1.0, index=RMc.columns)

        # Define the main function to minimize, based on the corrector strengths
        x = cp.Variable(knob_mat.shape[1])
        func_to_minimize = cp.norm(
            c_weight.to_numpy() * (corr_mat.to_numpy() @ cp.abs(x)), p=2)

        # Add regularizing terms for orbits if any were given
        if orbit_regularizer is not None:
            for knob_name, pen_def in orbit_regularizer.items():
                w, orb_loc, p = pen_def

                if not all([k in RMc.index for k in orb_loc]):
                    raise ValueError(
                            "Invalid index in orbit_regularizer")

                M = pd.DataFrame(0.0, index=RMc.loc[orb_loc, :].index,
                                columns=knob_mat.columns)
                M.loc[:, knob_name] = RMc.loc[
                    M.index, M.loc[:, knob_name].columns].to_numpy()

                func_to_minimize += w * cp.norm(
                    M.to_numpy() @ x
                    + added_orbits.loc[orb_loc, knob_name].to_numpy(), p=p)

        # Define the constraints
        constraints = [
            corr_mat.to_numpy() @ cp.abs(x) <= cs_bound,
            knob_mat.to_numpy() @ x == knob_rhs
        ]

        # Solve the problem
        sol = self._minimize(x, func_to_minimize, constraints, cvxpy_options)

        # Transform the output into a DataFrame of corrector strengths.
        x_series = pd.Series(sol, index=knob_mat.columns)
        cs_df = pd.DataFrame(0.0, index=RMc.columns, columns=knob_names)
        for idx, val in x_series.items():
            knob_name, corrector = idx
            cs_df.at[corrector, knob_name] = val

        # Make sure that correctors removed in the optimization when enforcing
        # parity are added back in after the optimization.
        if force_parity is not None:
            for knob_name, list_of_lists in force_parity.items():
                for parity_li in list_of_lists:
                    p1, c1 = parity_li[0]
                    for p2, c2 in parity_li[1:]:
                        cs_df.at[c2, knob_name] = p1 * p2 * cs_df.at[c1, knob_name]

        knob_df = RMc @ cs_df + added_orbits
        return cs_df, knob_df

    def _knob_def_to_series(self, knob_def):
        """ Returns the corresponding pandas Series for a given knob defintiion.

        Args:
            knob_def(dict): Dict of the form::

                        {beam : {element : {dimension : value}}}

                where knob_def[beam][element][dim] defines the orbit for beam
                'beam', at element 'element' and in dimension 'dimension' to
                be set to value 'value'.
        Returns:
            A Series indexed by (beam, dim, element).
        """
        index_li = []
        for beam in knob_def:
            for element in knob_def[beam]:
                for dim in knob_def[beam][element]:
                    index_li.append((beam, dim, element))

        series = pd.Series(0.0, index=pd.MultiIndex.from_tuples(index_li))
        for idx in series.index:
            beam, dim, element = idx
            series[beam, dim, element] = knob_def[beam][element][dim]
        return series

    def save_state(self, pickle_name='temp.pickle', overwrite=False,
                   shrink_twiss_table=True):
        """ Saves the solver state to a pickle in
        :py:attr:`config.PICKLE_JAR_PATH`.

        Args:
            pickle_name(str): Name of the pickle file the state is saved to.
            overwrite(bool): If True, overwrite any file with the same name in the
                pickle jar directory.
            shrink_twiss_table: If True, only writes the Twiss table columns in
                :py:attr:`config.MINIMUM_TWISS_COLUMNS` to the pickle.
        Raises:
            :py:exc:`IOError`: If pickle with
                with name :py:data:`pickle_name` already exists and
                :py:data:`overwrite` is False.

        """
        if not re.search(r'.pickle$', pickle_name):
            pickle_name = pickle_name + '.pickle'

        dump_dict = {
            'parent_twiss_table' : self.parent_twiss_table,
            'active_twiss_table' : self.twiss_table,
            'summ_table' : self.summ_table,
            'corrector_response_matrix' : self.corrector_response_matrix,
            'error_response_matrix' : self.error_response_matrix,
            '_orbit_is_wrt_reference_orbit' : self._orbit_is_wrt_reference_orbit,
            '_offset_mapping' : self._offset_mapping
        }

        if shrink_twiss_table and self.parent_twiss_table is not None:
            dump_dict['parent_twiss_table'] = self.parent_twiss_table.loc[:,
                    config.MINIMUM_TWISS_COLUMNS]
            dump_dict['active_twiss_table'] = self.twiss_table.loc[:,
                    config.MINIMUM_TWISS_COLUMNS]

        filename = os.path.join(config.PICKLE_JAR_PATH, pickle_name)

        if os.path.isfile(filename) and not overwrite:
            raise IOError(
                (f"Pickle with name '{pickle_name}' already exists in "
                 f"{config.PICKLE_JAR_PATH}. Call ``save_state`` "
                 'with ``overwrite=True`` to overwrite pickle.'))

        pickling = open(filename, 'wb')
        pickle.dump(dump_dict, pickling, protocol=4)
        pickling.close()
        print(f'Successfully saved state in {filename}')

    def load_state(self, pickle_name='temp.pickle'):
        """ Sets the solver state to a pickle in the
        :py:attr:`config.PICKLE_JAR_PATH`.

        Args:
          pickle_name(str): Name of a .pickle file found in the pickle jar
              directory.
        Raises:
            :py:exc:`IOError`: If :py:data:`picke_name` does not
                match any pickle in the pickle jar.
        """
        if not re.search(r'.pickle$', pickle_name):
            pickle_name = pickle_name + '.pickle'

        filename = os.path.join(config.PICKLE_JAR_PATH, pickle_name)
        try:
            unpickling = open(filename, 'rb')
        except:
            raise IOError(
                (f"No pickle with name '{pickle_name}' found "
                 f" in '{config.PICKLE_JAR_PATH}'."))
        state_dict = pickle.load(unpickling)

        self.twiss_table = state_dict['active_twiss_table']
        self.parent_twiss_table = state_dict['parent_twiss_table']
        self.summ_table = state_dict['summ_table']
        self.corrector_response_matrix = state_dict[
            'corrector_response_matrix']
        self.error_response_matrix = state_dict['error_response_matrix']
        self._orbit_is_wrt_reference_orbit = state_dict[
            '_orbit_is_wrt_reference_orbit']
        self._offset_mapping = state_dict['_offset_mapping']

        if self.error_response_matrix is not None:
            self.error_table = pd.Series(
                0.0, index=self.error_response_matrix.columns)
        else:
            self.error_table = None
        print(f'Successfully loaded state from {filename}')

    def get_saved_states(self):
        """ Returns a list of all saved states in the
        :py:attr:`config.PICKLE_JAR_PATH` path. """
        li = os.listdir(config.PICKLE_JAR_PATH)
        li = [x for x in li if re.search(r'.pickle$', x)]
        return li

