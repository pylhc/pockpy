""" Module for building the response matrices in POCKPy. """

import re
import itertools

import numpy as np
import pandas as pd

import pockpy.utils as utils
import pockpy.config as config

SPEED_OF_LIGHT = 299792458

class MatrixBuilder:
    """ Class tasked with the construction of response matrices.

    **Example**::

        >>> mb = MatrixBuilder(tt, st)
        >>> section = {
        ...     'LHCB1' : ('MQML.10L5.B1', 'MQML.10R5.B1'),
        ...     'LHCB2' : ('MQML.10L5.B2', 'MQML.10R5.B2')
        ... }
        >>> mb.build_matrices(section, keep_all_by_default=True)
        >>> mb.error_response_matrix # Now built!
        >>> mb.corrector_response_matrix # Also built!

    """

    def __init__(self, twiss_table, summ_table):

        # For some .tfs tables K0L is not present but, K0L=ANGLE
        if 'K0L' not in twiss_table.columns:
            twiss_table = twiss_table.rename(columns={'ANGLE' : 'K0L'})

        self.summ_table = summ_table
        self.twiss_table = twiss_table

        # Gets initialized after ``build_matrices`` is called.
        self.corrector_response_matrix = None
        self.error_response_matrix = None
        self.error_table = None


    # See eq. (1.11) in MAD-X User's Reference Manual 5.04.02
    def _compute_magnetic_rigidity(self, beam):
        """ Computes the magnetic rigidity of a beam.

        Args:
          beam(str): Name of the beam.

        Returns:
          The magnetic rigidity of the given beam.
        """

        p = self.summ_table[beam]['PC'] # GeV / C
        charge = self.summ_table[beam]['CHARGE']
        return p * 1e9 * charge / SPEED_OF_LIGHT

    def _get_response_matrix_index(self):
        """ Returns index for response matrices. """

        idx = []
        dims = ['PX', 'PY', 'X', 'Y']
        for beam in config.SEQUENCE_NAMES:
            elmts = list(self.twiss_table.loc[beam].index)
            idx += list(itertools.product([beam], dims, elmts))
        return pd.MultiIndex.from_tuples(idx, names=['BEAM', 'DIM', 'ELMT'])

    def _get_error_index(self):
        """ Sets up the internally kept error table.

        Internal method that assumes active_twiss_table is a valid pandas
        DataFrame containing the Twiss table for the analysis.

        Returns:
            A pandas Series with index corresponding to all possible sources of
            error-induced perturbation for the given beamline segment defined in
            the initial TFS files.
        """

        # NOTE: Everything with non-zero dipole and/or quadrupole field
        # strength is assumed to be an error source.
        df = self.twiss_table[self.twiss_table['K0L'].abs() > 0]
        dipole_names = list(df.index.get_level_values(1))
        dipole_names = utils.remove_duplicates(dipole_names)

        df = self.twiss_table[self.twiss_table['K1L'].abs() > 0]
        quadrupole_names = list(df.index.get_level_values(1))
        quadrupole_names = utils.remove_duplicates(quadrupole_names)

        # We also add the effect of BPMs shifting around
        bpm_names = self.twiss_table[
            self.twiss_table['KEYWORD'] == 'MONITOR'].index.get_level_values(1)
        bpm_names = [x for x in bpm_names if re.search(r'^BPM.*', x)]
        bpm_names = utils.remove_duplicates(bpm_names)

        # For compliance down the line, we enforce that slices of elements are
        # to be mapped to their original element, e.g. MQXFB.A2R5..1,
        # MQXFB.A2R5..2 etc. will all contribute to the errors of MQXFB.A2R5.
        #
        # Individual slices cannot be given unique values, if this
        # has to be changed at some point then here is the place to start
        pattern = r'\.\.\d+'
        dipole_names = [re.sub(pattern, '', x) for x in dipole_names]
        quadrupole_names = [re.sub(pattern, '', x) for x in quadrupole_names]
        dipole_names = utils.remove_duplicates(dipole_names)
        quadrupole_names = utils.remove_duplicates(quadrupole_names)

        # Define all errors which are currently considered
        dipole_errors = ['DKR0', 'DPSI', 'DX', 'DY']
        quadrupole_errors = ['DKR1', 'DX', 'DY', 'DPSI']
        bpm_errors = ['DX', 'DY']

        # Create the index for error_table
        idx = list(itertools.product(['DIPOLE'], dipole_names,
                                     dipole_errors))
        idx += list(itertools.product(['QUADRUPOLE'], quadrupole_names,
                                      quadrupole_errors))
        idx += list(itertools.product(['BPM'], bpm_names,
                                      bpm_errors))

        return pd.MultiIndex.from_tuples(
            idx, names=['TYPE', 'ERR_ELMT', 'ERR_TYPE'])

    def _get_corrector_index(self):
        """ Returns columns for corrector response matrix. """

        df = self.twiss_table[(self.twiss_table['KEYWORD'] == 'HKICKER')
                              | (self.twiss_table['KEYWORD'] == 'VKICKER')]
        cols = utils.remove_duplicates(list(df.index.get_level_values(1)))

        # Sort them by 'S', just for later convenience
        key = lambda x : self.twiss_table.loc[(slice(None), x), 'S'][0]
        return pd.Index(sorted(cols, key=key), name='CORRS')

    def _compute_corrector_response_matrix(self):
        """ Computes the corrector response matrix.

        This method is run interally as part of the setup_for_analysis call. It
        makes use of the Twiss table to build up the response matrix.

        Returns:
            A pandas DataFrame containing the corrector response matrix.
        """

        response_df = pd.DataFrame(0.0,
                                   index=self._get_response_matrix_index(),
                                   columns=self._get_corrector_index())

        print('Computing corrector matrix..')
        for corr in response_df:
            response_df[corr] = self._corrector_to_pert(corr, response_df.index)

        # We want the corrector strength expressed in [Tm] not [rad]
        for beam in self.summ_table.keys():
            scale_to_tm = self._compute_magnetic_rigidity(beam)
            response_df.loc[(beam, slice(None), slice(None)), :] /= scale_to_tm
        return response_df

    def _compute_error_response_matrix(self, concatenate_elements=True):
        """ Computes the error response matrix.

        This method is run interally as part of the setup_for_analysis call. It
        makes use of the Twiss table to build up the response matrix.

        Args:
            concatenate_elements(bool): If True, merges error sources according
                to the :py:attr:`CONNECTED_ELEMENTS` setting in the
                config file.
        Returns:
            A pandas DataFrame containing the error response matrix.
        """

        beams = self.summ_table.keys()
        response_df = pd.DataFrame(0.0,
                                   index=self._get_response_matrix_index(),
                                   columns=self._get_error_index())

        print('Computing error matrix..')
        for col in list(response_df.columns):
            response_df[col] = self._error_column_to_pert(col,
                                                          response_df.index)

        # Deal with connected error sources
        if concatenate_elements:
            for new_name, element_li in config.CONNECTED_ELEMENTS.items():
                name_mapping = {element_li[0] : new_name}
                response_df = response_df.rename(columns=name_mapping, level=1)
                for element in element_li[1:]:
                    try:
                        response_df.loc[:, (slice(None), new_name, slice(None))] += response_df.loc[:,
                            (slice(None), element, slice(None))].to_numpy()
                        response_df = response_df.drop(columns=element, level=1)
                    except KeyError:
                        # Elements included in config.CONNECTED_ELEMENTS could
                        # be outside of the range to be considered, or
                        # misspelt. In both cases, we raise no exception and
                        # continue.
                        continue

        # We use the convention that a BPM error 'DX' or 'DY' achieves an orbit
        # offset at its position and nowhere else. In effect, the orbit for a
        # given plane at a given BPM is given as
        #
        #   orbit = orbit_wrt_reference - D[XY].
        #
        # This allows for corrections to take misalignment into account.
        for col in response_df.loc[:, 'BPM']:
            bpm_name, shift_dim = col

            for beam in config.SEQUENCE_NAMES:
                try:
                    response_df.at[(beam, shift_dim[-1], bpm_name),
                                   ('BPM', bpm_name, shift_dim)] -= 1.0
                except KeyError:
                    # Only happens if BPM not part of given beam..
                    continue


        return response_df

    def build_matrices(self,
                       section=None,
                       keep_all_by_default=True,
                       keyword_li=None,
                       pattern_li=None,
                       concatenate_elements=True):
        """ Initializes all structures for subsequent analysis.

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
        self.twiss_table = self.slice_twiss_table(
            section=section,
            keep_all_by_default=keep_all_by_default,
            keyword_li=keyword_li,
            pattern_li=pattern_li
        )
        self.corrector_response_matrix = self._compute_corrector_response_matrix()
        self.error_response_matrix = self._compute_error_response_matrix(
            concatenate_elements=concatenate_elements)
        self.error_table = pd.Series(0.0,
                                     index=self.error_response_matrix.columns)

    def slice_twiss_table(self,
                          section=None,
                          keep_all_by_default=True,
                          keyword_li=None,
                          pattern_li=None):
        """ Returns a sliced Twiss table.

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
                and :py:data:`keyword_li` are used to exclude elements.
                Conversely, if False, only elements capable of inducing
                orbit perturbation and BPMs are kept, where instead
                :py:data:`keyword_li` and :py:data:`pattern_li` are used
                to include additional elements beyond the minimal ones.
            keyword_li(list): A list of keywords to match the 'KEYWORD' column
                in the Twiss table. Allows for exclusion or inclusion depending
                on the value of :py:data:`keep_all_by_default`.
            pattern_li(list): A list of patterns to match the element names
                in the Twiss table. Allows for exclusion or inclusion depending
                on the value of :py:data:`keep_all_by_default`.
        Returns:
            A slicing of the orginal Twiss table stored in parent_twiss_table.
        """
        tt = self.twiss_table
        if section is not None:
            row_li = []
            for beam, seq_def in section.items():
                start, end = (beam, seq_def[0]), (beam, seq_def[1])
                idx_start = tt.index.get_loc(start)
                idx_end = tt.index.get_loc(end)
                if idx_start <= idx_end:
                    row_li += tt.iloc[idx_start:idx_end+1,:].index.to_list()
                # Only possible if the section includes the area
                # where the Twiss table 'wraps'.
                else:
                    temp = tt.iloc[idx_start:, :].loc[beam].index.to_list()
                    temp += tt.iloc[:idx_end+1, :].loc[beam].index.to_list()
                    row_li += [(beam, x) for x in temp]
            tt = tt.loc[row_li, :]


        to_keep = pd.Series(keep_all_by_default, index=tt.index)
        if not keep_all_by_default:
            to_keep = to_keep | (tt['K0L'].abs() > 0.0)
            to_keep = to_keep | (tt['K1L'].abs() > 0.0)
            to_keep = to_keep | (tt['KEYWORD'] == 'HKICKER')
            to_keep = to_keep | (tt['KEYWORD'] == 'VKICKER')
            to_keep = to_keep | (tt['KEYWORD'] == 'MONITOR')

        if keyword_li is not None:
            for keyword in keyword_li:
                if keep_all_by_default:
                    # Remove elements that match.
                    to_keep = to_keep & ~(tt['KEYWORD'] == keyword)
                else:
                    # Add elements that match.
                    to_keep = to_keep | (tt['KEYWORD'] == keyword)

        if pattern_li is not None:
            matched_elements = set()
            for pattern in pattern_li:
                matched_elements |= set(
                    [e for e in tt.index.get_level_values(1)
                     if re.search(pattern, e)]
                )
            to_keep.loc[:, list(matched_elements)] = not keep_all_by_default

        return tt[to_keep]

    def _error_column_to_pert(self, col, index):
        """ Computes the closed orbit perturbation from a unit error.

        Args:
            col(tup): Tuple corresponding to a column in the error response matrix.
            index(pandas.Index): Index to be used for the output.
        Return:
            Series containing the perturbation.
        """

        pert = pd.Series(0.0, index=index)

        __, e, err_type = col
        for beam in config.SEQUENCE_NAMES:
            try:
                keyword = self.twiss_table.loc[(beam, e), 'KEYWORD']
            except KeyError:
                # Only happens if element does not exist for given beam
                continue
            for plane in 'XY':
                if keyword == 'MARKER':
                    i = 1
                    e_slice = e+f'..{i}'
                    n = pert.loc[beam, plane].shape[0]
                    x, xp = np.zeros(n), np.zeros(n)
                    while e_slice in self.twiss_table.loc[beam].index:
                        x_incr, xp_incr = self._error_to_perturbation(
                            e_slice, err_type, 1.0, plane, beam)
                        x += x_incr
                        xp += xp_incr
                        i += 1
                        e_slice = e+f'..{i}'
                else:
                    x, xp = self._error_to_perturbation(e, err_type, 1.0,
                                                        plane, beam)
                pert.loc[beam, plane, :] = x
                pert.loc[beam, 'P' + plane, :] = xp
        return pert

    def _corrector_to_pert(self, corr, index):
        """ Computes the closed orbit perturbation from a 1 rad kick.

        Args:
          corr(str): Orbit corrector name.
          index(pandas.Index): Index to be used for the output.
        Return:
          Series containing the perturbation.
        """
        pert = pd.Series(0.0, index=index)

        for beam in config.SEQUENCE_NAMES:
            try:
                if self.twiss_table.loc[(beam, corr), 'KEYWORD'] == 'HKICKER':
                    plane = 'X'
                else:
                    plane = 'Y'
            except KeyError:
                continue

            # The BV_FLAG indicates the orientation of the beam,
            # In HL-LHC, BV_FLAG = 1 for Beam 1, -1 for Beam 2
            kick = 1.0 * self.summ_table[beam]['BV_FLAG']
            x, xp = self._kick_to_perturbation(corr, kick, plane, beam)

            pert.loc[beam, plane, :] = x
            pert.loc[beam, 'P' + plane, :] = xp
        return pert

    def _kick_to_perturbation(self, element, kick, plane, beam,
                              return_as_numpy=True):
        """ Computes the perturbation of closed orbit given a kick.

        Args:
            element(str): Name of element being the source of the kick.
            kick(float): Value of the orbit kick.
            plane(str): Transverse plane of interest, either 'X' or 'Y'.
            beam(str): Name of the beam.
            return_as_numpy(bool): True if returned as numpy arrays
                rather than pandas Series.
        Returns:
            Induced perturbation in position and angle for the given plane.
        """

        # Use the correct twiss table
        tt = self.twiss_table.loc[beam]
        st = self.summ_table[beam]

        # Define the quantities to be used
        beta = tt['BET'+plane]
        alfa = tt['ALF'+plane]
        mu = tt['MU'+plane]
        if plane == 'X':
            Q = st['Q1']
        else:
            Q = st['Q2']

        bet0 = beta[element]
        mu0 = mu[element]

        # Make sure to scale mu
        mu  = mu * 2 * np.pi
        mu0 = mu0 * 2 * np.pi

        # Nice vectorization of the computation
        mu_arr = mu - mu0
        mu_arr += (mu_arr < 0) * Q * np.pi * 2
        x = (kick * np.sqrt(bet0 * beta) * np.cos(
            np.pi * Q - mu_arr)) / (2 * np.sin(np.pi * Q))
        px = kick * np.sqrt(bet0) * np.sin(np.pi*Q - mu_arr) / (
            2*np.sin(np.pi*Q)*np.sqrt(beta)) - alfa*x/beta

        if return_as_numpy:
            return x.to_numpy(), px.to_numpy()
        else:
            return x, px

    def _error_to_kick(self, elmt, error_type, error, plane, beam):
        """ Computes angular kick caused by a machine error.

        Args:
            elmt(str): Name of element with a machine error.
            error_type(str): Type of the error, e.g. 'DX', 'DPSY'.
            error(float): Magnitude of the error.
            plane(str): Transverse plane of interest, either 'X' or 'Y'.
            beam(str): Name of the beam.
        Returns:
            Value of the resulting orbit kick.
        """

        kick = 0
        try:
            r = self.twiss_table.loc[beam, elmt]
        except:
            raise KeyError(f'({beam}, {elmt}) not in Twiss table!')
        if error_type == 'DK0':
            if plane == 'X':
                kick += -error * np.cos(r['TILT'])
            else:
                kick += -error * np.sin(r['TILT'])
        elif error_type == 'DKR0':
            if plane == 'X':
                kick += -error * r['K0L'] * np.cos(r['TILT'])
            else:
                kick += -error * r['K0L'] * np.sin(r['TILT'])
        elif error_type == 'DS':
            raise NotImplementedError(
                'Longitudinal shift not yet implemented.')
        elif error_type == 'DPSI':
            if plane == 'X':
                kick += error * r['K0L'] * np.sin(r['TILT'])
                kick += -2 * error * np.cos(2 * r['TILT']) * r['K1L'] * r['Y']
                kick += 2 * error * np.sin(2 * r['TILT']) * r['K1L'] * r['X']
            else:
                kick += -error * r['K0L'] * np.cos(r['TILT'])
                kick += -2 * error * np.cos(2 * r['TILT']) * r['K1L'] * r['X']
                kick += -2 * error * np.sin(2 * r['TILT']) * r['K1L'] * r['Y']
        elif error_type == 'DK1':
            if plane == 'X':
                kick += -error * r['X'] * np.cos(2 * r['TILT'])
                kick += error * r['Y'] * np.sin(2 * r['TILT'])
            else:
                kick += error * r['Y'] * np.cos(2 * r['TILT'])
                kick += -error * r['X'] * np.sin(2 * r['TILT'])
        elif error_type == 'DKR1':
            if plane == 'X':
                kick += -error * r['K1L'] * r['X'] * np.cos(2 * r['TILT'])
                kick += error * r['K1L'] * r['Y'] * np.sin(2 * r['TILT'])
            else:
                kick += error * r['K1L'] * r['Y'] * np.cos(2 * r['TILT'])
                kick += -error * r['K1L'] * r['X'] * np.sin(2 * r['TILT'])
        elif error_type == 'DX':
            if plane == 'X':
                kick += error * r['K1L'] * np.cos(2 * r['TILT'])
            else:
                kick += error * r['K1L'] * np.sin(2 * r['TILT'])
        elif error_type == 'DY':
            if plane == 'X':
                kick += -error * r['K1L'] * np.sin(2 * r['TILT'])
            else:
                kick += -error * r['K1L'] * np.cos(2 * r['TILT'])
        else:
            raise ValueError('Invalid error_type provided.')
        return kick

    def _error_to_perturbation(self, element, error_type, error_value,
                               plane, beam, return_as_numpy=True):
        """ Computes the error caused by a source of error perturbation.

        Args:
            element(str): Name of the element inducing the perturbation.
            error_type(str): Name of error type, e.g. 'DX', 'DPSI'.
            error_value(float): Magnitude of the given error.
            plane(str): Transverse plane of interest, either 'X' or 'Y'.
            beam(str): Name of the beam.
            return_as_numpy(bool): True if returned as numpy arrays
                rather than pandas Series.
        Returns:
            Induced perturbation in position and angle for the given
            :py:data:`plane`.
        """
        kick = self._error_to_kick(
            element, error_type, error_value, plane, beam)
        return self._kick_to_perturbation(
            element, kick, plane, beam, return_as_numpy=return_as_numpy)
