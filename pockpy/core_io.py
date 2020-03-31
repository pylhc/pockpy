""" Module for MAD-X interaction and parsing of TFS tables to Twiss tables."""

import re
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import cpymad.madx
import tfs # found here: https://pypi.org/project/tfs-pandas/

import pockpy.config as config

def _format_cpymad_table(table, return_summ_table=False,
                         return_minimal_columns=False,
                         make_elements_unique=False):
    """ Formats a Twiss table stored inside cpymad to a DataFrame.

    Args:
        table(cpymad.Madx.table): Table inside cpymad.
        return_summ_table(bool): True if the SUMM table is to be formatted and
            returned. Only applicable to Twiss tables.
        return_minimal_columns(bool): True if only the columns used for POCKPy
            (as defined in :py:attr:`config.MINIMUM_TWISS_COLUMNS`)
            are to be returned for the table. Only useful for Twiss tables.
        enforce_element_uniqueness(bool): If True, will append '~i' to all
            element names in the index where i is the number of previous
            occurences of the element in the index when it is traversed in
            order. This results in unique elements being appended by '~0'.
            Useful when parsing Aperture tables where element names get
            degenerated.
    Returns:
        A table in a DataFrame, and a dict containing the SUMM table
        if format_summ is set to True.
    """

    # Create the DataFrame to return.
    df = pd.DataFrame()
    for col in table.keys():
        df[col] = getattr(table, col)

    # Capitalizes everything and set the index to be element names.
    df.columns = [x.upper() for x in df.columns]
    df = df.applymap(lambda x : x.upper() if type(x)==str else x)
    df = df.set_index('NAME', drop=True)

    # Removes a ':[0-9]' at the end of element names. Serves a purpose if
    # elements are properly named when sliced, as they are not so at this
    # point :d endings are removed for the time being.
    # On 27/09/19 drifts were appended by ':0' and every other element by
    # ':1'. Sliced elements have the form 'ELMT_NAME..{SLICE_NBR}:1'
    # NOTE: Will cause issues if MAD-X naming scheme for sliced elements is
    # updated.
    df.index = [re.sub(r':\d$', '', x) for x in df.index]

    if make_elements_unique:
        seen = {}
        new_index = []
        for e in df.index:
            if e not in seen:
                seen[e] = 0
            new_index.append(f'{e}~{seen[e]}')
            seen[e] += 1
        df.index = new_index

    if return_minimal_columns:
        df = df.loc[:, config.MINIMUM_TWISS_COLUMNS]

    if return_summ_table:
        d = {key.upper() : val for key, val in table.summary.items()}
        return df, d
    else:
        return df

def madx_to_tables(filename, makethin_option_li=None):
    """ Takes a .madx file for a machine and returns a Twiss table
    to be used for the for Solver.

    Args:
        filename(str): Path to a .madx file defining the machine.
    Returns:
        A Twiss table DataFrame and a Summ table expressed as a dict,
        both containing the requisite data for the sequences defined in
        :py:attr:`config.SEQUENCE_NAMES`.
    """

    madx = MadxWrapper()
    madx.call(filename)
    madx.remove_all_sextupole_fields()

    twiss_table_dict, summ_table_dict = OrderedDict(), OrderedDict()
    for sequence in config.SEQUENCE_NAMES:
        if makethin_option_li is not None:
            madx.makethin(sequence, makethin_option_li)
        madx.use_sequence(sequence)
        twiss_table_dict[sequence], summ_table_dict[sequence] = madx.twiss(
            return_minimal_columns=True)
    madx.quit()

    # Stack all individual Twiss tables
    twiss_table = pd.concat(twiss_table_dict)
    twiss_table.index.names = ['BEAM', 'ELEMENT']

    return twiss_table, summ_table_dict

def tfs_to_tables(filename_dict):
    """ Takes .tfs files and returns a Twiss table for Solver.

    Args:
        filename_dict(dict): A dict mapping from beam name to corresponding
            .tfs table.
    Returns:
        A DataFrame containing the Twiss table for input to Solver.
    """

    twiss_table_dict, summ_table_dict = OrderedDict(), OrderedDict()
    for beam, path in filename_dict.items():
        twiss_table_dict[beam], summ_table_dict[beam] = load_tfs_table(path)

    # Stack all individual Twiss tables
    twiss_table = pd.concat(twiss_table_dict)
    twiss_table.index.names = ['BEAM', 'ELEMENT']

    return twiss_table, summ_table_dict

def load_tfs_table(filename):
    """ Takes a .tfs file and returns the Twiss and SUMM tables.

    Args:
        filename(str): Path to .tfs file from a Twiss call.
    Returns:
        A Twiss and SUMM table for the given machine and beam.
    """

    # Returns a special DataFrame with the SUMM table stored as individual,
    # capitalized attributes of the returned DataFrame.
    special_df = tfs.read_tfs(filename, index='NAME')

    # Coerce into stadard dictionary and DataFrame
    d = dict(special_df.headers)
    df = pd.DataFrame(special_df)
    return df, d

class MadxWrapper:
    """ Wrapper of the MAD-X wrapper :py:mod:`cpymad`.

    This class is dedicated to providing useful combinations of MAD-X commands
    to be used as part of orbit correction analysis. For any other use of
    MAD-X, instead make use of :py:mod:`cpymad` directly.

    **Example**::

        >>> madx = pockpy.MadxWrapper()
        >>> madx.input('some_machine.madx')
        >>> madx.use_sequence('B1')
        >>> twiss1 = madx.twiss(return_summ_table=False)
        >>> shift_error = {'DX' : 1e-6}
        >>> madx.add_misalignment(
        ...     pattern='MQ.12R5.B1',
        ...     errors=shift_error
        ... )
        >>> twiss2 = madx.twiss(return_summ_table=False)
        >>> perturbation = twiss2['X'] - twiss1['X']

    """

    def __init__(self):
        # TODO: Investigate how std_out works, seems to mute all instances of
        # cpymad that gets created after having been set once.
        self._m = cpymad.madx.Madx()
        self._range = '#S/#E'


    def call(self, filename):
        """ Executes the provided .madx file in MAD-X.

        Args:
            filename(str): A path to a .madx file
        """
        self._m.call(filename)

    def input(self, cmd):
        """ Executes a given command in MAD-X.

        Args:
            cmd(str): A MAD-X command on a single line.
        """
        self._m.input(cmd)

    def twiss(self, centre=True, return_summ_table=True,
              return_minimal_columns=False):
        """ Runs Twiss inside MAD-X for the active sequence.

        Args:
            centre(bool): True if Twiss parameters are to be evaluated
                at the middle of elements, otherwise they are evalutated
                at their entrance.
            return_summ_table(bool): True if the SUMM table is to be
                returned as a second output.
            return_minimal_columns(bool): True if only the columns used
                for POCKPy (as defined in
                :py:attr:`config.MINIMUM_TWISS_COLUMNS`) are to be returned for
                the Twiss table.
        Returns:
            The Twiss and SUMM table from the Twiss call if
            :py:data:`return_summ_table` is True, otherwise just the Twiss
            table.
        Raises:
            :py:exc:`RuntimeError`: If no Twiss table is produced.
        """

        self._m.input(f'SELECT, FLAG=TWISS, CLEAR;')
        self._m.input(f'SELECT, FLAG=TWISS, RANGE={self._range};')
        self._m.input(f'TWISS, CENTRE={centre};')
        try:
            tbl = self._m.table['twiss']
        except KeyError:
            raise RuntimeError(
                'Twiss call did not produce a Twiss table.')

        return _format_cpymad_table(tbl,
                                    return_summ_table=True,
                                    return_minimal_columns=return_minimal_columns,
                                    make_elements_unique=False)

    def makethin(self, sequence, option_li):
        """ Executes a MAKETHIN command in MAD-X.

        Args:
            sequence(str): Name of sequence to be made thin.
            option_li(list): List of dicts as per::

                    option_li = [
                        {
                            'class' : 'QUADRUPOLE',
                            'slice' : 10
                        },
                        {
                            'pattern' : '^MQXF.*',
                            'slice' : 20
                        },
                    ]

                Any given dict must contain a 'slice' entry, and at
                least one of 'class' and 'pattern'.

                The selections are performed sequentially with the options
                of the first dictionary in the list applied first and
                terminated by a MAKETHIN command.
        Raises:
            :py:exc:`KeyError`: If :py:data:`option_li` does not match
                expected format.
        """

        self.use_sequence(seq=sequence)
        self._m.input(f'SELECT, FLAG=MAKETHIN, CLEAR;')
        self._m.input(f'SELECT, FLAG=MAKETHIN, THICK=TRUE, SLICE=0;')
        for option in option_li:
            command = f'SELECT, FLAG=MAKETHIN, THICK=FALSE, '
            if 'class' in option:
                command += f'CLASS={option["class"]}, '
            elif 'pattern' in option:
                command += f'PATTERN={option["pattern"]}, '
            else:
                raise KeyError(
                    "Either a 'class' or 'pattern' entry has to be included.")
            command += f'SLICE={option["slice"]};'
            self._m.input(command)
        self._m.input(f'MAKETHIN, SEQUENCE={sequence}, STYLE=TEAPOT;')


    def use_sequence(self, seq, start='#S', end='#E'):
        """ Sets the active sequence inside MAD-X.

        Args:
            seq(str): Sequence to activate.
            start(str): Element from which the active sequence starts.
                Defaults to the first element of the sequence.
            end(str): Element defining the end of the active sequence.
                Defaults to the last element of the sequence.
        """

        self._range = f'{start}/{end}'
        self._m.input(f'USE, SEQUENCE={seq}, RANGE={self._range};')

    def quit(self):
        """ Terminates the current MAD-X instance. """
        self._m.input('QUIT;')

    def verbose(self, switch=True):
        """ Makes the output less or more verbose inside the MAD-X instance.

        .. note:: Does not always result in any notable difference.

        Args:
            switch(bool): True for more verbose, False for less.
        """

        self._m.verbose(switch=switch)

    def add_misalignment(self, pattern, errors, add_errors=True):
        """ Adds misalignments to matched elements in the active sequence.

        Args:
            pattern(str): Regular expression to match elements in
                the active sequence which are to have added misalignments.
            errors(dict): Dictionary mapping from MAD-X errors to floats.
            add_errors(bool): True if errors introduced by the call are to
                increment already existing ones, otherwise they
                are overwritten.
        """

        self._m.input(f'EOPTION, ADD={add_errors};')
        self._m.input('SELECT, FLAG=ERROR, CLEAR;') # good practice
        self._m.input(f'SELECT, FLAG=ERROR, PATTERN={pattern};')

        # TODO: Find cleaner process.
        s = f'EALIGN'
        for x, y in errors.items():
            s += f', {x}={y}'
        s += ';'
        self._m.input(s)

    def set_corrector_strength(self, corrector_dict, increment=False):
        """ Adds strength to specified correctors in the active sequence.

        **NOTE**: Corrector strength is given in radians.

        Args:
            corrector_dict(dict): A dict-like (e.g. :py:class:`pandas.Series`)
                mapping from corrector name to corrector strength in radians.
            increment(bool): If True, increments the current corrector strength
                in the machine.
        """
        for corrector, strength in corrector_dict.items():
            if increment:
                self._m.elements[corrector].kick += strength
            else:
                self._m.elements[corrector].kick = strength

    def get_corrector_strength(self, index):
        """ Returns the corrector strength in the machine.

        **NOTE**: Corrector strength is given in radians.

        Args:
            index(list): List or pandas Index containing corrector
                names of interest.
        Returns:
            A pandas Series containing the corrector strength usage of each
            corrector specified by :py:data:`index`.
        """
        output = pd.Series(0.0, index=index)
        for corrector in index:
            output[corrector] = self._m.elements[corrector].kick
        return output

    def remove_errors(self, pattern='.*'):
        """ Removes all misalignments and field errors from matched elements.

        Args:
            pattern: Regular expression specifying all elements in the
                active sequence which are to have their misalignments and field
                errors removed. Defaults to maching all elements.
        """
        self._m.input('EOPTION, ADD=False;')
        self._m.input('SELECT, FLAG=ERROR, CLEAR;') # good practice
        self._m.input(f'SELECT, FLAG=ERROR, PATTERN={element_pattern};')
        self._m.input('EALIGN, DX=0, DY=0, DS=0, DPSI=0;')
        self._m.input('EFCOMP, RADIUS=1, ORDER=0, DKN={0, 0}, DKNR={0, 0}')

    def add_field_error(self, pattern, errors, add_errors=True):
        """ Adds field errors to elements in the active sequence.

        For more information about the impact of adding field errors and
        context, see the MAD-X documentation.

        Args:
            pattern(str): Regular expression defining elements in the active
                sequence which are to have field errors added to them.
            errors(dict): Dictionary of the form::

                    errors = {
                        'DKN19' : 3.0,
                        'DKS1' : 2.0,
                        'DKNR1' : 4.0,
                    }

                where the keys are using the MAD-X standard for ``EFCOMP``
                in the sense that 'DKN18' corrsponds to 'dkn(18)', and the
                values are the corresponding magnitude of the errors.
                The :py:mod:`pockpy.solver` notation where 'DK' and 'DKR' are
                interpreted as 'DKN' and 'DKNR' respectively is supported.
        Raises:
            :py:exc:`ValueError`: If :py:data:`errors` does not match expected
                format.
        """

        self._m.input(f'EOPTION, ADD={add_errors};')
        self._m.input('SELECT, FLAG=ERROR, CLEAR;') # good practice
        self._m.input(f'SELECT, FLAG=ERROR, PATTERN={pattern};')

        # Define the field error structure
        field_error_dict = {
            k : [0.0] * 21 for k in ['DKN', 'DKS', 'DKNR', 'DKSR']
        }

        order = None

        for err, val in errors.items():
            i = None
            key = None
            try:
                i = int(err[-2:])
                key = err[:-2]
            except:
                i = int(err[-1])
                key = err[:-1]
            try:
                field_error_dict[key][i] = val
            except:
                # Support for the same notation as in Solver, where it is
                # assumed that 'DK' and 'DKR' refer to the normal components.
                if key == 'DK':
                    field_error_dict['DKN'][i] = val
                elif key == 'DKR':
                    field_error_dict['DKNR'][i] = val
                else:
                    raise ValueError(
                        'Field error name does not match expected format.'
                    )

            if 'r' in key or 'R' in key:
                if order is None:
                    order = i * (-1 if 'S' in key else 1)
                else:
                    raise ValueError(
                        'Only one relative field error can be set per call.'
                    )

        if order is None:
            order = 0

        # Convert to string values for subsequent string manipulation..
        for key in field_error_dict:
            field_error_dict[key] = [str(x) for x in field_error_dict[key]]

        # TODO: Maybe find cleaner process?
        s = f'EFCOMP, ORDER={order}, RADIUS=1'
        for key, val in field_error_dict.items():
            s += f', {key}=' + '{' + ', '.join(val) + '}'
        s += ';'
        self._m.input(s)

    def add_solver_element_errors(self, errors, add_errors=True):
        """ Adds errors using an error dictionary compatible with Solver.

        Args:
            errors(dict): Dict of the form::

                    errors = {
                        pattern : {
                            error_type : val
                        }
                    }

                where pattern is a regular expression matching elements
                which are to have their errors updated and error_type is a valid
                type of error as used in Solver, i.e. one of::

                    ['DX', 'DY', 'DPSI', 'DS', 'DK0', 'DKR0', 'DK1', 'DKR1']

            add_errors(bool): True if all impacted errors are to be
                incremented, otherwise their previous errors are overwritten.
        """
        misalignment_keys = ['DX', 'DY', 'DPSI', 'DS']

        for pattern, error_dict in errors.items():
            misalignment_errors = {key : val for key, val in error_dict.items()
                                   if key in misalignment_keys}
            field_errors = {key : val for key, val in error_dict.items()
                            if key not in misalignment_keys}

            if len(misalignment_errors) > 0:
                self.add_misalignment(pattern, misalignment_errors,
                                      add_errors=add_errors)
            if len(field_errors) > 0:
                self.add_field_error(pattern, field_errors,
                                      add_errors=add_errors)

    def remove_all_sextupole_fields(self):
        """ Sets the strength of all sextupoles to zero. """
        for elt in self._m.elements:
            try:
                getattr(self._m.elements, elt).k2 = 0
                getattr(self._m.elements, elt).k2s = 0
            # KeyError gets thrown if the k2/k2s attributes aren't found
            # AttributeError (apparently) gets thrown under more rare
            # circumstances arising from cycling the lattice.
            except (AttributeError, KeyError) as e:
                pass

    def _aperture_call(
            self, co=0, deltap=0, optics_type='round',
            aperture_offset=config.APERTURE_OFFSET_DEFAULTS['LHCB1']):
        """ Performs an aperture computation in MAD-X for the active sequence.

        Runs an ``APERTURE`` call in MAD-X. For in-depth information, see the
        official MAD-X documentation.

        .. warning:: Assumes an (HL-)LHC machine.

        Args:
            co(float): Corresponds to the COR argument in ``APERTURE``.
            deltap(float): Corrsponds to the DP argument in ``APERTURE``.
            optics_type(str): Type of optics. Should be one of
                'round', 'flat' or 'injection'.
            aperture_offset(str): Path to a valid OFFSETELEM as defined in
                MAD-X. Note that OFFSETELEM is beam dependent.
        Returns:
            Resulting aperture table as a DataFrame.
        Raises:
            :py:exc:`RuntimeError`: If the aperture call doesn ot produce an
                aperture table.
        """
        if optics_type == 'round' or optics_type == 'flat':
            DParcx = 0.1
            DParcy = 0.1
            apbbeat = 1.1
        elif optics_type == 'injection':
            DParcx = 0.14
            DParcy = 0.14
            apbbeat = 1.05

        self._m.input(
            (f'aperture,range={self._range}, offsetelem={aperture_offset}, '
             f'cor={co}, dp={deltap}, interval=1, '
             f'halo={{6,6.0001,6,6}}, bbeat={apbbeat}, '
             f'dparx={DParcx}, dpary={DParcy};')
        )

        try:
            tbl = self._m.table['aperture']
        except KeyError:
            raise RuntimeError('Aperture call did not produce a Twiss table.')

        return _format_cpymad_table(tbl, make_elements_unique=True)

    def compute_aperture_table(self, aperture_offset, co=0, run_thrice=False,
                               optics_type='round'):
        """ Performs an aperture computation with pre-defined parameters.

        .. warning:: Assumes an (HL-)LHC machine.

        Runs an ``APERTURE`` call in MAD-X with some pre-defined parameters.
        For in-depth information, see the official MAD-X documentation.

        Args:
            aperture_offset(str): Path to a valid element OFFSETELEM as
                defined in MAD-X.
            co(float): Closed orbit uncertainty.
            run_thrice(float): If False runs a single ``APERTURE`` call to
                compute the aperture, otherwise runs the call for three
                different settings and returns the worst aperture among them
                for each point. The three scenarios are

                1. DeltaP = dPmax with a bucket edge of 0.
                2. DeltaP = -dPmax with a bucket edge of 0
                3. DeltaP = 0 with a bucket edge of dPmax.

            optics_type(str): Type of optics. Should be one of 'round',
                'flat' or 'injection'.
        Returns:
            A DataFrame with the available aperture.
        """
        if optics_type == 'round' or optics_type == 'flat':
            dPmax = 2e-4
        elif optics_type == 'injection':
            dPmax = 8.6e-4

        if run_thrice:
            # Aperture table columns that differ for the three cases
            aperture_cols = ['N1', 'N1X_M', 'N1Y_M']

            # Run Twiss and the aperture check three times for
            # momentum deviation 0, deltap and -deltap
            self._m.input(f'TWISS, DELTAP={dPmax};')
            tbl1 = self._aperture_call(
                co,
                0,
                optics_type,
                aperture_offset
            )
            self._m.input(f'TWISS, DELTAP=-{dPmax};')
            tbl2 = self._aperture_call(
                co,
                0,
                optics_type,
                aperture_offset
            )
            self._m.input('TWISS, DELTAP=0;')
            tbl3 = self._aperture_call(
                co,
                dPmax,
                optics_type,
                aperture_offset
            )

            # Use the worst case for the aperture
            tbl = tbl1
            for col in aperture_cols:
                tbl[col] = np.array(
                    [tbl1[col], tbl2[col], tbl3[col]]).min(axis=0)
        else:
            self._m.input('TWISS;')
            tbl = self._aperture_call(co, dPmax)

        return tbl

    def available_aperture_scan(self, aperture_offset, co_li, run_thrice=False,
                                optics_type='round', col_formatter=None):
        """ Performs multiple aperture computations for a list of closed orbit
        uncertainties, and pre-defined parameters.

        .. warning:: Assumes an (HL-)LHC machine.

        Runs an ``APERTURE`` call in MAD-X with some pre-defined parameters.
        For in-depth information, see the official MAD-X documentation.

        Args:
            aperture_offset(str): Path to a valid element OFFSETELEM as
                defined in MAD-X.
            co(list): Iterable, sorted in ascending order,
                of closed orbit uncertainties.
            run_thrice(float): If False runs a single ``APERTURE`` call to
                compute the aperture, otherwise runs the call for three
                different settings and returns the worst aperture among them
                for each point. The three scenarios are

                1. DeltaP = dPmax with a bucket edge of 0.
                2. DeltaP = -dPmax with a bucket edge of 0
                3. DeltaP = 0 with a bucket edge of dPmax.

            optics_type(str): Type of optics. Should be one of 'round',
                'flat' or 'injection'.
        Returns:
            A DataFrame with the available aperture given per provided
            closed orbit uncertainty in :py:data:`co_li`.
        """

        aperture_cols = ['N1', 'N1X_M', 'N1Y_M']
        aperture_dict = OrderedDict()
        for co in co_li:
            df = self.compute_aperture_table(
                aperture_offset=aperture_offset,
                co=co,
                run_thrice=run_thrice,
                optics_type=optics_type)
            aperture_dict[co] = df.loc[:, aperture_cols]

        df = pd.concat(aperture_dict, axis=1)

        # Swap order of levels in columns
        df = df.swaplevel(0, 1, axis=1)

        return df

