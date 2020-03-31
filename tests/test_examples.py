import unittest
import re

import numpy as np

import pockpy

test_path = pockpy.config.POCKPY_PATH + '/tests'

class SolverTesting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Setup a valid Solver object. """
        tfs_filename_dict = {
            'LHCB1' : test_path + '/data/testb1.tfs',
            'LHCB2' : test_path + '/data/testb2.tfs'
        }

        # Local analysis for all elements around IP5, Q25->Q25.
        cls.solver = pockpy.Solver(tfs_filename_dict=tfs_filename_dict)
        slice_region = {
            'LHCB1' : ('MQ.25L5.B1', 'MQ.25R5.B1'),
            'LHCB2' : ('MQ.25L5.B2', 'MQ.25R5.B2')
        }
        cls.solver.build_matrices(slice_region)

    def test_twiss_shape(self):
        """ Assert that the Twiss table has the correct shape. """
        self.assertEqual(2870, self.solver.twiss_table.shape[0])
        self.assertEqual(20, self.solver.twiss_table.shape[1])

    def test_twiss_element_renaming(self):
        """ Assert that ':0' endings have been removed in the .tfs parsing. """
        element_names = list(self.solver.twiss_table.index.get_level_values(1))
        for e in element_names:
            self.assertFalse(bool(re.search(r':\d+', e)))

    def test_add_errors(self):
        """ Assert that the add_errors method works as anticipated. """
        # Add elements to quadrupoles within a given range of elements
        element_range = {
            'LHCB1' : ('MQ.20L5.B1', 'MQ.20R5.B1'),
            'LHCB2' : ('MQ.20L5.B2', 'MQ.20R5.B2')
        }
        error_dict = {
            'QUADRUPOLE' : {
                '^MQ.*' : {'DX' : 1.0}
            }
        }
        self.solver.add_errors(error_dict, element_range=element_range)

        # Assert that no quadrupoles outside the element range have any errors
        table = self.solver.error_table
        pattern = r'MQ\.*?2[1-5].*'
        outside_elements = [e for e in table.index.get_level_values(1)
                            if re.search(pattern, e)]
        inside_elements = [e for e in table.index.get_level_values(1)
                           if e not in outside_elements]
        np.testing.assert_array_equal(
            table.loc['QUADRUPOLE', outside_elements, 'DX'], 0.0)

        # Assert that errors were added
        np.testing.assert_array_equal(
            table.loc['QUADRUPOLE', inside_elements, 'DX'], 1.0)

        # Assert that errors can be removed
        self.solver.reset_errors()
        np.testing.assert_array_equal(table, 0.0)

    def test_convex_correction(self):
        """ Assert that convex_correction works as expected.

        Currently only superficially tested, more options are available that
        are not tested here.
        """
        # Reset all errors
        self.solver.reset_errors()

        # Add a reference error
        error_dict = {
            'QUADRUPOLE' : {
                'MQML.10L5.B1' : {
                    'DX' : 2e-3,
                }
            }
        }
        self.solver.add_errors(error_dict)

        # Correct with lax constraints
        cs_bound = self.solver.get_corrector_series()
        cs_bound[:] = 5
        orbit_bound = self.solver.get_bpm_reading_series()
        orbit_bound[:] = 2e-3

        cs, residual = self.solver.convex_correction(orbit_bound=orbit_bound,
                                                     cs_bound=cs_bound)

        # Assert that the closest horizontal corrector is used the most
        self.assertEqual(cs.abs().idxmax(), 'MCBH.10L5.B1')


        # Assert that the solution manages to correct surrounding orbit to zero
        self.assertLess(residual.abs().loc['LHCB1', 'X', 'MQ.25L5.B1'], 1e-10)
        self.assertLess(residual.abs().loc['LHCB1', 'X', 'MQ.25R5.B1'], 1e-10)

        # Assert that the constraints are fulfilled
        np.testing.assert_array_less(cs.abs(), cs_bound)

        bpm_li = self.solver.get_bpm_list()
        np.testing.assert_array_less(residual.loc[:, ['X', 'Y'], bpm_li].abs(),
                                     orbit_bound)

        # Assert that the problem is reported as unfeasible if 'MCBH.10L5.B1'
        # is too constrained
        cs_bound['MCBH.10L5.B1'] = 0.5
        try:
            __, __ = self.solver.convex_correction(orbit_bound=orbit_bound,
                                                   cs_bound=cs_bound)
            self.assertTrue(False, 'Optimizer did not raise a RuntimeError.')
        except RuntimeError:
            pass

        # Reset all errors
        self.solver.reset_errors()

    def test_linear_correction(self):
        """ Assert that linear_correction works as expected.

        Currently only superficially tested, more options are available that
        are not tested here.
        """

        # Reset all errors
        self.solver.reset_errors()

        # Add a reference error
        error_dict = {
            'QUADRUPOLE' : {
                '^MQML.10L5.B1' : {
                    'DX' : 2e-3,
                }
            }
        }
        self.solver.add_errors(error_dict)

        # Solve it with a pseudoinverse-based correction
        cs, residual = self.solver.linear_correction(n_x=50)

        # Assert that the solution manages to correct surrounding orbit to zero
        self.assertLess(residual.abs().loc['LHCB1', 'X', 'MQ.25L5.B1'], 1e-10)
        self.assertLess(residual.abs().loc['LHCB1', 'X', 'MQ.25R5.B1'], 1e-10)

        # Assert that the closest horizontal corrector is used the most
        self.assertEqual(cs.abs().idxmax(), 'MCBH.10L5.B1')

        # Reset all errors
        self.solver.reset_errors()

    def test_knob_implementation(self):
        """ Assert that knob_implementation works as expected.

        Currently only superficially tested, more options are available that
        are not tested here.
        """

        # Define corrector strength bound
        cs_bound = self.solver.get_corrector_series()
        cs_bound[:] = 4.0

        cs_df, orbit_df = self.solver.knob_implementation(cs_bound=cs_bound)

        # Assert that the corrector strength constraint is fulfilled
        np.testing.assert_array_less(cs_df.abs().sum(axis=1), cs_bound)

        # Assert that the orbits implementing the knobs match the specification
        knobs = pockpy.config.get_knob_definition()
        for knob_name, data in knobs.items():
            knob_def = data['orbit_spec']
            target = self.solver._knob_def_to_series(knob_def)
            x = orbit_df.loc[target.index, knob_name]
            np.testing.assert_allclose(x, target, rtol=1e-6, atol=1e-12)

if __name__ == '__main__':
    unittest.main()
