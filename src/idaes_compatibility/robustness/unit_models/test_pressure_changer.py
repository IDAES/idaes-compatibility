#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
"""
This module contains the code for convergence testing of the
PressureChanger model
"""
import pytest
import os
import json
from collections import OrderedDict

import idaes.core.util.convergence.convergence_base as cb

import pyomo.environ as pe
from pyomo.common.fileutils import this_file_dir
from pyomo.common.unittest import assertStructuredAlmostEqual

from idaes.core import FlowsheetBlock
from idaes.models.unit_models.pressure_changer import (
    PressureChanger,
    ThermodynamicAssumption,
)
from idaes.core.solvers import get_solver

# Import property package for testing
from idaes.models.properties import iapws95 as pp


currdir = this_file_dir()
fname = os.path.join(currdir, "isothermal_pressure_changer.json")


@cb.register_convergence_class("IsothermalPressureChanger")
class IsothermalPressureChangerConvergenceEvaluation(cb.ConvergenceEvaluation):
    def get_specification(self):
        """
        Returns the convergence evaluation specification for the
        isothermal PressureChanger unit model

        Returns
        -------
           ConvergenceEvaluationSpecification
        """
        s = cb.ConvergenceEvaluationSpecification()

        s.add_sampled_input(
            name="Inlet_Flowrate",
            pyomo_path="fs.pc.control_volume.properties_in[0].flow_mass",
            lower=1,
            upper=1e6,
            distribution="uniform",
        )

        s.add_sampled_input(
            name="Inlet_Pressure",
            pyomo_path="fs.pc.control_volume.properties_in[0].pressure",
            lower=10,
            upper=2e8,
            distribution="uniform",
        )

        s.add_sampled_input(
            name="Inlet_Enthalpy",
            pyomo_path="fs.pc.control_volume.properties_in[0].enth_mass",
            lower=2e4,
            upper=4.4e6,
            distribution="uniform",
        )

        # TODO: Add deltaP as an input?
        return s

    def get_initialized_model(self):
        """
        Returns an initialized model for the PressureChanger unit model
        convergence evaluation

        Returns
        -------
           Pyomo model : returns a pyomo model of the PressureChanger unit
        """
        m = pe.ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.props = pp.Iapws95ParameterBlock(amount_basis=pp.AmountBasis.MASS)

        m.fs.pc = PressureChanger(
            property_package=m.fs.props,
            thermodynamic_assumption=ThermodynamicAssumption.isothermal,
        )

        m.fs.pc.deltaP.fix(-1e3)
        m.fs.pc.inlet[:].flow_mass.fix(27.5e3)
        m.fs.pc.inlet[:].enth_mass.fix(4000)
        m.fs.pc.inlet[:].pressure.fix(2e6)

        init_state = {"flow_mass": 27.5e3, "pressure": 2e6, "enth_mass": 4000}

        m.fs.pc.initialize(state_args=init_state)

        # Create a solver for initialization
        opt = self.get_solver()
        opt.solve(m)

        # return the initialized model
        return m


def test_isothermal_pressure_changer_robustness():
    ceval = IsothermalPressureChangerConvergenceEvaluation()

    solves, iters, restoration, regularization = ceval.compare_to_baseline(fname)

    assert solves == []
    assert iters == []
    assert restoration == []
    assert regularization == []
