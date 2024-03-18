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

from pyomo.environ import ConcreteModel
from pyomo.common.fileutils import this_file_dir

from idaes.core import FlowsheetBlock
from idaes.models.unit_models.pressure_changer import (
    PressureChanger,
    ThermodynamicAssumption,
)
# Import property package for testing
from idaes.models.properties import iapws95 as pp

from idaes.core.initialization import (
    BlockTriangularizationInitializer,
    InitializationStatus,
)

from idaes.core.surrogate.pysmo.sampling import LatinHypercubeSampling
from idaes.core.util.parameter_sweep import ParameterSweepSpecification
from idaes.core.util.model_diagnostics import IpoptConvergenceAnalysis


currdir = this_file_dir()
fname = os.path.join(currdir, "isothermal_pressure_changer.json")


def build_model():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.props = pp.Iapws95ParameterBlock(amount_basis=pp.AmountBasis.MASS)

    m.fs.unit = PressureChanger(
        property_package=m.fs.props,
        thermodynamic_assumption=ThermodynamicAssumption.isothermal,
    )

    m.fs.unit.deltaP.fix(-1e3)
    m.fs.unit.inlet[:].flow_mass.fix(27.5e3)
    m.fs.unit.inlet[:].enth_mass.fix(4000)
    m.fs.unit.inlet[:].pressure.fix(2e6)

    # init_state = {"flow_mass": 27.5e3, "pressure": 2e6, "enth_mass": 4000}

    initializer = BlockTriangularizationInitializer(constraint_tolerance=2e-5)
    initializer.initialize(m.fs.unit)

    assert initializer.summary[m.fs.unit]["status"] == InitializationStatus.Ok

    # return the initialized model
    return m


def generate_baseline():
    model = build_model()

    spec = ParameterSweepSpecification()
    spec.add_sampled_input("fs.unit.inlet.flow_mass[0]", lower=1, upper=1e6)
    spec.add_sampled_input("fs.unit.inlet.pressure[0]", lower=10, upper=2e8)
    spec.add_sampled_input("fs.unit.inlet.enth_mass[0]", lower=2e4, upper=4.4e6)
    spec.add_sampled_input("fs.unit.deltaP[0]", lower=-5e5, upper=5e5)
    spec.set_sampling_method(LatinHypercubeSampling)
    spec.set_sample_size(200)

    spec.generate_samples()

    ca = IpoptConvergenceAnalysis(
        model,
        input_specification=spec,
    )

    ca.run_convergence_analysis()

    ca.to_json_file(fname)

    ca.report_convergence_summary()


def test_isothermal_pressure_changer_robustness():
    model = build_model()
    ca = IpoptConvergenceAnalysis(model)

    ca.assert_baseline_comparison(fname)


if __name__ == "__main__":
    generate_baseline()
