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
HeatExchanger model
"""
import pytest
import os
import json

from pyomo.environ import ConcreteModel, Constraint
from pyomo.common.fileutils import this_file_dir

from idaes.core import FlowsheetBlock
from idaes.models.unit_models.heat_exchanger import (
    HeatExchanger,
    HeatExchangerFlowPattern,
    HX0DInitializer,
)
from idaes.models.properties.activity_coeff_models.BTX_activity_coeff_VLE import (
    BTXParameterBlock,
)
from idaes.core.initialization import (
    BlockTriangularizationInitializer,
    InitializationStatus,
)

from idaes.core.surrogate.pysmo.sampling import LatinHypercubeSampling
from idaes.core.util.parameter_sweep import ParameterSweepSpecification
from idaes.core.util.model_diagnostics import IpoptConvergenceAnalysis


currdir = this_file_dir()
fname = os.path.join(currdir, "hx0d.json")


def build_model():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = BTXParameterBlock(valid_phase="Liq")

    m.fs.unit = HeatExchanger(
        hot_side={"property_package": m.fs.properties},
        cold_side={"property_package": m.fs.properties},
        flow_pattern=HeatExchangerFlowPattern.cocurrent,
    )

    m.fs.unit.hot_side_inlet.flow_mol[0].fix(5)  # mol/s
    m.fs.unit.hot_side_inlet.temperature[0].fix(365)  # K
    m.fs.unit.hot_side_inlet.pressure[0].fix(101325)  # Pa
    m.fs.unit.hot_side_inlet.mole_frac_comp[0, "benzene"].fix(0.5)
    m.fs.unit.hot_side_inlet.mole_frac_comp[0, "toluene"].fix(0.5)

    m.fs.unit.cold_side_inlet.flow_mol[0].fix(1)  # mol/s
    m.fs.unit.cold_side_inlet.temperature[0].fix(300)  # K
    m.fs.unit.cold_side_inlet.pressure[0].fix(101325)  # Pa
    m.fs.unit.cold_side_inlet.mole_frac_comp[0, "benzene"].fix(0.5)
    m.fs.unit.cold_side_inlet.mole_frac_comp[0, "toluene"].fix(0.5)

    m.fs.unit.area.fix(1)
    m.fs.unit.overall_heat_transfer_coefficient.fix(100)

    initializer = HX0DInitializer()
    initializer.initialize(m.fs.unit)

    assert initializer.summary[m.fs.unit]["status"] == InitializationStatus.Ok

    return m


def generate_baseline():
    model = build_model()

    spec = ParameterSweepSpecification()
    spec.add_sampled_input("fs.unit.hot_side_inlet.flow_mol[0]", lower=1, upper=100)
    spec.add_sampled_input("fs.unit.hot_side_inlet.temperature[0]", lower=340, upper=380)
    spec.add_sampled_input("fs.unit.area", lower=0.1, upper=100)
    spec.add_sampled_input("fs.unit.overall_heat_transfer_coefficient[0]", lower=10, upper=1000)
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


def test_hx0d_robustness():
    model = build_model()
    ca = IpoptConvergenceAnalysis(model)

    ca.assert_baseline_comparison(fname)


if __name__ == "__main__":
    generate_baseline()
