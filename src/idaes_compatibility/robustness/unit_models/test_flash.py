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
Flash model
"""
import pytest
import os
import json

from pyomo.environ import ConcreteModel, Constraint
from pyomo.common.fileutils import this_file_dir

from idaes.core import FlowsheetBlock
from idaes.models.unit_models.flash import Flash, EnergySplittingType
from idaes.models.properties import iapws95
from idaes.core.initialization import (
    BlockTriangularizationInitializer,
    InitializationStatus,
)

from idaes.core.surrogate.pysmo.sampling import LatinHypercubeSampling
from idaes.core.util.parameter_sweep import ParameterSweepSpecification
from idaes.core.util.model_diagnostics import IpoptConvergenceAnalysis


currdir = this_file_dir()
fname = os.path.join(currdir, "flash.json")


def build_model():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = iapws95.Iapws95ParameterBlock(
        phase_presentation=iapws95.PhaseType.LG
    )

    m.fs.unit = Flash(
        property_package=m.fs.properties,
        ideal_separation=False,
        energy_split_basis=EnergySplittingType.enthalpy_split,
    )

    m.fs.unit.inlet.flow_mol.fix(100)
    m.fs.unit.inlet.enth_mol.fix(24000)
    m.fs.unit.inlet.pressure.fix(101325)

    m.fs.unit.heat_duty.fix(0)
    m.fs.unit.deltaP.fix(0)

    initializer = BlockTriangularizationInitializer(constraint_tolerance=2e-5)
    initializer.initialize(m.fs.unit)

    assert initializer.summary[m.fs.unit]["status"] == InitializationStatus.Ok

    return m


def generate_baseline():
    model = build_model()

    spec = ParameterSweepSpecification()
    spec.add_sampled_input("fs.unit.inlet.flow_mol[0]", lower=0.1, upper=1e4)
    spec.add_sampled_input("fs.unit.inlet.pressure[0]", lower=5e4, upper=5e5)
    spec.add_sampled_input("fs.unit.inlet.enth_mol[0]", lower=10, upper=8e4)
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


def test_flash_robustness():
    model = build_model()
    ca = IpoptConvergenceAnalysis(model)

    ca.assert_baseline_comparison(fname)


if __name__ == "__main__":
    generate_baseline()
