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
CSTR model
"""
import pytest
import os
import json

from pyomo.environ import ConcreteModel
from pyomo.common.fileutils import this_file_dir

from idaes.core import FlowsheetBlock
from idaes.models.unit_models.cstr import CSTR
from idaes.models.properties.examples.saponification_thermo import (
    SaponificationParameterBlock,
)
from idaes.models.properties.examples.saponification_reactions import (
    SaponificationReactionParameterBlock,
)
from idaes.core.initialization import (
    BlockTriangularizationInitializer,
    InitializationStatus,
)

from idaes.core.surrogate.pysmo.sampling import LatinHypercubeSampling
from idaes.core.util.parameter_sweep import ParameterSweepSpecification
from idaes.core.util.model_diagnostics import IpoptConvergenceAnalysis


currdir = this_file_dir()
fname = os.path.join(currdir, "cstr.json")


def build_model():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = SaponificationParameterBlock()
    m.fs.reactions = SaponificationReactionParameterBlock(
        property_package=m.fs.properties
    )

    m.fs.unit = CSTR(
        property_package=m.fs.properties,
        reaction_package=m.fs.reactions,
        has_equilibrium_reactions=False,
        has_heat_transfer=True,
        has_heat_of_reaction=True,
        has_pressure_change=True,
    )

    m.fs.unit.inlet.flow_vol.fix(1.0e-03)
    m.fs.unit.inlet.conc_mol_comp[0, "H2O"].fix(55388.0)
    m.fs.unit.inlet.conc_mol_comp[0, "NaOH"].fix(100.0)
    m.fs.unit.inlet.conc_mol_comp[0, "EthylAcetate"].fix(100.0)
    m.fs.unit.inlet.conc_mol_comp[0, "SodiumAcetate"].fix(1e-8)
    m.fs.unit.inlet.conc_mol_comp[0, "Ethanol"].fix(1e-8)

    m.fs.unit.inlet.temperature.fix(303.15)
    m.fs.unit.inlet.pressure.fix(101325.0)

    m.fs.unit.volume.fix(1.5e-03)
    m.fs.unit.outlet.temperature.fix(303.15)
    m.fs.unit.deltaP.fix(0)

    initializer = BlockTriangularizationInitializer(constraint_tolerance=2e-5)
    initializer.initialize(m.fs.unit)

    assert initializer.summary[m.fs.unit]["status"] == InitializationStatus.Ok

    return m


def generate_baseline():
    model = build_model()

    spec = ParameterSweepSpecification()
    spec.add_sampled_input("fs.unit.inlet.flow_vol[0]", lower=1e-3, upper=1)
    spec.add_sampled_input("fs.unit.inlet.conc_mol_comp[0,NaOH]", lower=10, upper=200)
    spec.add_sampled_input("fs.unit.inlet.conc_mol_comp[0,EthylAcetate]", lower=10, upper=200)
    spec.add_sampled_input("fs.unit.inlet.conc_mol_comp[0,SodiumAcetate]", lower=1e-8, upper=10)
    spec.add_sampled_input("fs.unit.inlet.conc_mol_comp[0,Ethanol]", lower=1e-8, upper=10)
    spec.add_sampled_input("fs.unit.inlet.pressure[0]", lower=2e3, upper=9e5)
    spec.add_sampled_input("fs.unit.inlet.temperature[0]", lower=300, upper=320)
    spec.add_sampled_input("fs.unit.volume[0]", lower=1e-3, upper=1)
    spec.add_sampled_input("fs.unit.outlet.temperature[0]", lower=300, upper=320)
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


def test_cstr_robustness():
    model = build_model()
    ca = IpoptConvergenceAnalysis(model)

    ca.assert_baseline_comparison(fname)


if __name__ == "__main__":
    generate_baseline()
