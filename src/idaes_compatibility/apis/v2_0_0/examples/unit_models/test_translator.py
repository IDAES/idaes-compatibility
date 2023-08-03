##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
import pytest

from pyomo.environ import ConcreteModel, Constraint, value, SolverFactory
from pyomo.opt import TerminationCondition, SolverStatus

from idaes.core.solvers import get_solver
from idaes.core import FlowsheetBlock
from idaes.models.unit_models import Translator
import idaes.logger as idaeslog
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.examples.BT_ideal import (
    configuration as configuration_FTPx,
)
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.core import Component
from pyomo.environ import units as pyunits
from idaes.core import LiquidPhase, VaporPhase
from idaes.models.properties.modular_properties.state_definitions import FcTP
from idaes.models.properties.modular_properties.eos.ideal import Ideal
from idaes.models.properties.modular_properties.phase_equil import SmoothVLE
from idaes.models.properties.modular_properties.phase_equil.bubble_dew import (
    IdealBubbleDew,
)
from idaes.models.properties.modular_properties.phase_equil.forms import fugacity
import idaes.models.properties.modular_properties.pure.Perrys as Perrys
import idaes.models.properties.modular_properties.pure.RPP4 as RPP4
import idaes.models.properties.modular_properties.pure.NIST as NIST


def test_examples():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    configuration_FcTP = {
        "components": {
            "benzene": {
                "type": Component,
                "dens_mol_liq_comp": Perrys,
                "enth_mol_liq_comp": Perrys,
                "enth_mol_ig_comp": RPP4,
                "pressure_sat_comp": NIST,
                "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
                "parameter_data": {
                    "mw": 78.1136e-3,  # [1]
                    "pressure_crit": 48.9e5,  # [1]
                    "temperature_crit": 562.2,  # [1]
                    "dens_mol_liq_comp_coeff": {
                        "eqn_type": 1,
                        "1": 1.0162,  # [2] pg. 2-98
                        "2": 0.2655,
                        "3": 562.16,
                        "4": 0.28212,
                    },
                    "cp_mol_ig_comp_coeff": {
                        "A": -3.392e1,  # [1]
                        "B": 4.739e-1,
                        "C": -3.017e-4,
                        "D": 7.130e-8,
                    },
                    "cp_mol_liq_comp_coeff": {
                        "1": 1.29e2,  # [2]
                        "2": -1.7e-1,
                        "3": 6.48e-4,
                        "4": 0,
                        "5": 0,
                    },
                    "enth_mol_form_liq_comp_ref": 49.0e3,  # [3]
                    "enth_mol_form_vap_comp_ref": 82.9e3,  # [3]
                    "pressure_sat_comp_coeff": {
                        "A": 4.72583,  # [NIST]
                        "B": 1660.652,
                        "C": -1.461,
                    },
                },
            },
            "toluene": {
                "type": Component,
                "dens_mol_liq_comp": Perrys,
                "enth_mol_liq_comp": Perrys,
                "enth_mol_ig_comp": RPP4,
                "pressure_sat_comp": NIST,
                "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
                "parameter_data": {
                    "mw": 92.1405e-3,  # [1]
                    "pressure_crit": 41e5,  # [1]
                    "temperature_crit": 591.8,  # [1]
                    "dens_mol_liq_comp_coeff": {
                        "eqn_type": 1,
                        "1": 0.8488,  # [2] pg. 2-98
                        "2": 0.26655,
                        "3": 591.8,
                        "4": 0.2878,
                    },
                    "cp_mol_ig_comp_coeff": {
                        "A": -2.435e1,
                        "B": 5.125e-1,
                        "C": -2.765e-4,
                        "D": 4.911e-8,
                    },
                    "cp_mol_liq_comp_coeff": {
                        "1": 1.40e2,  # [2]
                        "2": -1.52e-1,
                        "3": 6.95e-4,
                        "4": 0,
                        "5": 0,
                    },
                    "enth_mol_form_liq_comp_ref": 12.0e3,  # [3]
                    "enth_mol_form_vap_comp_ref": 50.1e3,  # [3]
                    "pressure_sat_comp_coeff": {
                        "A": 4.07827,  # [NIST]
                        "B": 1343.943,
                        "C": -53.773,
                    },
                },
            },
        },
        "phases": {
            "Liq": {"type": LiquidPhase, "equation_of_state": Ideal},
            "Vap": {"type": VaporPhase, "equation_of_state": Ideal},
        },
        "base_units": {
            "time": pyunits.s,
            "length": pyunits.m,
            "mass": pyunits.kg,
            "amount": pyunits.mol,
            "temperature": pyunits.K,
        },
        "state_definition": FcTP,
        "state_bounds": {
            "flow_mol_comp": (0, 100, 1000, pyunits.mol / pyunits.s),
            "temperature": (273.15, 300, 450, pyunits.K),
            "pressure": (5e4, 1e5, 1e6, pyunits.Pa),
        },
        "pressure_ref": 1e5,
        "temperature_ref": 300,
        "phases_in_equilibrium": [("Vap", "Liq")],
        "phase_equilibrium_state": {("Vap", "Liq"): SmoothVLE},
        "bubble_dew_method": IdealBubbleDew,
    }

    m.fs.properties_FTPx = GenericParameterBlock(
        **configuration_FTPx
    )  # Inlet property block
    m.fs.properties_FcTP = GenericParameterBlock(
        **configuration_FcTP
    )  # Outlet property block

    m.fs.translator = Translator(
        inlet_property_package=m.fs.properties_FTPx,
        outlet_property_package=m.fs.properties_FcTP,
    )

    DOF_initial = degrees_of_freedom(m)
    print("The initial degrees of freedom are: {0}".format(DOF_initial))
    assert DOF_initial == 9

    m.fs.translator.inlet.flow_mol.fix(
        100
    )  # converting to mol/s as unit basis is mol/s
    m.fs.translator.inlet.mole_frac_comp[0, "benzene"].fix(0.6)
    m.fs.translator.inlet.mole_frac_comp[0, "toluene"].fix(0.4)
    m.fs.translator.inlet.pressure.fix(101325)  # Pa
    m.fs.translator.inlet.temperature.fix(298)  # K

    blk = m.fs.translator

    blk.eq_benzene_balance = Constraint(
        expr=blk.properties_in[0].flow_mol
        * blk.properties_in[0].mole_frac_comp["benzene"]
        == blk.properties_out[0].flow_mol_comp["benzene"]
    )
    blk.eq_toluene_balance = Constraint(
        expr=blk.properties_in[0].flow_mol
        * blk.properties_in[0].mole_frac_comp["toluene"]
        == blk.properties_out[0].flow_mol_comp["toluene"]
    )
    blk.eq_equal_temperature = Constraint(
        expr=blk.properties_in[0].temperature == blk.properties_out[0].temperature
    )
    blk.eq_equal_pressure = Constraint(
        expr=blk.properties_in[0].pressure == blk.properties_out[0].pressure
    )

    DOF_final = degrees_of_freedom(m)
    print("The final degrees of freedom is: {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.translator.initialize(outlvl=idaeslog.WARNING)
    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert result.solver.termination_condition == TerminationCondition.optimal
    assert result.solver.status == SolverStatus.ok

    m.fs.translator.report()

    assert value(m.fs.translator.inlet.flow_mol[0]) == pytest.approx(100, rel=1e-6)
    assert value(m.fs.translator.inlet.mole_frac_comp[0, "benzene"]) == pytest.approx(
        0.6, rel=1e-6
    )
    assert value(m.fs.translator.inlet.mole_frac_comp[0, "toluene"]) == pytest.approx(
        0.4, rel=1e-6
    )

    assert value(m.fs.translator.outlet.flow_mol_comp[0, "benzene"]) == pytest.approx(
        60, rel=1e-6
    )
    assert value(m.fs.translator.outlet.flow_mol_comp[0, "toluene"]) == pytest.approx(
        40, rel=1e-6
    )
