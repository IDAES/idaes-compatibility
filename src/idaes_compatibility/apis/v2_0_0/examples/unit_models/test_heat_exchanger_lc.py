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
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    value,
    units,
    assert_optimal_termination,
)

import idaes.logger as idaeslog
from idaes.core import FlowsheetBlock
from idaes.models.properties import iapws95
from idaes.models.properties.iapws95 import htpx
from idaes.models.properties.modular_properties import GenericParameterBlock
from idaes.models.properties.modular_properties.examples.BT_ideal import configuration
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver
from idaes.models.unit_models import (
    HeatExchangerLumpedCapacitance,
    HeatExchangerFlowPattern,
)


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties_shell = iapws95.Iapws95ParameterBlock()
    m.fs.properties_tube = GenericParameterBlock(**configuration)

    m.fs.heat_exchanger = HeatExchangerLumpedCapacitance(
        hot_side_name="shell",
        cold_side_name="tube",
        shell={"property_package": m.fs.properties_shell},
        tube={"property_package": m.fs.properties_tube},
        flow_pattern=HeatExchangerFlowPattern.crossflow,
        dynamic_heat_balance=False,
    )

    DOF_initial = degrees_of_freedom(m)
    print("The initial DOF is {0}".format(DOF_initial))
    assert DOF_initial == 12

    h = htpx(400 * units.K, P=101325 * units.Pa)  # calculate IAPWS enthalpy
    m.fs.heat_exchanger.shell_inlet.flow_mol.fix(100)  # mol/s
    m.fs.heat_exchanger.shell_inlet.pressure.fix(101325)  # Pa
    m.fs.heat_exchanger.shell_inlet.enth_mol.fix(h)  # J/mol

    DOF_initial = degrees_of_freedom(m)
    print("The DOF is {0}".format(DOF_initial))

    m.fs.heat_exchanger.tube_inlet.flow_mol.fix(250)  # mol/s
    m.fs.heat_exchanger.tube_inlet.mole_frac_comp[0, "benzene"].fix(0.4)
    m.fs.heat_exchanger.tube_inlet.mole_frac_comp[0, "toluene"].fix(0.6)
    m.fs.heat_exchanger.tube_inlet.pressure.fix(101325)  # Pa
    m.fs.heat_exchanger.tube_inlet.temperature[0].fix(380)  # K

    DOF_final = degrees_of_freedom(m)
    print("The DOF is {0}".format(DOF_final))

    m.fs.heat_exchanger.area.fix(50)  # m2
    m.fs.heat_exchanger.ua_hot_side.fix(200 * 1000)  # W/m2/K
    m.fs.heat_exchanger.ua_cold_side.fix(200 * 1000)  # W/m2/K
    m.fs.heat_exchanger.crossflow_factor.fix(0.6)

    DOF_final = degrees_of_freedom(m)
    print("The DOF is {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.heat_exchanger.initialize(outlvl=idaeslog.INFO)
    opt = get_solver()
    solve_status = opt.solve(m)

    m.fs.heat_exchanger.report()

    assert_optimal_termination(solve_status)

    assert value(
        m.fs.heat_exchanger.shell.properties_out[0].temperature
    ) == pytest.approx(380.0, abs=1e-2)
    assert value(
        m.fs.heat_exchanger.tube.properties_out[0].temperature
    ) == pytest.approx(382.38, abs=1e-2)
