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
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.examples.BT_ideal import configuration
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver
from idaes.models.unit_models.heat_exchanger_1D import (
    HeatExchanger1D,
    HeatExchangerFlowPattern,
)


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties_shell = iapws95.Iapws95ParameterBlock()
    m.fs.properties_tube = GenericParameterBlock(**configuration)

    m.fs.heat_exchanger = HeatExchanger1D(
        hot_side_name="shell",
        cold_side_name="tube",
        shell={
            "property_package": m.fs.properties_shell,
            "transformation_method": "dae.finite_difference",
            "transformation_scheme": "BACKWARD",
        },
        tube={
            "property_package": m.fs.properties_tube,
            "transformation_method": "dae.finite_difference",
            "transformation_scheme": "BACKWARD",
        },
        finite_elements=20,
        flow_type=HeatExchangerFlowPattern.cocurrent,
    )

    DOF_initial = degrees_of_freedom(m)
    print("The initial DOF is {0}".format(DOF_initial))
    assert DOF_initial == 31

    h = htpx(450 * units.K, P=101325 * units.Pa)  # calculate IAPWS enthalpy
    m.fs.heat_exchanger.hot_side_inlet.flow_mol.fix(100)  # mol/s
    m.fs.heat_exchanger.hot_side_inlet.pressure.fix(101325)  # Pa
    m.fs.heat_exchanger.hot_side_inlet.enth_mol.fix(h)  # J/mol

    DOF_initial = degrees_of_freedom(m)
    print("The DOF is {0}".format(DOF_initial))

    m.fs.heat_exchanger.cold_side_inlet.flow_mol.fix(250)  # mol/s
    m.fs.heat_exchanger.cold_side_inlet.mole_frac_comp[0, "benzene"].fix(0.4)
    m.fs.heat_exchanger.cold_side_inlet.mole_frac_comp[0, "toluene"].fix(0.6)
    m.fs.heat_exchanger.cold_side_inlet.pressure.fix(101325)  # Pa
    m.fs.heat_exchanger.cold_side_inlet.temperature[0].fix(350)  # K

    DOF_final = degrees_of_freedom(m)
    print("The DOF is {0}".format(DOF_final))

    m.fs.heat_exchanger.area.fix(0.5)  # m2
    m.fs.heat_exchanger.length.fix(4.85)  # m
    m.fs.heat_exchanger.heat_transfer_coefficient.fix(500)  # W/m2/K

    DOF_final = degrees_of_freedom(m)
    print("The DOF is {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.heat_exchanger.initialize(outlvl=idaeslog.INFO)
    opt = get_solver()
    solve_status = opt.solve(m, tee=True)

    m.fs.heat_exchanger.report()

    assert_optimal_termination(solve_status)

    assert value(m.fs.heat_exchanger.hot_side_outlet.enth_mol[0]) == pytest.approx(
        htpx(444.47 * units.K, P=101325 * units.Pa), rel=1e-3
    )
    assert value(m.fs.heat_exchanger.cold_side_outlet.temperature[0]) == pytest.approx(
        368.39, rel=1e-3
    )

    m.fs.heat_exchanger.area.unfix()
    m.fs.heat_exchanger.hot_side_outlet.enth_mol.fix(
        htpx(444.47 * units.K, P=101325 * units.Pa)
    )

    DOF_final = degrees_of_freedom(m)
    print("The DOF is {0}".format(DOF_final))
    assert DOF_final == 0

    result = opt.solve(m)

    print(result)
    m.fs.heat_exchanger.report()

    assert_optimal_termination(result)
    assert value(m.fs.heat_exchanger.area) == pytest.approx(0.5, abs=1e-2)
    assert value(m.fs.heat_exchanger.cold_side_outlet.temperature[0]) == pytest.approx(
        368.39, rel=1e-3
    )
