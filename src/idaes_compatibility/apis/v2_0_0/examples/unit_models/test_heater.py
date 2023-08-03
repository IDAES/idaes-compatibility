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
import matplotlib.pyplot as plt
import numpy as np

from pyomo.environ import ConcreteModel, SolverFactory, value
from pyomo.opt import TerminationCondition, SolverStatus

from idaes.core import FlowsheetBlock
import idaes.logger as idaeslog
from idaes.models.properties.activity_coeff_models import BTX_activity_coeff_VLE
from idaes.models.unit_models.heater import Heater
from idaes.core.util.model_statistics import degrees_of_freedom


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = BTX_activity_coeff_VLE.BTXParameterBlock(
        valid_phase="Liq", activity_coeff_model="Ideal"
    )
    m.fs.heater = Heater(property_package=m.fs.properties)

    DOF_initial = degrees_of_freedom(m)
    print("The initial DOF is {0}".format(DOF_initial))
    assert DOF_initial == 6

    m.fs.heater.inlet.flow_mol.fix(
        1 * 1000 / 3600
    )  # converting to mol/s as unit basis is mol/s
    m.fs.heater.inlet.mole_frac_comp[0, "benzene"].fix(0.4)
    m.fs.heater.inlet.mole_frac_comp[0, "toluene"].fix(0.6)
    m.fs.heater.inlet.pressure.fix(101325)  # Pa
    m.fs.heater.inlet.temperature.fix(353)  # K

    m.fs.heater.outlet.temperature.fix(363)
    DOF_final = degrees_of_freedom(m)
    print("The final DOF is {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.heater.initialize(outlvl=idaeslog.WARNING)
    m.fs.heater.initialize(outlvl=idaeslog.INFO_HIGH)
    opt = SolverFactory("ipopt")
    solve_status = opt.solve(m, tee=True)

    assert solve_status.solver.termination_condition == TerminationCondition.optimal
    assert solve_status.solver.status == SolverStatus.ok

    m.fs.heater.heat_duty.display()
    m.fs.heater.report()

    assert m.fs.heater.heat_duty[0].value == pytest.approx(459.10, abs=1e-2)
    assert m.fs.heater.outlet.flow_mol[0].value == pytest.approx(0.27778, abs=1e-2)
    assert m.fs.heater.outlet.mole_frac_comp[0, "benzene"].value == pytest.approx(
        0.4, abs=1e-3
    )
    assert m.fs.heater.outlet.mole_frac_comp[0, "toluene"].value == pytest.approx(
        0.6, abs=1e-3
    )
    assert m.fs.heater.outlet.temperature[0].value == pytest.approx(363, abs=1e-2)
    assert m.fs.heater.outlet.pressure[0].value == pytest.approx(101325, abs=1)

    m.fs.heater.outlet.temperature.unfix()
    m.fs.heater.heat_duty.fix(459.10147722222354)
    solve_status = opt.solve(m, tee=True)

    assert solve_status.solver.termination_condition == TerminationCondition.optimal
    assert solve_status.solver.status == SolverStatus.ok

    m.fs.heater.outlet.temperature.display()
    m.fs.heater.report()

    assert m.fs.heater.heat_duty[0].value == pytest.approx(459.10, abs=1e-2)
    assert m.fs.heater.outlet.flow_mol[0].value == pytest.approx(0.27778, abs=1e-2)
    assert m.fs.heater.outlet.mole_frac_comp[0, "benzene"].value == pytest.approx(
        0.4, abs=1e-3
    )
    assert m.fs.heater.outlet.mole_frac_comp[0, "toluene"].value == pytest.approx(
        0.6, abs=1e-3
    )
    assert m.fs.heater.outlet.temperature[0].value == pytest.approx(363, abs=1e-2)
    assert m.fs.heater.outlet.pressure[0].value == pytest.approx(101325, abs=1)

    m.fs.heater.heat_duty.unfix()

    outlet_temp_fixed = [
        91.256405 + 273.15,
        90.828456 + 273.15,
        86.535145 + 273.15,
        89.383218 + 273.15,
        93.973657 + 273.15,
        85.377274 + 273.15,
        92.399101 + 273.15,
        94.151562 + 273.15,
        87.564579 + 273.15,
        88.767855 + 273.15,
    ]

    heat_duty = []
    for temp in outlet_temp_fixed:
        m.fs.heater.outlet.temperature.fix(temp)
        solve_status = opt.solve(m)
        if solve_status.solver.termination_condition == TerminationCondition.optimal:
            heat_duty.append(m.fs.heater.heat_duty[0].value)

    plt.figure("Q vs. Temperature")
    plt.plot(outlet_temp_fixed, heat_duty, "bo")
    plt.xlim(358.15, 368.15)
    plt.ylim(250, 700)
    plt.xlabel("Outlet Temperature (K)")
    plt.ylabel("Heat Duty (W)")
    plt.grid()
