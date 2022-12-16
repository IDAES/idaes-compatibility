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
import math

from pyomo.environ import ConcreteModel, Constraint, value, SolverFactory, units
from pyomo.opt import TerminationCondition, SolverStatus

from idaes.core.solvers import get_solver
from idaes.core import FlowsheetBlock
from idaes.models.unit_models import Valve
from idaes.models.unit_models import ValveFunctionType
import idaes.logger as idaeslog
from idaes.models.properties import iapws95
from idaes.core.util.model_statistics import degrees_of_freedom


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = iapws95.Iapws95ParameterBlock()
    m.fs.valve = Valve(
        valve_function_callback=ValveFunctionType.linear,
        property_package=m.fs.properties,
    )

    DOF_initial = degrees_of_freedom(m)
    print("The initial degrees of freedom is: {0}".format(DOF_initial))
    assert DOF_initial == 3

    fin = 1000  # mol/s
    pin = 202650  # Pa
    pout = 101325  # Pa
    tin = 298  # K

    hin = iapws95.htpx(T=tin * units.K, P=pin * units.Pa)  # J/mol
    cv = 1000 / math.sqrt(pin - pout) / 0.5

    m.fs.valve.inlet.enth_mol[0].fix(hin)
    m.fs.valve.inlet.flow_mol[0].fix(fin)
    m.fs.valve.inlet.pressure[0].fix(pin)
    m.fs.valve.outlet.pressure[0].set_value(pout)
    m.fs.valve.Cv.fix(cv)
    m.fs.valve.valve_opening.fix(0.5)

    DOF_final = degrees_of_freedom(m)
    print("The final degrees of freedom is: {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.valve.initialize(outlvl=idaeslog.WARNING)
    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert result.solver.termination_condition == TerminationCondition.optimal
    assert result.solver.status == SolverStatus.ok

    m.fs.valve.report()

    assert value(m.fs.valve.outlet.flow_mol[0]) == pytest.approx(1000, rel=1e-6)
    assert value(m.fs.valve.outlet.pressure[0]) == pytest.approx(101325, rel=1e-6)
    assert value(m.fs.valve.outlet.enth_mol[0]) == pytest.approx(1880.557, rel=1e-3)
