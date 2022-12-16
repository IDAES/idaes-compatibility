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

from pyomo.environ import ConcreteModel, SolverFactory, value, units
from pyomo.opt import TerminationCondition, SolverStatus

from idaes.core import FlowsheetBlock
import idaes.logger as idaeslog
from idaes.models.properties import iapws95
from idaes.models.properties.helmholtz.helmholtz import PhaseType
from idaes.models.unit_models.pressure_changer import Turbine
from idaes.core.util.model_statistics import degrees_of_freedom


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = iapws95.Iapws95ParameterBlock()

    m.fs.turbine_case_1 = Turbine(property_package=m.fs.properties)

    DOF_initial = degrees_of_freedom(m)
    print("The initial DOF is {0}".format(DOF_initial))
    assert DOF_initial == 5

    m.fs.turbine_case_1.inlet.flow_mol[0].fix(100)
    m.fs.turbine_case_1.inlet.enth_mol[0].fix(
        value(iapws95.htpx(T=390 * units.K, P=150000 * units.Pa))
    )
    m.fs.turbine_case_1.inlet.pressure[0].fix(150000)
    m.fs.turbine_case_1.deltaP.fix(-10000)
    m.fs.turbine_case_1.efficiency_isentropic.fix(0.9)

    DOF_final = degrees_of_freedom(m)
    print("The final DOF is {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.turbine_case_1.initialize(outlvl=idaeslog.INFO)
    opt = SolverFactory("ipopt")
    solve_status = opt.solve(m, tee=True)

    assert solve_status.solver.termination_condition == TerminationCondition.optimal
    assert solve_status.solver.status == SolverStatus.ok

    m.fs.turbine_case_1.outlet.pressure.display()
    m.fs.turbine_case_1.report()

    assert m.fs.turbine_case_1.outlet.pressure[0].value == pytest.approx(
        140000, rel=1e-5
    )
    assert m.fs.turbine_case_1.work_mechanical[0].value == pytest.approx(
        -19597.158, rel=1e-5
    )

    m.fs.turbine_case_2 = Turbine(property_package=m.fs.properties)

    DOF_initial = degrees_of_freedom(m.fs.turbine_case_2)
    print("The initial DOF is {0}".format(DOF_initial))
    assert DOF_initial == 5

    m.fs.turbine_case_2.inlet.flow_mol[0].fix(100)
    m.fs.turbine_case_2.inlet.enth_mol[0].fix(
        value(iapws95.htpx(T=390 * units.K, P=150000 * units.Pa))
    )
    m.fs.turbine_case_2.inlet.pressure[0].fix(150000)
    m.fs.turbine_case_2.ratioP.fix(14 / 15)
    m.fs.turbine_case_2.efficiency_isentropic.fix(0.9)

    DOF_final = degrees_of_freedom(m.fs.turbine_case_2)
    print("The final DOF is {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.turbine_case_2.initialize(outlvl=idaeslog.INFO)
    opt = SolverFactory("ipopt")
    solve_status = opt.solve(m.fs.turbine_case_2, tee=True)

    assert solve_status.solver.termination_condition == TerminationCondition.optimal
    assert solve_status.solver.status == SolverStatus.ok

    m.fs.turbine_case_2.outlet.pressure[0].display()
    m.fs.turbine_case_2.report()

    assert m.fs.turbine_case_2.outlet.pressure[0].value == pytest.approx(
        140000, rel=1e-5
    )
    assert m.fs.turbine_case_2.work_mechanical[0].value == pytest.approx(
        -19597.158, rel=1e-5
    )
