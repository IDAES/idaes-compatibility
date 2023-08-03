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
    SolverFactory,
    value,
    units,
    TerminationCondition,
    SolverStatus,
)

from idaes.core import FlowsheetBlock
import idaes.logger as idaeslog
from idaes.models.properties.swco2 import SWCO2ParameterBlock, StateVars, htpx
from idaes.models.unit_models.pressure_changer import (
    PressureChanger,
    ThermodynamicAssumption,
)
from idaes.core.util.model_statistics import degrees_of_freedom


def test_example():
    # Create the model
    m = ConcreteModel()
    m.fs = FlowsheetBlock(
        dynamic=False
    )  # dynamic or ss flowsheet needs to be specified here

    m.fs.properties = SWCO2ParameterBlock()

    m.fs.compr_case_1 = PressureChanger(
        dynamic=False,
        property_package=m.fs.properties,
        compressor=True,
        thermodynamic_assumption=ThermodynamicAssumption.isentropic,
    )

    DOF_initial = degrees_of_freedom(m)
    print("The initial DOF is {0}".format(DOF_initial))
    assert DOF_initial == 5

    ### Fix Inlet Stream Conditions
    m.fs.compr_case_1.inlet.flow_mol[0].fix(91067)  # mol/s
    m.fs.compr_case_1.inlet.enth_mol[0].fix(
        value(htpx(T=308.15 * units.K, P=9.1107e06 * units.Pa))
    )  # T in K, P in Pa
    m.fs.compr_case_1.inlet.pressure[0].fix(9.1107e06)

    m.fs.compr_case_1.deltaP.fix(2.5510e07)
    m.fs.compr_case_1.efficiency_isentropic.fix(0.85)

    DOF_final = degrees_of_freedom(m)
    print("The final DOF is {0}".format(DOF_final))
    assert DOF_final == 0

    # Initialize the flowsheet, and set the output at INFO level
    m.fs.compr_case_1.initialize(outlvl=idaeslog.INFO)

    opt = SolverFactory("ipopt")
    solve_status = opt.solve(m, tee=True)

    assert solve_status.solver.termination_condition == TerminationCondition.optimal
    assert solve_status.solver.status == SolverStatus.ok

    ### View Results
    m.fs.compr_case_1.outlet.pressure.display()
    m.fs.compr_case_1.report()

    # Check results
    assert m.fs.compr_case_1.outlet.pressure[0].value == pytest.approx(
        34620700.0, abs=1e-2
    )
    assert m.fs.compr_case_1.work_isentropic[0].value == pytest.approx(
        135439976.18, rel=1e-5
    )
    assert m.fs.compr_case_1.work_mechanical[0].value == pytest.approx(
        159341148.45, rel=1e-5
    )

    ## Case 2: Fix pressure ratio and isentropic efficiency
    m.fs.compr_case_2 = PressureChanger(
        dynamic=False,
        property_package=m.fs.properties,
        compressor=True,
        thermodynamic_assumption=ThermodynamicAssumption.isentropic,
    )

    DOF_initial = degrees_of_freedom(m.fs.compr_case_2)
    print("The initial DOF is {0}".format(DOF_initial))
    assert DOF_initial == 5

    ### Fix Inlet Stream Conditions
    m.fs.compr_case_2.inlet.flow_mol[0].fix(
        91067
    )  # converting to mol/s as unit basis is mol/s
    m.fs.compr_case_2.inlet.enth_mol[0].fix(
        value(htpx(T=308.15 * units.K, P=9.1107e06 * units.Pa))
    )
    m.fs.compr_case_2.inlet.pressure[0].fix(9.1107e06)
    m.fs.compr_case_2.ratioP.fix(3.8)
    m.fs.compr_case_2.efficiency_isentropic.fix(0.85)

    DOF_final = degrees_of_freedom(m.fs.compr_case_2)
    print("The final DOF is {0}".format(DOF_final))
    assert DOF_final == 0

    ### Initialization
    m.fs.compr_case_2.initialize(outlvl=idaeslog.INFO)
    opt = SolverFactory("ipopt")
    solve_status = opt.solve(m.fs.compr_case_2, tee=True)

    assert solve_status.solver.termination_condition == TerminationCondition.optimal
    assert solve_status.solver.status == SolverStatus.ok

    m.fs.compr_case_2.outlet.pressure[0].display()
    m.fs.compr_case_2.report()

    assert m.fs.compr_case_2.outlet.pressure[0].value == pytest.approx(
        34620660, abs=1e-2
    )
    assert m.fs.compr_case_2.work_isentropic[0].value == pytest.approx(
        135439779.20953986, rel=1e-5
    )
    assert m.fs.compr_case_2.work_mechanical[0].value == pytest.approx(
        159340916.71710572, rel=1e-5
    )
