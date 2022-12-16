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

from pyomo.environ import ConcreteModel, value, TerminationCondition, SolverStatus

from idaes.core.solvers import get_solver
from idaes.core import FlowsheetBlock
from idaes.models.unit_models.feed_flash import FeedFlash, FlashType
import idaes.logger as idaeslog
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
from idaes.models.properties.modular_properties.examples.BT_ideal import configuration
from idaes.models.properties import iapws95


def test_example():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties_1 = GenericParameterBlock(**configuration)
    m.fs.properties_2 = iapws95.Iapws95ParameterBlock(phase_presentation=iapws95.PhaseType.LG)

    m.fs.feed_1 = FeedFlash(property_package=m.fs.properties_1)

    DOF_initial = degrees_of_freedom(m)
    print('The initial degrees of freedom are: {0}'.format(DOF_initial))
    assert DOF_initial == 5

    m.fs.feed_1.flow_mol.fix(100) # converting to mol/s as unit basis is mol/s
    m.fs.feed_1.mole_frac_comp[0, "benzene"].fix(0.6)
    m.fs.feed_1.mole_frac_comp[0, "toluene"].fix(0.4)
    m.fs.feed_1.pressure.fix(101325) # Pa
    m.fs.feed_1.temperature.fix(298) # K

    DOF_final = degrees_of_freedom(m)
    print('The final degrees of freedom is: {0}'.format(DOF_final))
    assert DOF_final == 0

    m.fs.feed_1.initialize(outlvl=idaeslog.WARNING)

    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert result.solver.termination_condition == TerminationCondition.optimal
    assert result.solver.status == SolverStatus.ok

    m.fs.feed_1.report()

    assert value(m.fs.feed_1.outlet.flow_mol[0]) == pytest.approx(100, rel=1e-6)
    assert value(m.fs.feed_1.mole_frac_comp[0, "benzene"]) == pytest.approx(0.6, rel=1e-6)
    assert value(m.fs.feed_1.mole_frac_comp[0, "toluene"]) == pytest.approx(0.4, rel=1e-6)

    ## Case 2:
    m.fs.feed_2 = FeedFlash(property_package=m.fs.properties_2, flash_type=FlashType.isenthalpic)

    DOF_initial = degrees_of_freedom(m)
    print('The initial degrees of freedom are: {0}'.format(DOF_initial))
    assert DOF_initial == 3

    m.fs.feed_2.flow_mol.fix(100) # converting to mol/s as unit basis is mol/s
    m.fs.feed_2.enth_mol.fix(24000) # J/mol
    m.fs.feed_2.pressure.fix(101325) # Pa

    DOF_final = degrees_of_freedom(m)
    print('The final degrees of freedom is: {0}'.format(DOF_final))
    assert DOF_final == 0

    m.fs.feed_2.initialize(outlvl=idaeslog.WARNING)
    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert result.solver.termination_condition == TerminationCondition.optimal
    assert result.solver.status == SolverStatus.ok

    m.fs.feed_2.report()
    assert value(m.fs.feed_2.outlet.flow_mol[0]) == pytest.approx(100, rel=1e-6)
    assert value(m.fs.feed_2.outlet.enth_mol[0]) == pytest.approx(24000, rel=1e-6)
