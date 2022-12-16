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
from idaes.models.unit_models import Feed
import idaes.logger as idaeslog
from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
from idaes.models.properties.modular_properties.examples.BT_ideal import configuration
from idaes.core.util.model_statistics import degrees_of_freedom


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = GenericParameterBlock(**configuration)
    m.fs.feed = Feed(property_package=m.fs.properties)

    DOF_initial = degrees_of_freedom(m)
    print('The initial degrees of freedom are: {0}'.format(DOF_initial))
    assert DOF_initial == 5

    m.fs.feed.flow_mol.fix(100) # converting to mol/s as unit basis is mol/s
    m.fs.feed.mole_frac_comp[0, "benzene"].fix(0.6)
    m.fs.feed.mole_frac_comp[0, "toluene"].fix(0.4)
    m.fs.feed.pressure.fix(101325) # Pa
    m.fs.feed.temperature.fix(298) # K

    DOF_final = degrees_of_freedom(m)
    print('The final degrees of freedom is: {0}'.format(DOF_final))
    assert DOF_final == 0

    m.fs.feed.initialize(outlvl=idaeslog.WARNING)
    solver = get_solver()
    result = solver.solve(m, tee=True)

    # Check if termination condition is optimal
    assert result.solver.termination_condition == TerminationCondition.optimal
    assert result.solver.status == SolverStatus.ok

    m.fs.feed.report()
    assert value(m.fs.feed.outlet.flow_mol[0]) == pytest.approx(100, rel=1e-6)
    assert value(m.fs.feed.mole_frac_comp[0, "benzene"]) == pytest.approx(0.6, rel=1e-6)
    assert value(m.fs.feed.mole_frac_comp[0, "toluene"]) == pytest.approx(0.4, rel=1e-6)
