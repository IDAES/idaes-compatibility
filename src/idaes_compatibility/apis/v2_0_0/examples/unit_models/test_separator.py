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

from pyomo.environ import ConcreteModel, value
from pyomo.opt import TerminationCondition, SolverStatus

from idaes.core.solvers import get_solver
from idaes.core import FlowsheetBlock
from idaes.core import MaterialBalanceType
from idaes.models.unit_models import Separator
from idaes.models.unit_models.separator import SplittingType
import idaes.logger as idaeslog
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.examples.CO2_H2O_Ideal_VLE import (
    configuration,
)
from idaes.core.util.model_statistics import degrees_of_freedom


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = GenericParameterBlock(**configuration)
    m.fs.sep_1 = Separator(
        property_package=m.fs.properties,
        split_basis=SplittingType.totalFlow,
        outlet_list=["a1", "b1", "c1"],  # creates three outlet streams
        ideal_separation=False,
        has_phase_equilibrium=False,
    )

    DOF_initial = degrees_of_freedom(m)
    print("The initial degrees of freedom are: {0}".format(DOF_initial))
    assert DOF_initial == 7

    m.fs.sep_1.inlet.flow_mol.fix(10)  # converting to mol/s as unit basis is mol/s
    m.fs.sep_1.inlet.mole_frac_comp[0, "H2O"].fix(0.9)
    m.fs.sep_1.inlet.mole_frac_comp[0, "CO2"].fix(0.1)
    m.fs.sep_1.inlet.pressure.fix(101325)  # Pa
    m.fs.sep_1.inlet.temperature.fix(353)  # K
    m.fs.sep_1.split_fraction[0, "a1"].fix(0.2)
    m.fs.sep_1.split_fraction[0, "b1"].fix(0.5)

    DOF_final = degrees_of_freedom(m)
    print("The final degrees of freedom is: {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.sep_1.initialize(outlvl=idaeslog.WARNING)

    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert result.solver.termination_condition == TerminationCondition.optimal
    assert result.solver.status == SolverStatus.ok

    m.fs.sep_1.report()

    assert value(m.fs.sep_1.a1.flow_mol[0]) == pytest.approx(2, rel=1e-6)
    assert value(m.fs.sep_1.a1.pressure[0]) == pytest.approx(101325, rel=1e-6)
    assert value(m.fs.sep_1.a1.temperature[0]) == pytest.approx(353, rel=1e-6)
    assert value(m.fs.sep_1.a1.mole_frac_comp[0, "H2O"]) == pytest.approx(0.9, rel=1e-6)
    assert value(m.fs.sep_1.a1.mole_frac_comp[0, "CO2"]) == pytest.approx(0.1, rel=1e-6)

    assert value(m.fs.sep_1.b1.flow_mol[0]) == pytest.approx(5, rel=1e-6)
    assert value(m.fs.sep_1.b1.pressure[0]) == pytest.approx(101325, rel=1e-6)
    assert value(m.fs.sep_1.b1.temperature[0]) == pytest.approx(353, rel=1e-6)
    assert value(m.fs.sep_1.b1.mole_frac_comp[0, "H2O"]) == pytest.approx(0.9, rel=1e-6)
    assert value(m.fs.sep_1.b1.mole_frac_comp[0, "CO2"]) == pytest.approx(0.1, rel=1e-6)

    assert value(m.fs.sep_1.c1.flow_mol[0]) == pytest.approx(3, rel=1e-6)
    assert value(m.fs.sep_1.c1.pressure[0]) == pytest.approx(101325, rel=1e-6)
    assert value(m.fs.sep_1.c1.temperature[0]) == pytest.approx(353, rel=1e-6)
    assert value(m.fs.sep_1.c1.mole_frac_comp[0, "H2O"]) == pytest.approx(0.9, rel=1e-6)
    assert value(m.fs.sep_1.c1.mole_frac_comp[0, "CO2"]) == pytest.approx(0.1, rel=1e-6)

    m.fs.sep_2 = Separator(
        property_package=m.fs.properties,
        split_basis=SplittingType.phaseFlow,
        outlet_list=["a2", "b2"],
        ideal_separation=False,
        has_phase_equilibrium=False,
    )

    DOF_initial = degrees_of_freedom(m)
    print("The initial degrees of freedom are: {0}".format(DOF_initial))
    assert DOF_initial == 7

    m.fs.sep_2.inlet.flow_mol.fix(10)  # converting to mol/s as unit basis is mol/s
    m.fs.sep_2.inlet.mole_frac_comp[0, "H2O"].fix(0.9)
    m.fs.sep_2.inlet.mole_frac_comp[0, "CO2"].fix(0.1)
    m.fs.sep_2.inlet.pressure.fix(101325)  # Pa
    m.fs.sep_2.inlet.temperature.fix(353)  # K
    m.fs.sep_2.split_fraction[0, "a2", "Vap"].fix(0.8)
    m.fs.sep_2.split_fraction[0, "b2", "Liq"].fix(0.8)

    DOF_final = degrees_of_freedom(m)
    print("The final degrees of freedom is: {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.sep_2.initialize(outlvl=idaeslog.WARNING)
    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert result.solver.termination_condition == TerminationCondition.optimal
    assert result.solver.status == SolverStatus.ok

    m.fs.sep_2.report()

    assert value(m.fs.sep_2.a2.flow_mol[0]) == pytest.approx(3.1220, rel=1e-3)
    assert value(m.fs.sep_2.a2.pressure[0]) == pytest.approx(101325, rel=1e-6)
    assert value(m.fs.sep_2.a2.temperature[0]) == pytest.approx(353, rel=1e-6)
    assert value(m.fs.sep_2.a2.mole_frac_comp[0, "H2O"]) == pytest.approx(
        0.74375, rel=1e-5
    )
    assert value(m.fs.sep_2.a2.mole_frac_comp[0, "CO2"]) == pytest.approx(
        0.25625, rel=1e-5
    )
    assert value(m.fs.sep_2.b2.flow_mol[0]) == pytest.approx(6.878, rel=1e-3)
    assert value(m.fs.sep_2.b2.pressure[0]) == pytest.approx(101325, rel=1e-6)
    assert value(m.fs.sep_2.b2.temperature[0]) == pytest.approx(353, rel=1e-6)
    assert value(m.fs.sep_2.b2.mole_frac_comp[0, "H2O"]) == pytest.approx(
        0.97092, rel=1e-5
    )
    assert value(m.fs.sep_2.b2.mole_frac_comp[0, "CO2"]) == pytest.approx(
        0.029078, rel=1e-5
    )

    m.fs.sep_3 = Separator(
        property_package=m.fs.properties,
        split_basis=SplittingType.componentFlow,
        outlet_list=["a3", "b3", "c3", "d3"],
        ideal_separation=False,
        has_phase_equilibrium=False,
    )

    DOF_initial = degrees_of_freedom(m)
    print("The initial degrees of freedom are: {0}".format(DOF_initial))
    assert DOF_initial == 11

    m.fs.sep_3.inlet.flow_mol.fix(10)  # converting to mol/s as unit basis is mol/s
    m.fs.sep_3.inlet.mole_frac_comp[0, "H2O"].fix(0.9)
    m.fs.sep_3.inlet.mole_frac_comp[0, "CO2"].fix(0.1)
    m.fs.sep_3.inlet.pressure.fix(101325)  # Pa
    m.fs.sep_3.inlet.temperature.fix(353)  # K

    m.fs.sep_3.split_fraction[0, "a3", "H2O"].fix(0.25)
    m.fs.sep_3.split_fraction[0, "b3", "H2O"].fix(0.5)
    m.fs.sep_3.split_fraction[0, "c3", "H2O"].fix(0.1)

    m.fs.sep_3.split_fraction[0, "a3", "CO2"].fix(0.1)
    m.fs.sep_3.split_fraction[0, "b3", "CO2"].fix(0.2)
    m.fs.sep_3.split_fraction[0, "c3", "CO2"].fix(0.3)

    DOF_final = degrees_of_freedom(m)
    print("The final degrees of freedom is: {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.sep_3.initialize(outlvl=idaeslog.WARNING)

    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert result.solver.termination_condition == TerminationCondition.optimal
    assert result.solver.status == SolverStatus.ok

    m.fs.sep_3.report()

    assert value(m.fs.sep_3.a3.flow_mol[0]) == pytest.approx(2.35, rel=1e-3)
    assert value(m.fs.sep_3.a3.pressure[0]) == pytest.approx(101325, rel=1e-6)
    assert value(m.fs.sep_3.a3.temperature[0]) == pytest.approx(353, rel=1e-6)
    assert value(m.fs.sep_3.a3.mole_frac_comp[0, "H2O"]) == pytest.approx(
        0.95745, rel=1e-5
    )
    assert value(m.fs.sep_3.a3.mole_frac_comp[0, "CO2"]) == pytest.approx(
        0.042553, rel=1e-5
    )
    assert value(m.fs.sep_3.b3.flow_mol[0]) == pytest.approx(4.7, rel=1e-3)
    assert value(m.fs.sep_3.b3.pressure[0]) == pytest.approx(101325, rel=1e-6)
    assert value(m.fs.sep_3.b3.temperature[0]) == pytest.approx(353, rel=1e-6)
    assert value(m.fs.sep_3.b3.mole_frac_comp[0, "H2O"]) == pytest.approx(
        0.95745, rel=1e-5
    )
    assert value(m.fs.sep_3.b3.mole_frac_comp[0, "CO2"]) == pytest.approx(
        0.042553, rel=1e-5
    )
    assert value(m.fs.sep_3.c3.flow_mol[0]) == pytest.approx(1.2, rel=1e-3)
    assert value(m.fs.sep_3.c3.pressure[0]) == pytest.approx(101325, rel=1e-6)
    assert value(m.fs.sep_3.c3.temperature[0]) == pytest.approx(353, rel=1e-6)
    assert value(m.fs.sep_3.c3.mole_frac_comp[0, "H2O"]) == pytest.approx(
        0.75, rel=1e-5
    )
    assert value(m.fs.sep_3.c3.mole_frac_comp[0, "CO2"]) == pytest.approx(
        0.25, rel=1e-5
    )
    assert value(m.fs.sep_3.d3.flow_mol[0]) == pytest.approx(1.75, rel=1e-3)
    assert value(m.fs.sep_3.d3.pressure[0]) == pytest.approx(101325, rel=1e-6)
    assert value(m.fs.sep_3.d3.temperature[0]) == pytest.approx(353, rel=1e-6)
    assert value(m.fs.sep_3.d3.mole_frac_comp[0, "H2O"]) == pytest.approx(
        0.77143, rel=1e-5
    )
    assert value(m.fs.sep_3.d3.mole_frac_comp[0, "CO2"]) == pytest.approx(
        0.22857, rel=1e-5
    )
