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
from idaes.models.unit_models import StateJunction
import idaes.logger as idaeslog
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.examples.BT_ideal import (
    configuration as configuration,
)
from idaes.models.properties.examples.saponification_thermo import (
    SaponificationParameterBlock,
)
from idaes.core.util.model_statistics import degrees_of_freedom


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties_1 = GenericParameterBlock(**configuration)
    m.fs.properties_2 = SaponificationParameterBlock()

    m.fs.statejunction_1 = StateJunction(property_package=m.fs.properties_1)

    DOF_initial = degrees_of_freedom(m)
    print("The initial degrees of freedom is: {0}".format(DOF_initial))
    assert DOF_initial == 5

    m.fs.statejunction_1.inlet.flow_mol.fix(
        100
    )  # converting to mol/s as unit basis is mol/s
    m.fs.statejunction_1.inlet.mole_frac_comp[0, "benzene"].fix(0.6)
    m.fs.statejunction_1.inlet.mole_frac_comp[0, "toluene"].fix(0.4)
    m.fs.statejunction_1.inlet.pressure.fix(101325)  # Pa
    m.fs.statejunction_1.inlet.temperature.fix(298.15)  # K

    DOF_final = degrees_of_freedom(m)
    print("The final degrees of freedom is: {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.statejunction_1.initialize(outlvl=idaeslog.WARNING)
    solver = get_solver()
    result = solver.solve(m, tee=True)

    assert result.solver.termination_condition == TerminationCondition.optimal
    assert result.solver.status == SolverStatus.ok

    m.fs.statejunction_1.report()

    assert value(m.fs.statejunction_1.outlet.flow_mol[0]) == pytest.approx(
        100, rel=1e-6
    )
    assert value(m.fs.statejunction_1.outlet.pressure[0]) == pytest.approx(
        101325, rel=1e-6
    )
    assert value(m.fs.statejunction_1.outlet.temperature[0]) == pytest.approx(
        298.15, rel=1e-6
    )
    assert value(
        m.fs.statejunction_1.outlet.mole_frac_comp[0, "benzene"]
    ) == pytest.approx(0.6, rel=1e-6)
    assert value(
        m.fs.statejunction_1.outlet.mole_frac_comp[0, "toluene"]
    ) == pytest.approx(0.4, rel=1e-6)

    m.fs.statejunction_2 = StateJunction(property_package=m.fs.properties_2)

    DOF_initial = degrees_of_freedom(m)
    print("The initial degrees of freedom is: {0}".format(DOF_initial))
    assert DOF_initial == 0

    m.fs.statejunction_2.inlet.flow_vol.fix(10)  # m^3/s
    m.fs.statejunction_2.inlet.conc_mol_comp[0, "H2O"].fix(5000)  # mol/m^3
    m.fs.statejunction_2.inlet.conc_mol_comp[0, "NaOH"].fix(25)  # mol/m^3
    m.fs.statejunction_2.inlet.conc_mol_comp[0, "EthylAcetate"].fix(50)  # mol/m^3
    m.fs.statejunction_2.inlet.conc_mol_comp[0, "SodiumAcetate"].fix(100)  # mol/m^3
    m.fs.statejunction_2.inlet.conc_mol_comp[0, "Ethanol"].fix(200)  # mol/m^3
    m.fs.statejunction_2.inlet.pressure.fix(101325)  # Pa
    m.fs.statejunction_2.inlet.temperature.fix(298.15)  # K

    DOF_final = degrees_of_freedom(m)
    print("The final degrees of freedom is: {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.statejunction_2.initialize(outlvl=idaeslog.WARNING)
    m.fs.statejunction_2.report()

    assert value(m.fs.statejunction_2.outlet.flow_vol[0]) == pytest.approx(10, rel=1e-6)
    assert value(m.fs.statejunction_2.outlet.pressure[0]) == pytest.approx(
        101325, rel=1e-6
    )
    assert value(m.fs.statejunction_2.outlet.temperature[0]) == pytest.approx(
        298.15, rel=1e-6
    )
    assert value(m.fs.statejunction_2.outlet.conc_mol_comp[0, "H2O"]) == pytest.approx(
        5000, rel=1e-6
    )
    assert value(m.fs.statejunction_2.outlet.conc_mol_comp[0, "NaOH"]) == pytest.approx(
        25, rel=1e-6
    )
    assert value(
        m.fs.statejunction_2.outlet.conc_mol_comp[0, "EthylAcetate"]
    ) == pytest.approx(50, rel=1e-6)
    assert value(
        m.fs.statejunction_2.outlet.conc_mol_comp[0, "SodiumAcetate"]
    ) == pytest.approx(100, rel=1e-6)
    assert value(
        m.fs.statejunction_2.outlet.conc_mol_comp[0, "Ethanol"]
    ) == pytest.approx(200, rel=1e-6)
