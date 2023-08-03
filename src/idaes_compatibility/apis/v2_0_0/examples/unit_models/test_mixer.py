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
from pyomo.environ import ConcreteModel, SolverFactory, value
from pyomo.opt import TerminationCondition, SolverStatus

from idaes.core import FlowsheetBlock
from idaes.models.unit_models import Mixer, MomentumMixingType
import idaes.logger as idaeslog
from idaes.models.properties.activity_coeff_models import BTX_activity_coeff_VLE
from idaes.core.util.model_statistics import degrees_of_freedom


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = BTX_activity_coeff_VLE.BTXParameterBlock(
        valid_phase="Liq", activity_coeff_model="Ideal"
    )
    m.fs.mixer_1 = Mixer(
        property_package=m.fs.properties,
        num_inlets=2,
        momentum_mixing_type=MomentumMixingType.minimize,
    )

    DOF_initial = degrees_of_freedom(m)
    print("The initial degrees of freedom is: {0}".format(DOF_initial))
    assert DOF_initial == 10

    m.fs.mixer_1.inlet_1.flow_mol.fix(100)  # converting to mol/s as unit basis is mol/s
    m.fs.mixer_1.inlet_1.mole_frac_comp[0, "benzene"].fix(1)
    m.fs.mixer_1.inlet_1.mole_frac_comp[0, "toluene"].fix(0)
    m.fs.mixer_1.inlet_1.pressure.fix(101325)  # Pa
    m.fs.mixer_1.inlet_1.temperature.fix(353)  # K

    m.fs.mixer_1.inlet_2.flow_mol.fix(100)  # converting to mol/s as unit basis is mol/s
    m.fs.mixer_1.inlet_2.mole_frac_comp[0, "benzene"].fix(0)
    m.fs.mixer_1.inlet_2.mole_frac_comp[0, "toluene"].fix(1)
    m.fs.mixer_1.inlet_2.pressure.fix(202650)  # Pa
    m.fs.mixer_1.inlet_2.temperature.fix(356)  # K

    DOF_final = degrees_of_freedom(m)
    print("The final degrees of freedom is: {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.mixer_1.initialize(outlvl=idaeslog.WARNING)

    opt = SolverFactory("ipopt")
    result = opt.solve(m, tee=True)

    assert result.solver.termination_condition == TerminationCondition.optimal
    assert result.solver.status == SolverStatus.ok

    m.fs.mixer_1.report()

    molflow_1 = m.fs.mixer_1.inlet_1.flow_mol[0].value
    molflow_2 = m.fs.mixer_1.inlet_2.flow_mol[0].value
    benzene_1 = m.fs.mixer_1.inlet_1.mole_frac_comp[0, "benzene"].value
    toluene_2 = m.fs.mixer_1.inlet_2.mole_frac_comp[0, "toluene"].value

    total_enthalpy_in1 = (
        molflow_1 * m.fs.mixer_1.inlet_1_state[0].enth_mol_phase["Liq"].value
    )
    total_enthalpy_in2 = (
        molflow_2 * m.fs.mixer_1.inlet_2_state[0].enth_mol_phase["Liq"].value
    )

    molar_enthalpy_out = m.fs.mixer_1.mixed_state[0].enth_mol_phase["Liq"].value
    mole_flow_out = m.fs.mixer_1.outlet.flow_mol[0].value
    total_enthalpy_out = mole_flow_out * molar_enthalpy_out

    assert mole_flow_out == pytest.approx(200, abs=1e-3)
    assert m.fs.mixer_1.outlet.mole_frac_comp[0, "benzene"].value == pytest.approx(
        0.5, abs=1e-3
    )
    assert m.fs.mixer_1.outlet.mole_frac_comp[0, "toluene"].value == pytest.approx(
        0.5, abs=1e-3
    )
    assert m.fs.mixer_1.outlet.temperature[0].value == pytest.approx(354.61, abs=1e-2)
    assert m.fs.mixer_1.outlet.pressure[0].value == pytest.approx(101325, abs=1)
    assert (
        total_enthalpy_out - total_enthalpy_in1 - total_enthalpy_in2
        == pytest.approx(0, abs=1e-2)
    )

    m.fs.mixer_2 = Mixer(
        property_package=m.fs.properties,
        inlet_list=["benzene_inlet", "toluene_inlet"],
        momentum_mixing_type=MomentumMixingType.equality,
    )
    DOF_init = degrees_of_freedom(m.fs.mixer_2)
    print("The initial degrees of freedom is: {0}".format(DOF_init))
    assert DOF_init == 9

    m.fs.mixer_2.benzene_inlet.flow_mol.fix(
        100
    )  # converting to mol/s as unit basis is mol/s
    m.fs.mixer_2.benzene_inlet.mole_frac_comp[0, "benzene"].fix(1)
    m.fs.mixer_2.benzene_inlet.mole_frac_comp[0, "toluene"].fix(0)
    m.fs.mixer_2.benzene_inlet.pressure.fix(
        101325
    )  # Pa , Another option is m1.fs.mixer2.inlet2.pressure.fix(202650)
    m.fs.mixer_2.benzene_inlet.temperature.fix(353)  # K

    m.fs.mixer_2.toluene_inlet.flow_mol.fix(
        100
    )  # converting to mol/s as unit basis is mol/s
    m.fs.mixer_2.toluene_inlet.mole_frac_comp[0, "benzene"].fix(0)
    m.fs.mixer_2.toluene_inlet.mole_frac_comp[0, "toluene"].fix(1)
    m.fs.mixer_2.toluene_inlet.temperature.fix(356)  # K

    DOF_final = degrees_of_freedom(m.fs.mixer_2)
    print("The final degrees of freedom is: {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.mixer_2.initialize(outlvl=idaeslog.WARNING)
    opt = SolverFactory("ipopt")
    result = opt.solve(m.fs.mixer_2, tee=True)

    assert result.solver.termination_condition == TerminationCondition.optimal
    assert result.solver.status == SolverStatus.ok

    m.fs.mixer_2.report()

    assert m.fs.mixer_2.outlet.flow_mol[0].value == pytest.approx(200, abs=1e-2)
    assert m.fs.mixer_2.outlet.mole_frac_comp[0, "benzene"].value == pytest.approx(
        0.5, abs=1e-3
    )
    assert m.fs.mixer_2.outlet.mole_frac_comp[0, "toluene"].value == pytest.approx(
        0.5, abs=1e-3
    )
    assert m.fs.mixer_2.outlet.temperature[0].value == pytest.approx(354.61, abs=1e-2)
    assert m.fs.mixer_2.outlet.pressure[0].value == pytest.approx(101325, abs=1)
