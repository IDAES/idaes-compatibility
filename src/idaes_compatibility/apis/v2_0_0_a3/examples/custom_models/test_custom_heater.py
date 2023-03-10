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

import pyomo.environ as pe
from pyomo.util.check_units import assert_units_consistent

from pyomo.common.config import ConfigBlock, ConfigValue, In
from idaes.core import (
    ControlVolume0DBlock,
    declare_process_block_class,
    EnergyBalanceType,
    MomentumBalanceType,
    MaterialBalanceType,
    UnitModelBlockData,
    useDefault,
    FlowsheetBlock,
)
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.misc import add_object_reference
from .methanol_param_VLE import PhysicalParameterBlock


def test_example():
    def make_control_volume(unit, name, config):
        if config.dynamic is not False:
            raise ValueError("IdealGasIsentropcCompressor does not support dynamics")
        if config.has_holdup is not False:
            raise ValueError("IdealGasIsentropcCompressor does not support holdup")

        control_volume = ControlVolume0DBlock(
            property_package=config.property_package,
            property_package_args=config.property_package_args,
        )

        setattr(unit, name, control_volume)

        control_volume.add_state_blocks(
            has_phase_equilibrium=config.has_phase_equilibrium
        )
        control_volume.add_material_balances(
            balance_type=config.material_balance_type,
            has_phase_equilibrium=config.has_phase_equilibrium,
        )
        control_volume.add_total_enthalpy_balances(
            has_heat_of_reaction=False, has_heat_transfer=True, has_work_transfer=False
        )
        control_volume.add_total_pressure_balances(has_pressure_change=False)

    def make_config_block(config):
        config.declare(
            "material_balance_type",
            ConfigValue(
                default=MaterialBalanceType.componentPhase,
                domain=In(MaterialBalanceType),
            ),
        )
        config.declare(
            "energy_balance_type",
            ConfigValue(
                default=EnergyBalanceType.enthalpyTotal,
                domain=In([EnergyBalanceType.enthalpyTotal]),
            ),
        )
        config.declare(
            "momentum_balance_type",
            ConfigValue(
                default=MomentumBalanceType.pressureTotal,
                domain=In([MomentumBalanceType.pressureTotal]),
            ),
        )
        config.declare(
            "has_phase_equilibrium", ConfigValue(default=False, domain=In([False]))
        )
        config.declare(
            "has_pressure_change", ConfigValue(default=False, domain=In([False]))
        )
        config.declare(
            "property_package",
            ConfigValue(default=useDefault, domain=is_physical_parameter_block),
        )
        config.declare("property_package_args", ConfigBlock(implicit=True))

    @declare_process_block_class("Heater")
    class HeaterData(UnitModelBlockData):
        CONFIG = UnitModelBlockData.CONFIG()
        make_config_block(CONFIG)

        def build(self):
            super(HeaterData, self).build()

            make_control_volume(self, "control_volume", self.config)

            self.add_inlet_port()
            self.add_outlet_port()

            add_object_reference(self, "heat", self.control_volume.heat[0.0])

    m = pe.ConcreteModel()
    m.fs = fs = FlowsheetBlock(dynamic=False)
    fs.properties = props = PhysicalParameterBlock(
        Cp=0.038056, valid_phase="Vap"
    )  # MJ/kmol-K

    fs.heater = Heater(property_package=props, has_phase_equilibrium=False)
    fs.heater.inlet.flow_mol.fix(1)  # kmol
    fs.heater.inlet.mole_frac_comp[0, "CH3OH"].fix(0.25)
    fs.heater.inlet.mole_frac_comp[0, "CH4"].fix(0.25)
    fs.heater.inlet.mole_frac_comp[0, "H2"].fix(0.25)
    fs.heater.inlet.mole_frac_comp[0, "CO"].fix(0.25)
    fs.heater.inlet.pressure.fix(0.1)  # MPa
    fs.heater.inlet.temperature.fix(3)  # hK [100K]
    fs.heater.heat.fix(5)  # MJ

    opt = pe.SolverFactory("ipopt")
    res = opt.solve(m, tee=False)
    print(res.solver.termination_condition)
    fs.heater.outlet.display()

    assert_units_consistent(m)

    assert res.solver.termination_condition == pe.TerminationCondition.optimal
    assert pe.value(fs.heater.outlet.temperature[0]) == pytest.approx(4.3138, abs=1e-3)
