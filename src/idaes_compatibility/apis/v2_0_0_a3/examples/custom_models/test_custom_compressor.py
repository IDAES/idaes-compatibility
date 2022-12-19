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
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.util.check_units import assert_units_consistent

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
from .methanol_param_VLE import PhysicalParameterBlock
from idaes.core.util.misc import add_object_reference


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
            has_heat_of_reaction=False, has_heat_transfer=False, has_work_transfer=True
        )

    def add_isentropic(unit, name, config):
        unit.pressure_ratio = pe.Var(initialize=1.0, bounds=(1, None))
        cons = pe.ConstraintList()
        setattr(unit, name, cons)
        inlet = unit.control_volume.properties_in[0.0]
        outlet = unit.control_volume.properties_out[0.0]
        gamma = inlet.params.gamma
        cons.add(inlet.pressure * unit.pressure_ratio == outlet.pressure)
        cons.add(
            outlet.temperature
            == (
                inlet.temperature
                + 1
                / config.compressor_efficiency
                * (
                    inlet.temperature * unit.pressure_ratio ** ((gamma - 1) / gamma)
                    - inlet.temperature
                )
            )
        )

    def make_compressor_config_block(config):
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
                default=MomentumBalanceType.none, domain=In([MomentumBalanceType.none])
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
        config.declare("compressor_efficiency", ConfigValue(default=0.75, domain=float))

    @declare_process_block_class("IdealGasIsentropicCompressor")
    class IdealGasIsentropicCompressorData(UnitModelBlockData):
        CONFIG = UnitModelBlockData.CONFIG()
        make_compressor_config_block(CONFIG)

        def build(self):
            super(IdealGasIsentropicCompressorData, self).build()

            make_control_volume(self, "control_volume", self.config)
            add_isentropic(self, "isentropic", self.config)

            self.add_inlet_port()
            self.add_outlet_port()

            add_object_reference(self, "work", self.control_volume.work[0.0])

    m = pe.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = props = PhysicalParameterBlock(
        Cp=0.038056, valid_phase="Vap"
    )  # MJ/kmol-K

    m.fs.compressor = IdealGasIsentropicCompressor(
        property_package=props, has_phase_equilibrium=False
    )
    m.fs.compressor.inlet.flow_mol.fix(1)  # kmol
    m.fs.compressor.inlet.mole_frac_comp[0, "CH3OH"].fix(0.25)
    m.fs.compressor.inlet.mole_frac_comp[0, "CH4"].fix(0.25)
    m.fs.compressor.inlet.mole_frac_comp[0, "H2"].fix(0.25)
    m.fs.compressor.inlet.mole_frac_comp[0, "CO"].fix(0.25)
    m.fs.compressor.inlet.pressure.fix(0.14)  # MPa
    m.fs.compressor.inlet.temperature.fix(2.9315)  # hK [100K]
    m.fs.compressor.outlet.pressure.fix(0.56)  # MPa

    opt = pe.SolverFactory("ipopt")
    opt.options["linear_solver"] = "ma27"
    res = opt.solve(m, tee=True)
    print(res.solver.termination_condition)
    m.fs.compressor.outlet.display()
    print("work: ", round(m.fs.compressor.work.value, 2), " MJ")  # MJ

    assert_units_consistent(m)

    assert res.solver.termination_condition == pe.TerminationCondition.optimal
    assert pe.value(m.fs.compressor.work) == pytest.approx(5.2616, abs=1e-2)
