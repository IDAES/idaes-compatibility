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
    check_optimal_termination,
    ConcreteModel,
    Constraint,
    Expression,
    Objective,
    maximize,
    Var,
    Set,
    TransformationFactory,
    value,
    exp,
    units as pyunits,
)
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.models.unit_models import Feed, SkeletonUnitModel, Mixer, Product
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers import get_solver
from pyomo.util.check_units import assert_units_consistent

from . import eg_h2o_ideal as thermo_props
from idaes.models.properties.modular_properties import GenericParameterBlock
from idaes.core.util.constants import Constants


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.thermo_params = GenericParameterBlock(**thermo_props.config_dict)

    m.fs.WATER = Feed(property_package=m.fs.thermo_params)
    m.fs.GLYCOL = Feed(property_package=m.fs.thermo_params)

    m.fs.M101 = Mixer(
        property_package=m.fs.thermo_params, inlet_list=["water_feed", "glycol_feed"]
    )

    m.fs.RETENTATE = Product(property_package=m.fs.thermo_params)
    m.fs.PERMEATE = Product(property_package=m.fs.thermo_params)

    m.fs.pervap = SkeletonUnitModel(dynamic=False)
    m.fs.pervap.comp_list = Set(initialize=["water", "ethylene_glycol"])
    m.fs.pervap.phase_list = Set(initialize=["Liq"])

    m.fs.pervap.flow_in = Var(
        m.fs.time,
        m.fs.pervap.phase_list,
        m.fs.pervap.comp_list,
        initialize=1.0,
        units=pyunits.mol / pyunits.s,
    )
    m.fs.pervap.temperature_in = Var(m.fs.time, initialize=298.15, units=pyunits.K)
    m.fs.pervap.pressure_in = Var(m.fs.time, initialize=101e3, units=pyunits.Pa)

    m.fs.pervap.perm_flow = Var(
        m.fs.time,
        m.fs.pervap.phase_list,
        m.fs.pervap.comp_list,
        initialize=1.0,
        units=pyunits.mol / pyunits.s,
    )
    m.fs.pervap.ret_flow = Var(
        m.fs.time,
        m.fs.pervap.phase_list,
        m.fs.pervap.comp_list,
        initialize=1.0,
        units=pyunits.mol / pyunits.s,
    )
    m.fs.pervap.temperature_out = Var(m.fs.time, initialize=298.15, units=pyunits.K)
    m.fs.pervap.pressure_out = Var(m.fs.time, initialize=101e3, units=pyunits.Pa)
    m.fs.pervap.vacuum = Var(m.fs.time, initialize=1.3e3, units=pyunits.Pa)

    inlet_dict = {
        "flow_mol_phase_comp": m.fs.pervap.flow_in,
        "temperature": m.fs.pervap.temperature_in,
        "pressure": m.fs.pervap.pressure_in,
    }
    retentate_dict = {
        "flow_mol_phase_comp": m.fs.pervap.ret_flow,
        "temperature": m.fs.pervap.temperature_out,
        "pressure": m.fs.pervap.pressure_out,
    }
    permeate_dict = {
        "flow_mol_phase_comp": m.fs.pervap.perm_flow,
        "temperature": m.fs.pervap.temperature_out,
        "pressure": m.fs.pervap.vacuum,
    }

    m.fs.pervap.add_ports(name="inlet", member_dict=inlet_dict)
    m.fs.pervap.add_ports(name="retentate", member_dict=retentate_dict)
    m.fs.pervap.add_ports(name="permeate", member_dict=permeate_dict)

    energy_activation_dict = {
        (0, "Liq", "water"): 51e3,
        (0, "Liq", "ethylene_glycol"): 53e3,
    }
    m.fs.pervap.energy_activation = Var(
        m.fs.time,
        m.fs.pervap.phase_list,
        m.fs.pervap.comp_list,
        initialize=energy_activation_dict,
        units=pyunits.J / pyunits.mol,
    )
    m.fs.pervap.energy_activation.fix()

    permeance_dict = {
        (0, "Liq", "water"): 5611320,
        (0, "Liq", "ethylene_glycol"): 22358.88,
    }  # calculated from literature data
    m.fs.pervap.permeance = Var(
        m.fs.time,
        m.fs.pervap.phase_list,
        m.fs.pervap.comp_list,
        initialize=permeance_dict,
        units=pyunits.mol / pyunits.s / pyunits.m**2,
    )
    m.fs.pervap.permeance.fix()

    m.fs.pervap.area = Var(m.fs.time, initialize=6, units=pyunits.m**2)
    m.fs.pervap.area.fix()

    latent_heat_dict = {
        (0, "Liq", "water"): 40.660e3,
        (0, "Liq", "ethylene_glycol"): 56.9e3,
    }
    m.fs.pervap.latent_heat_of_vaporization = Var(
        m.fs.time,
        m.fs.pervap.phase_list,
        m.fs.pervap.comp_list,
        initialize=latent_heat_dict,
        units=pyunits.J / pyunits.mol,
    )
    m.fs.pervap.latent_heat_of_vaporization.fix()
    m.fs.pervap.heat_duty = Var(
        m.fs.time, initialize=1, units=pyunits.J / pyunits.s
    )  # we will calculate this later

    def rule_permeate_flux(pervap, t, p, i):
        return pervap.permeate.flow_mol_phase_comp[t, p, i] / pervap.area[t] == (
            pervap.permeance[t, p, i]
            * exp(
                -pervap.energy_activation[t, p, i]
                / (Constants.gas_constant * pervap.inlet.temperature[t])
            )
        )

    m.fs.pervap.eq_permeate_flux = Constraint(
        m.fs.time,
        m.fs.pervap.phase_list,
        m.fs.pervap.comp_list,
        rule=rule_permeate_flux,
    )

    def rule_duty(pervap, t):
        return pervap.heat_duty[t] == sum(
            pervap.latent_heat_of_vaporization[t, p, i]
            * pervap.permeate.flow_mol_phase_comp[t, p, i]
            for p in pervap.phase_list
            for i in pervap.comp_list
        )

    m.fs.pervap.eq_duty = Constraint(m.fs.time, rule=rule_duty)

    def rule_retentate_flow(pervap, t, p, i):
        return pervap.retentate.flow_mol_phase_comp[t, p, i] == (
            pervap.inlet.flow_mol_phase_comp[t, p, i]
            - pervap.permeate.flow_mol_phase_comp[t, p, i]
        )

    m.fs.pervap.eq_retentate_flow = Constraint(
        m.fs.time,
        m.fs.pervap.phase_list,
        m.fs.pervap.comp_list,
        rule=rule_retentate_flow,
    )

    m.fs.s01 = Arc(source=m.fs.WATER.outlet, destination=m.fs.M101.water_feed)
    m.fs.s02 = Arc(source=m.fs.GLYCOL.outlet, destination=m.fs.M101.glycol_feed)
    m.fs.s03 = Arc(source=m.fs.M101.outlet, destination=m.fs.pervap.inlet)
    m.fs.s04 = Arc(source=m.fs.pervap.permeate, destination=m.fs.PERMEATE.inlet)
    m.fs.s05 = Arc(source=m.fs.pervap.retentate, destination=m.fs.RETENTATE.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    print(degrees_of_freedom(m))

    m.fs.WATER.outlet.flow_mol_phase_comp[0, "Liq", "water"].fix(0.34)  # mol/s
    m.fs.WATER.outlet.flow_mol_phase_comp[0, "Liq", "ethylene_glycol"].fix(
        1e-6
    )  # mol/s
    m.fs.WATER.outlet.temperature.fix(318.15)  # K
    m.fs.WATER.outlet.pressure.fix(101.325e3)  # Pa

    m.fs.GLYCOL.outlet.flow_mol_phase_comp[0, "Liq", "water"].fix(1e-6)  # mol/s
    m.fs.GLYCOL.outlet.flow_mol_phase_comp[0, "Liq", "ethylene_glycol"].fix(
        0.66
    )  # mol/s
    m.fs.GLYCOL.outlet.temperature.fix(318.15)  # K
    m.fs.GLYCOL.outlet.pressure.fix(101.325e3)  # Pa

    def rule_temp_out(pervap, t):
        return pervap.inlet.temperature[t] == pervap.retentate.temperature[t]

    m.fs.pervap.temperature_out_calculation = Constraint(m.fs.time, rule=rule_temp_out)

    def rule_pres_out(pervap, t):
        return pervap.inlet.pressure[t] == pervap.retentate.pressure[t]

    m.fs.pervap.pressure_out_calculation = Constraint(m.fs.time, rule=rule_pres_out)

    m.fs.PERMEATE.inlet.pressure.fix(1.3e3)

    assert degrees_of_freedom(m) == 0

    def my_initialize(unit, **kwargs):
        unit.inlet.flow_mol_phase_comp.fix()
        unit.inlet.pressure.fix()
        unit.inlet.temperature.fix()

        for t in m.fs.time:

            calculate_variable_from_constraint(
                unit.permeate.flow_mol_phase_comp[t, "Liq", "water"],
                unit.eq_permeate_flux[t, "Liq", "water"],
            )

            calculate_variable_from_constraint(
                unit.permeate.flow_mol_phase_comp[t, "Liq", "ethylene_glycol"],
                unit.eq_permeate_flux[t, "Liq", "ethylene_glycol"],
            )

            calculate_variable_from_constraint(unit.heat_duty[t], unit.eq_duty[t])

            calculate_variable_from_constraint(
                unit.retentate.flow_mol_phase_comp[t, "Liq", "water"],
                unit.eq_retentate_flow[t, "Liq", "water"],
            )

            calculate_variable_from_constraint(
                unit.retentate.flow_mol_phase_comp[t, "Liq", "ethylene_glycol"],
                unit.eq_retentate_flow[t, "Liq", "ethylene_glycol"],
            )

            calculate_variable_from_constraint(
                unit.retentate.temperature[t], unit.temperature_out_calculation[t]
            )

            calculate_variable_from_constraint(
                unit.retentate.pressure[t], unit.pressure_out_calculation[t]
            )

        assert degrees_of_freedom(unit) == 0
        if degrees_of_freedom(unit) == 0:
            res = solver.solve(unit, tee=True)
        unit.inlet.flow_mol_phase_comp.unfix()
        unit.inlet.temperature.unfix()
        unit.inlet.pressure.unfix()
        print("Custom initialization routine complete: ", res.solver.message)

    solver = get_solver()

    m.fs.WATER.initialize()
    propagate_state(m.fs.s01)

    m.fs.GLYCOL.initialize()
    propagate_state(m.fs.s02)

    m.fs.pervap.config.initializer = my_initialize
    my_initialize(m.fs.pervap)
    propagate_state(m.fs.s03)

    m.fs.PERMEATE.initialize()
    propagate_state(m.fs.s04)

    m.fs.RETENTATE.initialize()

    results = solver.solve(m, tee=True)

    m.fs.WATER.report()
    m.fs.GLYCOL.report()
    m.fs.PERMEATE.report()
    m.fs.RETENTATE.report()

    m.fs.inlet_water_frac = Expression(
        expr=(
            m.fs.pervap.inlet.flow_mol_phase_comp[0, "Liq", "water"]
            / sum(
                m.fs.pervap.inlet.flow_mol_phase_comp[0, "Liq", i]
                for i in m.fs.pervap.comp_list
            )
        )
    )
    m.fs.permeate_water_frac = Expression(
        expr=(
            m.fs.pervap.permeate.flow_mol_phase_comp[0, "Liq", "water"]
            / sum(
                m.fs.pervap.permeate.flow_mol_phase_comp[0, "Liq", i]
                for i in m.fs.pervap.comp_list
            )
        )
    )
    m.fs.separation_factor = Expression(
        expr=(m.fs.permeate_water_frac / (1 - m.fs.permeate_water_frac))
        / (m.fs.inlet_water_frac / (1 - m.fs.inlet_water_frac))
    )

    print(f"Inlet water mole fraction: {value(m.fs.inlet_water_frac)}")
    print(f"Permeate water mole fraction: {value(m.fs.permeate_water_frac)}")
    print(f"Separation factor: {value(m.fs.separation_factor)}")
    print(f"Condensation duty: {value(m.fs.pervap.heat_duty[0]/1000)} kW")
    print(
        f"Duty per mole water recovered: {value(m.fs.pervap.heat_duty[0]/(1000*m.fs.PERMEATE.inlet.flow_mol_phase_comp[0, 'Liq', 'water']*3600))} kW-h / mol"
    )

    assert check_optimal_termination(results)
    assert_units_consistent(m)

    assert value(
        m.fs.PERMEATE.inlet.flow_mol_phase_comp[0, "Liq", "water"]
    ) == pytest.approx(0.1426, rel=1e-3)
    assert value(
        m.fs.PERMEATE.inlet.flow_mol_phase_comp[0, "Liq", "ethylene_glycol"]
    ) == pytest.approx(0.0002667, rel=1e-3)
    assert value(
        m.fs.RETENTATE.inlet.flow_mol_phase_comp[0, "Liq", "water"]
    ) == pytest.approx(0.1974, rel=1e-3)
    assert value(
        m.fs.RETENTATE.inlet.flow_mol_phase_comp[0, "Liq", "ethylene_glycol"]
    ) == pytest.approx(0.6597, rel=1e-3)
    assert value(m.fs.separation_factor) == pytest.approx(1038, rel=1e-3)
    assert value(m.fs.pervap.heat_duty[0]) == pytest.approx(5813, rel=1e-3)

    m.fs.WATER.outlet.flow_mol_phase_comp[0, "Liq", "water"].unfix()
    m.fs.GLYCOL.outlet.flow_mol_phase_comp[0, "Liq", "ethylene_glycol"].unfix()
    m.fs.total_flow = Constraint(
        expr=m.fs.WATER.outlet.flow_mol_phase_comp[0, "Liq", "water"]
        + m.fs.GLYCOL.outlet.flow_mol_phase_comp[0, "Liq", "ethylene_glycol"]
        == 1 * pyunits.mol / pyunits.s
    )

    m.fs.sep_min = Constraint(expr=m.fs.separation_factor >= 100)
    m.fs.obj = Objective(expr=m.fs.inlet_water_frac, sense=maximize)

    results = solver.solve(m, tee=True)

    m.fs.WATER.report()
    m.fs.GLYCOL.report()
    m.fs.PERMEATE.report()
    m.fs.RETENTATE.report()

    print(f"Inlet water mole fraction: {value(m.fs.inlet_water_frac)}")
    print(f"Permeate water mole fraction: {value(m.fs.permeate_water_frac)}")
    print(f"Separation factor: {value(m.fs.separation_factor)}")
    print(f"Condensation duty: {value(m.fs.pervap.heat_duty[0]/1000)} kW")
    print(
        f"Duty per mole water recovered: {value(m.fs.pervap.heat_duty[0]/(1000*m.fs.PERMEATE.inlet.flow_mol_phase_comp[0, 'Liq', 'water']*3600))} kW-h / mol"
    )

    assert check_optimal_termination(results)
    assert_units_consistent(m)

    assert value(
        m.fs.PERMEATE.inlet.flow_mol_phase_comp[0, "Liq", "water"]
    ) == pytest.approx(0.1426, rel=1e-3)
    assert value(
        m.fs.PERMEATE.inlet.flow_mol_phase_comp[0, "Liq", "ethylene_glycol"]
    ) == pytest.approx(0.0002667, rel=1e-3)
    assert value(
        m.fs.RETENTATE.inlet.flow_mol_phase_comp[0, "Liq", "water"]
    ) == pytest.approx(0.6998, rel=1e-3)
    assert value(
        m.fs.RETENTATE.inlet.flow_mol_phase_comp[0, "Liq", "ethylene_glycol"]
    ) == pytest.approx(0.1573, rel=1e-3)
    assert value(m.fs.separation_factor) == pytest.approx(100.0, rel=1e-3)
    assert value(m.fs.pervap.heat_duty[0]) == pytest.approx(5813, rel=1e-3)
