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
    Constraint,
    Var,
    ConcreteModel,
    Expression,
    Objective,
    TransformationFactory,
    value,
    units as pyunits,
)
from pyomo.network import Arc
from pyomo.environ import TerminationCondition

from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.base.generic_reaction import (
    GenericReactionParameterBlock,
)
from idaes.models.unit_models import (
    Feed,
    Mixer,
    Compressor,
    Heater,
    EquilibriumReactor,
    Product,
)

from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state

# TODO: Backwardcompatibility test should not rely on community code
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop
from . import msr_reaction as reaction_props


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    thermo_props_config_dict = get_prop(components=["CH4", "H2O", "H2", "CO", "CO2"])
    m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)
    m.fs.reaction_params = GenericReactionParameterBlock(
        property_package=m.fs.thermo_params, **reaction_props.config_dict
    )

    m.fs.CH4 = Feed(property_package=m.fs.thermo_params)
    m.fs.H2O = Feed(property_package=m.fs.thermo_params)
    m.fs.PROD = Product(property_package=m.fs.thermo_params)
    m.fs.M101 = Mixer(
        property_package=m.fs.thermo_params, inlet_list=["methane_feed", "steam_feed"]
    )
    m.fs.H101 = Heater(
        property_package=m.fs.thermo_params,
        has_pressure_change=False,
        has_phase_equilibrium=False,
    )
    m.fs.C101 = Compressor(property_package=m.fs.thermo_params)
    m.fs.R101 = EquilibriumReactor(
        property_package=m.fs.thermo_params,
        reaction_package=m.fs.reaction_params,
        has_equilibrium_reactions=True,
        has_rate_reactions=False,
        has_heat_of_reaction=True,
        has_heat_transfer=True,
        has_pressure_change=False,
        has_phase_equilibrium=True,
    )

    m.fs.s01 = Arc(source=m.fs.CH4.outlet, destination=m.fs.M101.methane_feed)
    m.fs.s02 = Arc(source=m.fs.H2O.outlet, destination=m.fs.M101.steam_feed)
    m.fs.s03 = Arc(source=m.fs.M101.outlet, destination=m.fs.C101.inlet)
    m.fs.s04 = Arc(source=m.fs.C101.outlet, destination=m.fs.H101.inlet)
    m.fs.s05 = Arc(source=m.fs.H101.outlet, destination=m.fs.R101.inlet)
    m.fs.s06 = Arc(source=m.fs.R101.outlet, destination=m.fs.PROD.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    m.fs.hyd_prod = Expression(
        expr=pyunits.convert(
            m.fs.PROD.inlet.flow_mol[0]
            * m.fs.PROD.inlet.mole_frac_comp[0, "H2"]
            * m.fs.thermo_params.H2.mw,  # MW defined in properties as kg/mol
            to_units=pyunits.Mlb / pyunits.yr,
        )
    )  # converting kg/s to MM lb/year
    m.fs.cooling_cost = Expression(
        expr=2.12e-8 * (m.fs.R101.heat_duty[0])
    )  # the reaction is endothermic, so R101 duty is positive
    m.fs.heating_cost = Expression(
        expr=2.2e-7 * m.fs.H101.heat_duty[0]
    )  # the stream must be heated to T_rxn, so H101 duty is positive
    m.fs.compression_cost = Expression(
        expr=1.2e-6 * m.fs.C101.work_isentropic[0]
    )  # the stream must be pressurized, so the C101 work is positive
    m.fs.operating_cost = Expression(
        expr=(
            3600
            * 8000
            * (m.fs.heating_cost + m.fs.cooling_cost + m.fs.compression_cost)
        )
    )

    print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 20

    m.fs.CH4.outlet.mole_frac_comp[0, "CH4"].fix(1)
    m.fs.CH4.outlet.mole_frac_comp[0, "H2O"].fix(1e-5)
    m.fs.CH4.outlet.mole_frac_comp[0, "H2"].fix(1e-5)
    m.fs.CH4.outlet.mole_frac_comp[0, "CO"].fix(1e-5)
    m.fs.CH4.outlet.mole_frac_comp[0, "CO2"].fix(1e-5)
    m.fs.CH4.outlet.flow_mol.fix(75 * pyunits.mol / pyunits.s)
    m.fs.CH4.outlet.temperature.fix(298.15 * pyunits.K)
    m.fs.CH4.outlet.pressure.fix(1e5 * pyunits.Pa)

    m.fs.H2O.outlet.mole_frac_comp[0, "CH4"].fix(1e-5)
    m.fs.H2O.outlet.mole_frac_comp[0, "H2O"].fix(1)
    m.fs.H2O.outlet.mole_frac_comp[0, "H2"].fix(1e-5)
    m.fs.H2O.outlet.mole_frac_comp[0, "CO"].fix(1e-5)
    m.fs.H2O.outlet.mole_frac_comp[0, "CO2"].fix(1e-5)
    m.fs.H2O.outlet.flow_mol.fix(234 * pyunits.mol / pyunits.s)
    m.fs.H2O.outlet.temperature.fix(373.15 * pyunits.K)
    m.fs.H2O.outlet.pressure.fix(1e5 * pyunits.Pa)

    m.fs.C101.outlet.pressure.fix(pyunits.convert(2 * pyunits.bar, to_units=pyunits.Pa))
    m.fs.C101.efficiency_isentropic.fix(0.90)
    m.fs.H101.outlet.temperature.fix(500 * pyunits.K)

    m.fs.R101.conversion = Var(
        initialize=0.80, bounds=(0, 1), units=pyunits.dimensionless
    )  # fraction

    m.fs.R101.conv_constraint = Constraint(
        expr=m.fs.R101.conversion
        * m.fs.R101.inlet.flow_mol[0]
        * m.fs.R101.inlet.mole_frac_comp[0, "CH4"]
        == (
            m.fs.R101.inlet.flow_mol[0] * m.fs.R101.inlet.mole_frac_comp[0, "CH4"]
            - m.fs.R101.outlet.flow_mol[0] * m.fs.R101.outlet.mole_frac_comp[0, "CH4"]
        )
    )

    # TODO: Why is conversion being fixed in an equilibrium reactor?!?
    m.fs.R101.conversion.fix(0.80)

    print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0

    m.fs.CH4.initialize()
    propagate_state(arc=m.fs.s01)

    m.fs.H2O.initialize()
    propagate_state(arc=m.fs.s02)

    m.fs.M101.initialize()
    propagate_state(arc=m.fs.s03)

    m.fs.C101.initialize()
    propagate_state(arc=m.fs.s04)

    m.fs.H101.initialize()
    propagate_state(arc=m.fs.s05)

    m.fs.R101.initialize()
    propagate_state(arc=m.fs.s06)

    m.fs.PROD.initialize()

    solver = get_solver()
    results = solver.solve(m, tee=True)

    assert results.solver.termination_condition == TerminationCondition.optimal

    print(f"operating cost = ${value(m.fs.operating_cost)/1e6:0.3f} million per year")

    assert value(m.fs.operating_cost) / 1e6 == pytest.approx(45.933, rel=1e-3)

    m.fs.R101.report()

    print()
    print(f"Conversion achieved = {value(m.fs.R101.conversion):.1%}")

    assert value(m.fs.R101.conversion) == pytest.approx(0.800, rel=1e-3)
    assert value(m.fs.R101.heat_duty[0]) / 1e6 == pytest.approx(27.605, rel=1e-3)
    assert value(m.fs.R101.outlet.temperature[0]) / 1e2 == pytest.approx(
        8.6856, rel=1e-3
    )

    m.fs.objective = Objective(expr=m.fs.operating_cost)

    m.fs.R101.conversion.fix(0.90)

    m.fs.C101.outlet.pressure.unfix()
    m.fs.C101.outlet.pressure[0].setlb(
        pyunits.convert(2 * pyunits.bar, to_units=pyunits.Pa)
    )  # pressurize to at least 2 bar
    m.fs.C101.outlet.pressure[0].setub(
        pyunits.convert(10 * pyunits.bar, to_units=pyunits.Pa)
    )  # at most, pressurize to 10 bar

    m.fs.H101.outlet.temperature.unfix()
    m.fs.H101.heat_duty[0].setlb(
        0 * pyunits.J / pyunits.s
    )  # outlet temperature is equal to or greater than inlet temperature
    m.fs.H101.outlet.temperature[0].setub(1000 * pyunits.K)  # at most, heat to 1000 K

    assert degrees_of_freedom(m) == 2

    results = solver.solve(m, tee=True)

    assert results.solver.termination_condition == TerminationCondition.optimal

    print(f"operating cost = ${value(m.fs.operating_cost)/1e6:0.3f} million per year")

    print()
    print("Compressor results")

    m.fs.C101.report()

    print()
    print("Heater results")

    m.fs.H101.report()

    print()
    print("Equilibrium reactor results")

    m.fs.R101.report()

    assert value(m.fs.operating_cost) / 1e6 == pytest.approx(43.309, rel=1e-3)

    print("Optimal Values")
    print()

    print(f"C101 outlet pressure = {value(m.fs.C101.outlet.pressure[0])/1E6:0.3f} MPa")
    print()

    print(f"C101 outlet temperature = {value(m.fs.C101.outlet.temperature[0]):0.3f} K")
    print()

    print(f"H101 outlet temperature = {value(m.fs.H101.outlet.temperature[0]):0.3f} K")

    print()
    print(f"R101 outlet temperature = {value(m.fs.R101.outlet.temperature[0]):0.3f} K")

    print()
    print(f"Hydrogen produced = {value(m.fs.hyd_prod):0.3f} MM lb/year")

    print()
    print(f"Conversion achieved = {value(m.fs.R101.conversion):.1%}")

    assert value(m.fs.C101.outlet.pressure[0]) / 1e6 == pytest.approx(0.2000, rel=1e-3)
    assert value(m.fs.C101.outlet.temperature[0]) / 100 == pytest.approx(
        4.23345, rel=1e-3
    )
    assert value(m.fs.H101.outlet.temperature[0]) / 100 == pytest.approx(
        4.23345, rel=1e-3
    )
    assert value(m.fs.R101.outlet.temperature[0]) / 100 == pytest.approx(
        9.10044, rel=1e-3
    )
    assert value(m.fs.hyd_prod) == pytest.approx(33.648, rel=1e-3)
    assert value(m.fs.R101.conversion) * 100 == pytest.approx(90.0, rel=1e-3)
