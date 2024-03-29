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
from idaes.models.properties.modular_properties import (
    GenericParameterBlock,
    GenericReactionParameterBlock,
)
from idaes.models.unit_models import Feed, Mixer, Heater, PFR, Product

from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from . import egprod_ideal as thermo_props
from . import egprod_reaction as reaction_props


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.thermo_params = GenericParameterBlock(**thermo_props.config_dict)
    m.fs.reaction_params = GenericReactionParameterBlock(
        property_package=m.fs.thermo_params, **reaction_props.config_dict
    )

    m.fs.OXIDE = Feed(property_package=m.fs.thermo_params)
    m.fs.ACID = Feed(property_package=m.fs.thermo_params)
    m.fs.PROD = Product(property_package=m.fs.thermo_params)
    m.fs.M101 = Mixer(
        property_package=m.fs.thermo_params,
        inlet_list=["reagent_feed", "catalyst_feed"],
    )
    m.fs.H101 = Heater(
        property_package=m.fs.thermo_params,
        has_pressure_change=False,
        has_phase_equilibrium=False,
    )

    m.fs.R101 = PFR(
        property_package=m.fs.thermo_params,
        reaction_package=m.fs.reaction_params,
        has_equilibrium_reactions=False,
        has_heat_of_reaction=True,
        has_heat_transfer=True,
        has_pressure_change=False,
        transformation_method="dae.finite_difference",
        transformation_scheme="BACKWARD",
        finite_elements=20,
    )

    m.fs.s01 = Arc(source=m.fs.OXIDE.outlet, destination=m.fs.M101.reagent_feed)
    m.fs.s02 = Arc(source=m.fs.ACID.outlet, destination=m.fs.M101.catalyst_feed)
    m.fs.s03 = Arc(source=m.fs.M101.outlet, destination=m.fs.H101.inlet)
    m.fs.s04 = Arc(source=m.fs.H101.outlet, destination=m.fs.R101.inlet)
    m.fs.s05 = Arc(source=m.fs.R101.outlet, destination=m.fs.PROD.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    m.fs.eg_prod = Expression(
        expr=pyunits.convert(
            m.fs.PROD.inlet.flow_mol_phase_comp[0, "Liq", "ethylene_glycol"]
            * m.fs.thermo_params.ethylene_glycol.mw,  # MW defined in properties as kg/mol
            to_units=pyunits.Mlb / pyunits.yr,
        )
    )  # converting kg/s to MM lb/year
    m.fs.cooling_cost = Expression(
        expr=2.12e-8
        * (
            -sum(
                m.fs.R101.heat_duty[0, x]
                for x in m.fs.R101.control_volume.length_domain
            )
        )
    )  # the reaction is exothermic, so R101 duty is negative
    m.fs.heating_cost = Expression(
        expr=2.2e-7 * m.fs.H101.heat_duty[0]
    )  # the stream must be heated to T_rxn, so H101 duty is positive
    m.fs.operating_cost = Expression(
        expr=(3600 * 8000 * (m.fs.heating_cost + m.fs.cooling_cost))
    )

    print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 35

    m.fs.OXIDE.outlet.flow_mol_phase_comp[0, "Liq", "ethylene_oxide"].fix(
        58.0 * pyunits.mol / pyunits.s
    )
    m.fs.OXIDE.outlet.flow_mol_phase_comp[0, "Liq", "water"].fix(
        39.6 * pyunits.mol / pyunits.s
    )  # calculated from 16.1 mol EO / cudm in stream
    m.fs.OXIDE.outlet.flow_mol_phase_comp[0, "Liq", "sulfuric_acid"].fix(
        1e-5 * pyunits.mol / pyunits.s
    )
    m.fs.OXIDE.outlet.flow_mol_phase_comp[0, "Liq", "ethylene_glycol"].fix(
        1e-5 * pyunits.mol / pyunits.s
    )
    m.fs.OXIDE.outlet.temperature.fix(298.15 * pyunits.K)
    m.fs.OXIDE.outlet.pressure.fix(1e5 * pyunits.Pa)

    m.fs.ACID.outlet.flow_mol_phase_comp[0, "Liq", "ethylene_oxide"].fix(
        1e-5 * pyunits.mol / pyunits.s
    )
    m.fs.ACID.outlet.flow_mol_phase_comp[0, "Liq", "water"].fix(
        200 * pyunits.mol / pyunits.s
    )
    m.fs.ACID.outlet.flow_mol_phase_comp[0, "Liq", "sulfuric_acid"].fix(
        0.334 * pyunits.mol / pyunits.s
    )  # calculated from 0.9 wt% SA in stream
    m.fs.ACID.outlet.flow_mol_phase_comp[0, "Liq", "ethylene_glycol"].fix(
        1e-5 * pyunits.mol / pyunits.s
    )
    m.fs.ACID.outlet.temperature.fix(298.15 * pyunits.K)
    m.fs.ACID.outlet.pressure.fix(1e5 * pyunits.Pa)

    m.fs.H101.outlet.temperature.fix(328.15 * pyunits.K)

    m.fs.R101.conversion = Var(
        bounds=(0, 1), initialize=0.80, units=pyunits.dimensionless
    )  # fraction

    m.fs.R101.conv_constraint = Constraint(
        expr=m.fs.R101.conversion
        * m.fs.R101.inlet.flow_mol_phase_comp[0, "Liq", "ethylene_oxide"]
        == (
            m.fs.R101.inlet.flow_mol_phase_comp[0, "Liq", "ethylene_oxide"]
            - m.fs.R101.outlet.flow_mol_phase_comp[0, "Liq", "ethylene_oxide"]
        )
    )

    for x in m.fs.R101.control_volume.length_domain:
        if x == 0:
            continue
        m.fs.R101.control_volume.properties[0, x].temperature.fix(
            328.15 * pyunits.K
        )  # equal inlet reactor temperature

    m.fs.R101.conversion.fix(0.5)
    m.fs.R101.length.fix(1 * pyunits.m)
    m.fs.R101.heat_duty.setub(
        0 * pyunits.J / pyunits.m / pyunits.s
    )  # heat duty is only used for cooling

    print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0

    m.fs.OXIDE.initialize()
    propagate_state(arc=m.fs.s01)

    m.fs.ACID.initialize()
    propagate_state(arc=m.fs.s01)

    m.fs.M101.initialize()
    propagate_state(arc=m.fs.s03)

    m.fs.H101.initialize()
    propagate_state(arc=m.fs.s04)

    m.fs.R101.initialize()
    propagate_state(arc=m.fs.s05)

    m.fs.PROD.initialize()

    solver = get_solver()
    results = solver.solve(m, tee=True)

    assert results.solver.termination_condition == TerminationCondition.optimal

    print(f"operating cost = ${value(m.fs.operating_cost)/1e6:0.3f} million per year")

    assert value(m.fs.operating_cost) / 1e6 == pytest.approx(43.175798, abs=1e-3)

    m.fs.R101.report()

    print()
    print(f"Conversion achieved = {value(m.fs.R101.conversion):.1%}")
    print()
    print(
        f"Total heat duty required = "
        f"{value(sum(m.fs.R101.heat_duty[0, x] for x in m.fs.R101.control_volume.length_domain))/1e6:0.3f}"
        f" MJ"
    )
    print()
    print(f"Tube area required = {value(m.fs.R101.area):0.3f} m^2")
    print()
    print(f"Tube length required = {value(m.fs.R101.length):0.3f} m")
    print()
    print(f"Tube volume required = {value(m.fs.R101.volume):0.3f} m^3")

    assert value(m.fs.R101.conversion) == pytest.approx(0.5000, abs=1e-3)
    assert value(m.fs.R101.area) == pytest.approx(1.1490, abs=1e-3)
    assert value(
        sum(m.fs.R101.heat_duty[0, x] for x in m.fs.R101.control_volume.length_domain)
    ) / 1e6 == pytest.approx(-70.708, abs=1e-3)
    assert value(m.fs.R101.outlet.temperature[0]) / 1e2 == pytest.approx(
        3.2815, abs=1e-3
    )

    m.fs.objective = Objective(expr=m.fs.operating_cost)
    m.fs.eg_prod_con = Constraint(
        expr=m.fs.eg_prod >= 200 * pyunits.Mlb / pyunits.yr
    )  # MM lb/year
    m.fs.R101.conversion.fix(0.90)
    m.fs.R101.volume.setlb(0 * pyunits.m**3)
    m.fs.R101.volume.setub(pyunits.convert(5000 * pyunits.gal, to_units=pyunits.m**3))
    m.fs.R101.length.unfix()
    m.fs.R101.length.setlb(0 * pyunits.m)
    m.fs.R101.length.setub(5 * pyunits.m)
    m.fs.H101.outlet.temperature.unfix()
    m.fs.H101.outlet.temperature[0].setlb(328.15 * pyunits.K)
    m.fs.H101.outlet.temperature[0].setub(
        470.45 * pyunits.K
    )  # highest component boiling point (ethylene glycol)

    for x in m.fs.R101.control_volume.length_domain:
        if x == 0:
            continue
        m.fs.R101.control_volume.properties[
            0, x
        ].temperature.unfix()  # allow for temperature change in each finite element

    assert degrees_of_freedom(m) == 22

    results = solver.solve(m, tee=True)
    assert results.solver.termination_condition == TerminationCondition.optimal

    print(f"operating cost = ${value(m.fs.operating_cost)/1e6:0.3f} million per year")

    print()
    print("Heater results")

    m.fs.H101.report()

    print()
    print("PFR reactor results")

    m.fs.R101.report()

    assert value(m.fs.operating_cost) / 1e6 == pytest.approx(15.538911, abs=1e-3)
    assert value(m.fs.R101.area) == pytest.approx(2.7874, abs=1e-3)

    print("Optimal Values")
    print()

    print(f"H101 outlet temperature = {value(m.fs.H101.outlet.temperature[0]):0.3f} K")

    print()
    print(
        "Total heat duty required = ",
        value(
            sum(
                m.fs.R101.heat_duty[0, x]
                for x in m.fs.R101.control_volume.length_domain
            )
        )
        / 1e6,
        "MJ",
    )
    print()
    print(f"Tube area required = {value(m.fs.R101.area):0.3f} m^2")

    print()
    print(f"Tube length required = {value(m.fs.R101.length):0.3f} m")

    print()
    print(
        f"Assuming a 20% design factor for reactor volume,"
        f"total CSTR volume required = {value(1.2*m.fs.R101.volume):0.3f}"
        f" m^3 = {value(pyunits.convert(1.2*m.fs.R101.volume, to_units=pyunits.gal)):0.3f} gal"
    )

    print()
    print(f"Ethylene glycol produced = {value(m.fs.eg_prod):0.3f} MM lb/year")

    print()
    print(f"Conversion achieved = {value(m.fs.R101.conversion):.1%}")

    assert value(m.fs.H101.outlet.temperature[0]) / 100 == pytest.approx(
        3.2815, abs=1e-3
    )
    assert value(
        sum(m.fs.R101.heat_duty[0, x] for x in m.fs.R101.control_volume.length_domain)
    ) / 1e6 == pytest.approx(-25.443, abs=1e-3)
    assert value(m.fs.R101.area) == pytest.approx(2.7874, abs=1e-3)
    assert value(m.fs.R101.control_volume.length) == pytest.approx(5.000, abs=1e-3)
    assert value(m.fs.R101.volume * 1.2) == pytest.approx(16.725, abs=1e-3)
    assert value(m.fs.eg_prod) == pytest.approx(225.415, abs=1e-3)
    assert value(m.fs.R101.conversion) * 100 == pytest.approx(90.000, abs=1e-3)
