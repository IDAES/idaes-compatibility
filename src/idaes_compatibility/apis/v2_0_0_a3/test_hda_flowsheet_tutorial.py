##################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
"""
Backward compatibility test for HDA flowsheet tutorial
"""
import pytest

from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Objective,
                           SolverFactory,
                           TerminationCondition,
                           TransformationFactory,
                           value)
from pyomo.network import Arc, SequentialDecomposition

from idaes.core import FlowsheetBlock
from idaes.models.unit_models import (PressureChanger,
                                      Mixer,
                                      Separator as Splitter,
                                      Heater,
                                      StoichiometricReactor,
                                      Flash)

from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.tables import create_stream_table_dataframe, stream_table_dataframe_to_string
import idaes.logger as idaeslog

from . import hda_ideal_VLE as thermo_props
from . import hda_reaction as reaction_props


def test_tutorial():
    ## Constructing the Flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.thermo_params = thermo_props.HDAParameterBlock()
    m.fs.reaction_params = reaction_props.HDAReactionParameterBlock(
            property_package=m.fs.thermo_params)


    m.fs.M101 = Mixer(property_package=m.fs.thermo_params,
                      inlet_list=["toluene_feed", "hydrogen_feed", "vapor_recycle"])
    m.fs.H101 = Heater(property_package=m.fs.thermo_params,
                       has_pressure_change=False,
                       has_phase_equilibrium=True)
    m.fs.R101 = StoichiometricReactor(
                property_package=m.fs.thermo_params,
                reaction_package=m.fs.reaction_params,
                has_heat_of_reaction=True,
                has_heat_transfer=True,
                has_pressure_change=False)
    m.fs.F101 = Flash(property_package=m.fs.thermo_params,
                      has_heat_transfer=True,
                      has_pressure_change=True)
    m.fs.S101 = Splitter(property_package=m.fs.thermo_params,
                         ideal_separation=False,
                         outlet_list=["purge", "recycle"])
    m.fs.C101 = PressureChanger(
                property_package=m.fs.thermo_params,
                compressor=True,
                thermodynamic_assumption=ThermodynamicAssumption.isothermal)
    m.fs.F102 = Flash(property_package=m.fs.thermo_params,
                      has_heat_transfer=True,
                      has_pressure_change=True)

    # ## Connecting Unit Models using Arcs
    m.fs.s03 = Arc(source=m.fs.M101.outlet, destination=m.fs.H101.inlet)
    m.fs.s04 = Arc(source=m.fs.H101.outlet, destination=m.fs.R101.inlet)
    m.fs.s05 = Arc(source=m.fs.R101.outlet, destination=m.fs.F101.inlet)
    m.fs.s06 = Arc(source=m.fs.F101.vap_outlet, destination=m.fs.S101.inlet)
    m.fs.s08 = Arc(source=m.fs.S101.recycle, destination=m.fs.C101.inlet)
    m.fs.s09 = Arc(source=m.fs.C101.outlet,
                   destination=m.fs.M101.vapor_recycle)
    m.fs.s10 = Arc(source=m.fs.F101.liq_outlet, destination=m.fs.F102.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    ## Adding expressions to compute purity and operating costs
    m.fs.purity = Expression(
            expr=m.fs.F102.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"] /
            (m.fs.F102.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"]
             + m.fs.F102.vap_outlet.flow_mol_phase_comp[0, "Vap", "toluene"]))
    m.fs.cooling_cost = Expression(expr=0.212e-7 * (-m.fs.F101.heat_duty[0]) +
                                       0.212e-7 * (-m.fs.R101.heat_duty[0]))
    m.fs.heating_cost = Expression(expr=2.2e-7 * m.fs.H101.heat_duty[0] +
                                       1.9e-7 * m.fs.F102.heat_duty[0])
    m.fs.operating_cost = Expression(expr=(3600 * 24 * 365 *
                                               (m.fs.heating_cost +
                                                m.fs.cooling_cost)))

    ## Fixing feed conditions
    print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 29

    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Vap", "benzene"].fix(1e-5)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Vap", "toluene"].fix(1e-5)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Vap", "hydrogen"].fix(1e-5)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Vap", "methane"].fix(1e-5)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "benzene"].fix(1e-5)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "toluene"].fix(0.30)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "hydrogen"].fix(1e-5)
    m.fs.M101.toluene_feed.flow_mol_phase_comp[0, "Liq", "methane"].fix(1e-5)
    m.fs.M101.toluene_feed.temperature.fix(303.2)
    m.fs.M101.toluene_feed.pressure.fix(350000)

    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "benzene"].fix(1e-5)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "toluene"].fix(1e-5)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "hydrogen"].fix(0.30)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Vap", "methane"].fix(0.02)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Liq", "benzene"].fix(1e-5)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Liq", "toluene"].fix(1e-5)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Liq", "hydrogen"].fix(1e-5)
    m.fs.M101.hydrogen_feed.flow_mol_phase_comp[0, "Liq", "methane"].fix(1e-5)
    m.fs.M101.hydrogen_feed.temperature.fix(303.2)
    m.fs.M101.hydrogen_feed.pressure.fix(350000)

    ## Fixing unit model specifications
    m.fs.H101.outlet.temperature.fix(600)
    m.fs.R101.conversion = Var(initialize=0.75, bounds=(0, 1))
    m.fs.R101.conv_constraint = Constraint(
        expr=m.fs.R101.conversion*m.fs.R101.inlet.
        flow_mol_phase_comp[0, "Vap", "toluene"] ==
        (m.fs.R101.inlet.flow_mol_phase_comp[0, "Vap", "toluene"] -
         m.fs.R101.outlet.flow_mol_phase_comp[0, "Vap", "toluene"]))

    m.fs.R101.conversion.fix(0.75)
    m.fs.R101.heat_duty.fix(0)

    m.fs.F101.vap_outlet.temperature.fix(325.0)
    m.fs.F101.deltaP.fix(0)

    m.fs.F102.vap_outlet.temperature.fix(375)
    m.fs.F102.deltaP.fix(-200000)

    m.fs.S101.split_fraction[0, "purge"].fix(0.2)
    m.fs.C101.outlet.pressure.fix(350000)

    print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0

    ## Initialization
    seq = SequentialDecomposition()
    seq.options.select_tear_method = "heuristic"
    seq.options.tear_method = "Wegstein"
    seq.options.iterLim = 3

    # Using the SD tool
    G = seq.create_graph(m)
    heuristic_tear_set = seq.tear_set_arcs(G, method="heuristic")
    order = seq.calculation_order(G)

    for o in heuristic_tear_set:
        print(o.name)
    for o in order:
        print(o[0].name)

    tear_guesses = {
            "flow_mol_phase_comp": {
                    (0, "Vap", "benzene"): 1e-5,
                    (0, "Vap", "toluene"): 1e-5,
                    (0, "Vap", "hydrogen"): 0.30,
                    (0, "Vap", "methane"): 0.02,
                    (0, "Liq", "benzene"): 1e-5,
                    (0, "Liq", "toluene"): 0.30,
                    (0, "Liq", "hydrogen"): 1e-5,
                    (0, "Liq", "methane"): 1e-5},
            "temperature": {0: 303},
            "pressure": {0: 350000}}
    seq.set_guesses_for(m.fs.H101.inlet, tear_guesses)

    def function(unit):
        unit.initialize(outlvl=idaeslog.INFO)

    seq.run(m, function)

    # Create the solver object
    from idaes.core.solvers import get_solver
    solver = get_solver()
    results = solver.solve(m, tee=True)

    # Check solver solve status
    assert results.solver.termination_condition == TerminationCondition.optimal

    print('operating cost = $', value(m.fs.operating_cost))
    assert value(m.fs.operating_cost) == pytest.approx(419122.3387, abs=1e-3)

    m.fs.F102.report()
    print()
    print('benzene purity = ', value(m.fs.purity))

    assert value(m.fs.purity) == pytest.approx(0.82429, abs=1e-3)
    assert value(m.fs.F102.heat_duty[0]) == pytest.approx(7352.4828, abs=1e-3)
    assert value(m.fs.F102.vap_outlet.pressure[0]) == pytest.approx(1.5000e+05, abs=1e-3)

    st = create_stream_table_dataframe({"Reactor": m.fs.s05, "Light Gases": m.fs.s06})
    print(stream_table_dataframe_to_string(st))

    ## Optimization
    m.fs.objective = Objective(expr=m.fs.operating_cost)

    m.fs.H101.outlet.temperature.unfix()
    m.fs.R101.heat_duty.unfix()
    m.fs.F101.vap_outlet.temperature.unfix()
    m.fs.F102.vap_outlet.temperature.unfix()
    m.fs.F102.deltaP.unfix()

    assert degrees_of_freedom(m) == 5

    m.fs.H101.outlet.temperature[0].setlb(500)
    m.fs.H101.outlet.temperature[0].setub(600)
    m.fs.R101.outlet.temperature[0].setlb(600)
    m.fs.R101.outlet.temperature[0].setub(800)
    m.fs.F101.vap_outlet.temperature[0].setlb(298.0)
    m.fs.F101.vap_outlet.temperature[0].setub(450.0)
    m.fs.F102.vap_outlet.temperature[0].setlb(298.0)
    m.fs.F102.vap_outlet.temperature[0].setub(450.0)
    m.fs.F102.vap_outlet.pressure[0].setlb(105000)
    m.fs.F102.vap_outlet.pressure[0].setub(110000)

    m.fs.overhead_loss = Constraint(
            expr=m.fs.F101.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"] <=
            0.20 * m.fs.R101.outlet.flow_mol_phase_comp[0, "Vap", "benzene"])
    m.fs.product_flow = Constraint(
            expr=m.fs.F102.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"] >=
            0.15)
    m.fs.product_purity = Constraint(expr=m.fs.purity >= 0.80)

    results = solver.solve(m, tee=True)
    assert results.solver.termination_condition == TerminationCondition.optimal

    ## Optimization Results
    print('operating cost = $', value(m.fs.operating_cost))
    print()
    print('Product flow rate and purity in F102')
    m.fs.F102.report()
    print()
    print('benzene purity = ', value(m.fs.purity))
    print()
    print('Overhead loss in F101')
    m.fs.F101.report()

    assert value(m.fs.operating_cost) == pytest.approx(312786.338, abs=1e-3)
    assert value(m.fs.purity) == pytest.approx(0.818827, abs=1e-3)

    print('Optimal Values')
    print()
    print('H101 outlet temperature = ', value(m.fs.H101.outlet.temperature[0]), 'K')
    print()
    print('R101 outlet temperature = ', value(m.fs.R101.outlet.temperature[0]), 'K')
    print()
    print('F101 outlet temperature = ', value(m.fs.F101.vap_outlet.temperature[0]), 'K')
    print()
    print('F102 outlet temperature = ', value(m.fs.F102.vap_outlet.temperature[0]), 'K')
    print('F102 outlet pressure = ', value(m.fs.F102.vap_outlet.pressure[0]), 'Pa')

    assert value(m.fs.H101.outlet.temperature[0]) == pytest.approx(500, abs=1e-3)
    assert value(m.fs.R101.outlet.temperature[0]) == pytest.approx(696.112, abs=1e-3)
    assert value(m.fs.F101.vap_outlet.temperature[0]) == pytest.approx(301.878, abs=1e-3)
    assert value(m.fs.F102.vap_outlet.temperature[0]) == pytest.approx(362.935, abs=1e-3)
    assert value(m.fs.F102.vap_outlet.pressure[0]) == pytest.approx(105000, abs=1e-2)
