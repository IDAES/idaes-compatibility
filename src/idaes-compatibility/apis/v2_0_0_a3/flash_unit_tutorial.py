#################################################################################
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
Backward compatibility test for flash unit tutorial/
"""
import numpy as np
import matplotlib.pyplot as plt

from pyomo.environ import ConcreteModel, SolverFactory, Constraint, value

from idaes.core import FlowsheetBlock
import idaes.logger as idaeslog
from idaes.models.properties.activity_coeff_models.BTX_activity_coeff_VLE import (
    BTXParameterBlock,
)
from idaes.models.unit_models import Flash
from idaes.core.util.model_statistics import degrees_of_freedom

from .workshoptools import solve_successful


def test_tutorial():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.pprint()

    m.fs.properties = BTXParameterBlock(
        valid_phase=("Liq", "Vap"), activity_coeff_model="Ideal", state_vars="FTPz"
    )

    m.fs.flash = Flash(property_package=m.fs.properties)

    # Check degrees of freedom
    help(degrees_of_freedom)
    print("Degrees of Freedom =", degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 7

    # Add inlet specifications given above
    m.fs.flash.inlet.flow_mol.fix(1)
    m.fs.flash.inlet.temperature.fix(368)
    m.fs.flash.inlet.pressure.fix(101325)
    m.fs.flash.inlet.mole_frac_comp[0, "benzene"].fix(0.5)
    m.fs.flash.inlet.mole_frac_comp[0, "toluene"].fix(0.5)
    m.fs.flash.heat_duty.fix(0)
    m.fs.flash.deltaP.fix(0)

    # Recheck DoF
    print("Degrees of Freedom =", degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0

    # initialize the flash unit
    m.fs.flash.initialize(outlvl=idaeslog.INFO)

    # create the ipopt solver
    solver = SolverFactory("ipopt")
    status = solver.solve(m, tee=True)

    # Check for optimal solution
    from pyomo.environ import TerminationCondition

    assert status.solver.termination_condition == TerminationCondition.optimal

    # Print the pressure of the flash vapor outlet
    print("Pressure =", value(m.fs.flash.vap_outlet.pressure[0]))

    print()
    print("Output from display:")
    # Call display on vap_outlet and liq_outlet of the flash
    m.fs.flash.vap_outlet.display()
    m.fs.flash.liq_outlet.display()

    m.fs.flash.report()

    # Check optimal solution values
    import pytest

    assert value(m.fs.flash.liq_outlet.flow_mol[0]) == pytest.approx(0.6038, abs=1e-3)
    assert value(m.fs.flash.liq_outlet.mole_frac_comp[0, "benzene"]) == pytest.approx(
        0.4121, abs=1e-3
    )
    assert value(m.fs.flash.liq_outlet.mole_frac_comp[0, "toluene"]) == pytest.approx(
        0.5878, abs=1e-3
    )
    assert value(m.fs.flash.liq_outlet.temperature[0]) == pytest.approx(368, abs=1e-3)
    assert value(m.fs.flash.liq_outlet.pressure[0]) == pytest.approx(101325, abs=1e-3)

    assert value(m.fs.flash.vap_outlet.flow_mol[0]) == pytest.approx(0.3961, abs=1e-3)
    assert value(m.fs.flash.vap_outlet.mole_frac_comp[0, "benzene"]) == pytest.approx(
        0.6339, abs=1e-3
    )
    assert value(m.fs.flash.vap_outlet.mole_frac_comp[0, "toluene"]) == pytest.approx(
        0.3660, abs=1e-3
    )
    assert value(m.fs.flash.vap_outlet.temperature[0]) == pytest.approx(368, abs=1e-3)
    assert value(m.fs.flash.vap_outlet.pressure[0]) == pytest.approx(101325, abs=1e-3)

    ## Studying Purity as a Function of Heat Duty
    # create the empty lists to store the results that will be plotted
    Q = []
    V = []

    # re-initialize model
    m.fs.flash.initialize(outlvl=idaeslog.WARNING)

    # Todo: Write the for loop specification using numpy's linspace
    for duty in np.linspace(-17000, 25000, 50):
        # fix the heat duty
        m.fs.flash.heat_duty.fix(duty)

        # append the value of the duty to the Q list
        Q.append(duty)

        # print the current simulation
        print("Simulating with Q = ", value(m.fs.flash.heat_duty[0]))

        # Solve the model
        status = solver.solve(m)

        # append the value for vapor fraction if the solve was successful
        if solve_successful(status):
            V.append(value(m.fs.flash.vap_outlet.flow_mol[0]))
            print("... solve successful.")
        else:
            V.append(0.0)
            print("... solve failed.")

    # Create and show the figure
    plt.figure("Vapor Fraction")
    plt.plot(Q, V)
    plt.grid()
    plt.xlabel("Heat Duty [J]")
    plt.ylabel("Vapor Fraction [-]")
    # plt.show()

    # Todo: generate a figure of heat duty vs. mole fraction of Benzene in the vapor
    Q = []
    V = []

    for duty in np.linspace(-17000, 25000, 50):
        # fix the heat duty
        m.fs.flash.heat_duty.fix(duty)

        # append the value of the duty to the Q list
        Q.append(duty)

        # print the current simulation
        print("Simulating with Q = ", value(m.fs.flash.heat_duty[0]))

        # solve the model
        status = solver.solve(m)

        # append the value for vapor fraction if the solve was successful
        if solve_successful(status):
            V.append(value(m.fs.flash.vap_outlet.mole_frac_comp[0, "benzene"]))
            print("... solve successful.")
        else:
            V.append(0.0)
            print("... solve failed.")

    plt.figure("Purity")
    plt.plot(Q, V)
    plt.grid()
    plt.xlabel("Heat Duty [J]")
    plt.ylabel("Vapor Benzene Mole Fraction [-]")
    # plt.show()

    # re-initialize the model - this may or may not be required depending on current state but safe to initialize
    m.fs.flash.heat_duty.fix(0)
    m.fs.flash.initialize(outlvl=idaeslog.WARNING)

    # Unfix the heat_duty variable
    m.fs.flash.heat_duty.unfix()

    # Todo: Add a new constraint (benzene mole fraction to 0.6)
    m.benz_purity_con = Constraint(
        expr=m.fs.flash.vap_outlet.mole_frac_comp[0, "benzene"] == 0.6
    )

    # solve the problem
    status = solver.solve(m, tee=True)

    # Check stream condition
    m.fs.flash.report()

    # Check for solver status
    assert status.solver.termination_condition == TerminationCondition.optimal

    # Check for optimal values
    assert value(m.fs.flash.liq_outlet.flow_mol[0]) == pytest.approx(0.4516, abs=1e-3)
    assert value(m.fs.flash.liq_outlet.mole_frac_comp[0, "benzene"]) == pytest.approx(
        0.3786, abs=1e-3
    )
    assert value(m.fs.flash.liq_outlet.mole_frac_comp[0, "toluene"]) == pytest.approx(
        0.6214, abs=1e-3
    )
    assert value(m.fs.flash.liq_outlet.temperature[0]) == pytest.approx(
        369.07, abs=1e-2
    )
    assert value(m.fs.flash.liq_outlet.pressure[0]) == pytest.approx(101325, abs=1e-3)

    assert value(m.fs.flash.vap_outlet.flow_mol[0]) == pytest.approx(0.5483, abs=1e-3)
    assert value(m.fs.flash.vap_outlet.mole_frac_comp[0, "benzene"]) == pytest.approx(
        0.6, abs=1e-3
    )
    assert value(m.fs.flash.vap_outlet.mole_frac_comp[0, "toluene"]) == pytest.approx(
        0.4, abs=1e-3
    )
    assert value(m.fs.flash.vap_outlet.temperature[0]) == pytest.approx(
        369.07, abs=1e-2
    )
    assert value(m.fs.flash.vap_outlet.pressure[0]) == pytest.approx(101325, abs=1e-3)
