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
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

from pyomo.environ import *
from pyomo.network import Arc, SequentialDecomposition

from idaes.core import *
from idaes.models.unit_models import (
    PressureChanger,
    CSTR,
    Flash,
    Heater,
    Mixer,
    Separator,
)
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption

from idaes.core.util.model_statistics import degrees_of_freedom


def test_tutorial():
    check_count = 0
    # -----------------------------------------------------------------------------
    # Test available solvers
    if SolverFactory("ipopt").available():
        print("Solver Availability Check:  Passed")
        check_count += 1
    else:
        print("Solver Availability Check:  FAILED")

    # -----------------------------------------------------------------------------
    # Check model construction and solving
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.v = Var(m.fs.time)

    def cons_rule(b, t):
        return b.v[t] == 1

    m.fs.c = Constraint(m.fs.time, rule=cons_rule)

    # Create a solver
    solver = SolverFactory("ipopt")
    results = solver.solve(m.fs)

    if (
        results.solver.termination_condition == TerminationCondition.optimal
        and results.solver.status == SolverStatus.ok
    ):
        check_count += 1

    assert check_count == 2

    x = 5
    print(x)

    x = 8
    print(x)

    T_degC = 20
    T_degF = (T_degC * 9.0 / 5.0) + 32.0
    if T_degF < 70:
        print("The room is too cold.")

    xlist = list()
    for i in range(11):
        # Todo: use the append method of list to append the correct value
        xlist.append(i * 5)
    print(xlist)

    xlist = [i * 5 for i in range(11)]
    print(xlist)
    print(len(xlist))

    ylist = list()
    for x in xlist:
        ylist.append(x**2)
    print(ylist)

    ylist = [x**2 for x in xlist]
    print(ylist)

    areas = dict()
    areas["South Dakota"] = 199742
    areas["Oklahoma"] = 181035
    print(areas)

    areas_mi = dict()
    for state_name, area in areas.items():
        # Todo: convert the area to sq. mi and assign to the areas_mi dict.
        areas_mi[state_name] = area * (0.62137**2)
    print(areas_mi)

    # Todo: define areas_mi using a dictionary comprehension and print the result
    areas_mi = {k: v * (0.62137**2) for k, v in areas.items()}
    print(areas_mi)

    xlist = list(np.linspace(0, 50, 16))
    ylist = [x**2 for x in xlist]
    print(xlist)
    print(ylist)

    plt.plot(xlist, ylist)
    plt.title("Embedded x vs y figure")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["data"])
    # plt.show()

    x = list(np.linspace(0, 2 * math.pi, 100))
    # Todo: create the list for y
    y = [math.sin(xv) for xv in x]

    # Todo: Generate the figure
    plt.plot(x, y)
    plt.title("Trig: sin function")
    plt.xlabel("x in radians")
    plt.ylabel("sin(x)")
    # plt.show()

    df_sin = pd.DataFrame({"x": x, "sin(x) (radians)": y})
    print(df_sin)
    # df_sin.to_csv('sin_data.csv')

    model = ConcreteModel()
    model.x = Var()
    model.y = Var()

    model.obj = Objective(expr=model.x**2 + model.y**2)
    model.con = Constraint(expr=model.x + model.y == 1)
    status = SolverFactory("ipopt").solve(model, tee=True)
    print("x =", value(model.x))
    print("y =", value(model.y))
    print("obj =", value(model.obj))

    assert value(model.obj) == 0.5
    assert value(model.x) == 0.5
    assert value(model.y) == 0.5

    print("*** Output from model.pprint():")
    model.pprint()

    print()
    print("*** Output from model.display():")
    model.display()
