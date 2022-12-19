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

import numpy as np
import matplotlib.pyplot as plt

import pyomo.environ as pyo
import idaes.core.solvers.petsc as petsc  # petsc utilities module
from idaes.core.solvers.features import dae  # DAE example/test problem

from unittest.mock import patch
import matplotlib.pyplot as plt


@patch("matplotlib.pyplot.show")
def test_example(mock_show):
    m, y1, y2, y3, y4, y5, y6 = dae(nfe=10)

    # See the initial conditions:
    print("at t = 0:")
    print(f"    y1 = {pyo.value(m.y[0, 1])}")
    print(f"    y2 = {pyo.value(m.y[0, 2])}")
    print(f"    y3 = {pyo.value(m.y[0, 3])}")
    print(f"    y4 = {pyo.value(m.y[0, 4])}")
    print(f"    y5 = {pyo.value(m.y[0, 5])}")

    result = petsc.petsc_dae_by_time_element(
        m,
        time=m.t,
        between=[m.t.first(), m.t.last()],
        ts_options={
            "--ts_type": "cn",  # Crank–Nicolson
            "--ts_adapt_type": "basic",
            "--ts_dt": 0.01,
            "--ts_save_trajectory": 1,
        },
    )
    tj = result.trajectory
    res = result.results

    # Verify results
    assert abs(y1 - pyo.value(m.y[180, 1])) / y1 < 1e-3
    assert abs(y2 - pyo.value(m.y[180, 2])) / y2 < 1e-3
    assert abs(y3 - pyo.value(m.y[180, 3])) / y3 < 1e-3
    assert abs(y4 - pyo.value(m.y[180, 4])) / y4 < 1e-3
    assert abs(y5 - pyo.value(m.y[180, 5])) / y5 < 1e-3
    assert abs(y6 - pyo.value(m.y6[180])) / y6 < 1e-3

    a = plt.plot(m.t, [pyo.value(m.y[t, 1]) for t in m.t], label="y1")
    a = plt.plot(m.t, [pyo.value(m.y[t, 2]) for t in m.t], label="y2")
    a = plt.plot(m.t, [pyo.value(m.y[t, 3]) for t in m.t], label="y3")
    a = plt.plot(m.t, [pyo.value(m.y[t, 4]) for t in m.t], label="y4")
    a = plt.plot(m.t, [pyo.value(m.y[t, 5]) for t in m.t], label="y5")
    a = plt.plot(m.t, [pyo.value(m.y6[t]) for t in m.t], label="y6")
    a = plt.legend()
    a = plt.ylabel("Concentration (mol/l)")
    a = plt.xlabel("time (min)")

    a = plt.plot(tj.time, tj.get_vec(m.y[180, 1]), label="y1")
    a = plt.plot(tj.time, tj.get_vec(m.y[180, 2]), label="y2")
    a = plt.plot(tj.time, tj.get_vec(m.y[180, 3]), label="y3")
    a = plt.plot(tj.time, tj.get_vec(m.y[180, 4]), label="y4")
    a = plt.plot(tj.time, tj.get_vec(m.y[180, 5]), label="y5")
    a = plt.plot(tj.time, tj.get_vec(m.y6[180]), label="y6")
    a = plt.legend()
    a = plt.ylabel("Concentration (mol/l)")
    a = plt.xlabel("time (min)")

    a = plt.plot(tj.time, tj.get_vec(m.y[180, 2]), label="y2")
    a = plt.plot(tj.time, tj.get_vec(m.y[180, 4]), label="y4")
    a = plt.legend()
    a = plt.ylabel("Concentration (mol/l)")
    a = plt.xlabel("time (min)")

    a = plt.plot(tj.vecs["_time"], tj.vecs[str(m.y[180, 2])], label="y2")
    a = plt.legend()
    a = plt.ylabel("Concentration (mol/l)")
    a = plt.xlabel("time (min)")
    a = plt.xlim(0, 2)

    tji = tj.interpolate(np.linspace(0, 180, 181))

    a = plt.plot(tji.time, tji.get_vec(m.y[180, 1]), label="y1")
    a = plt.plot(tji.time, tji.get_vec(m.y[180, 2]), label="y2")
    a = plt.plot(tji.time, tji.get_vec(m.y[180, 3]), label="y3")
    a = plt.plot(tji.time, tji.get_vec(m.y[180, 4]), label="y4")
    a = plt.plot(tji.time, tji.get_vec(m.y[180, 5]), label="y5")
    a = plt.plot(tji.time, tji.get_vec(m.y6[180]), label="y6")
    a = plt.legend()
    a = plt.ylabel("Concentration (mol/l)")
    a = plt.xlabel("time (min)")

    a = plt.plot(tji.time, tji.get_vec(m.y[180, 2]), label="y2 interpolate dt=1")
    a = plt.plot(tj.time, tj.get_vec(m.y[180, 2]), label="y2 original")
    a = plt.legend()
    a = plt.ylabel("Concentration (mol/l)")
    a = plt.xlabel("time (min)")
    a = plt.xlim(0, 2)
