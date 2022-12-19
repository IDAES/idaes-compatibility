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
import pyomo.dae as pyodae
import idaes.core.solvers.petsc as petsc  # petsc utilities module
from . import pid
from idaes.core.util.math import smooth_max, smooth_min

from unittest.mock import patch


@patch("matplotlib.pyplot.show")
def test_example(mock_show):
    m = pid.create_model(
        time_set=[0, 12],
        nfe=1,
        calc_integ=True,
    )

    result = petsc.petsc_dae_by_time_element(
        m,
        time=m.fs.time,
        ts_options={
            "--ts_type": "beuler",
            "--ts_dt": 0.1,
            "--ts_monitor": "",  # set initial step to 0.1
            "--ts_save_trajectory": 1,
        },
    )
    tj = result.trajectory  # trajectroy data
    res = result.results  # solver status list

    a = plt.plot(tj.time, tj.get_vec(m.fs.valve_1.valve_opening[12]))
    a = plt.ylabel("valve 1 fraction open")
    a = plt.xlabel("time (s)")

    a = plt.plot(
        tj.time, tj.get_vec(m.fs.tank.control_volume.properties_out[12].pressure)
    )
    a = plt.ylabel("tank pressure (Pa)")
    a = plt.xlabel("time (s)")

    assert pyo.value(
        m.fs.tank.control_volume.properties_out[12].pressure
    ) == pytest.approx(3e5, rel=1e-5)
    assert pyo.value(m.fs.valve_1.valve_opening[12]) == pytest.approx(
        0.705927, rel=1e-5
    )

    m = pid.create_model(
        time_set=[0, 24],
        nfe=1,
        calc_integ=True,
    )
    # time_var will be an explicit time variable we can use in constraints.
    m.fs.time_var = pyo.Var(m.fs.time)

    m.fs.valve_1.control_volume.properties_in[0].pressure.unfix()
    m.fs.valve_1.control_volume.properties_in[24].pressure.unfix()

    m.fs.time_var[0].fix(m.fs.time.first())

    @m.fs.Constraint(m.fs.time)
    def inlet_pressure_eqn(b, t):
        return b.valve_1.control_volume.properties_in[t].pressure == smooth_min(
            600000, smooth_max(500000, 50000 * (b.time_var[t] - 10) + 500000)
        )

    result = petsc.petsc_dae_by_time_element(
        m,
        time=m.fs.time,
        timevar=m.fs.time_var,
        ts_options={
            "--ts_type": "beuler",
            "--ts_dt": 0.1,
            "--ts_monitor": "",  # set initial step to 0.1
            "--ts_save_trajectory": 1,
        },
    )
    tj = result.trajectory  # trajectroy data
    res = result.results  # solver status list

    a = plt.plot(
        tj.time, tj.get_vec(m.fs.valve_1.control_volume.properties_in[24].pressure)
    )
    a = plt.ylabel("inlet pressure (Pa)")
    a = plt.xlabel("time (s)")

    a = plt.plot(tj.time, tj.get_vec(m.fs.valve_1.valve_opening[24]))
    a = plt.ylabel("valve 1 fraction open")
    a = plt.xlabel("time (s)")

    a = plt.plot(
        tj.time, tj.get_vec(m.fs.tank.control_volume.properties_out[24].pressure)
    )
    a = plt.ylabel("tank pressure (Pa)")
    a = plt.xlabel("time (s)")

    assert pyo.value(
        m.fs.valve_1.control_volume.properties_in[24].pressure
    ) == pytest.approx(6e5, rel=1e-5)
    assert pyo.value(
        m.fs.tank.control_volume.properties_out[24].pressure
    ) == pytest.approx(3e5, rel=1e-5)
    assert pyo.value(m.fs.valve_1.valve_opening[24]) == pytest.approx(
        0.543422, rel=1e-5
    )

    m = pid.create_model(
        time_set=[0, 10, 12, 24],
        nfe=3,
        calc_integ=True,
    )
    # time_var will be an explicit time variable we can use in constraints.
    m.fs.time_var = pyo.Var(m.fs.time)

    m.fs.valve_1.control_volume.properties_in[0].pressure.fix(500000)
    m.fs.valve_1.control_volume.properties_in[10].pressure.fix(500000)
    m.fs.valve_1.control_volume.properties_in[12].pressure.set_value(600000)
    m.fs.valve_1.control_volume.properties_in[12].pressure.unfix()
    m.fs.valve_1.control_volume.properties_in[24].pressure.fix(600000)

    @m.fs.Constraint(m.fs.time)
    def inlet_pressure_eqn(b, t):
        return (
            b.valve_1.control_volume.properties_in[t].pressure
            == 50000 * (b.time_var[t] - 10) + 500000
        )

    m.fs.inlet_pressure_eqn.deactivate()
    m.fs.inlet_pressure_eqn[12].activate()

    result = petsc.petsc_dae_by_time_element(
        m,
        time=m.fs.time,
        timevar=m.fs.time_var,
        ts_options={
            "--ts_type": "beuler",
            "--ts_dt": 0.1,
            "--ts_monitor": "",  # set initial step to 0.1
            "--ts_save_trajectory": 1,
        },
    )
    tj = result.trajectory  # trajectroy data
    res = result.results  # solver status list

    a = plt.plot(
        tj.time, tj.get_vec(m.fs.valve_1.control_volume.properties_in[24].pressure)
    )
    a = plt.ylabel("inlet pressure (Pa)")
    a = plt.xlabel("time (s)")

    a = plt.plot(tj.time, tj.get_vec(m.fs.valve_1.valve_opening[24]))
    a = plt.ylabel("valve 1 fraction open")
    a = plt.xlabel("time (s)")

    a = plt.plot(
        tj.time, tj.get_vec(m.fs.tank.control_volume.properties_out[24].pressure)
    )
    a = plt.ylabel("tank pressure (Pa)")
    a = plt.xlabel("time (s)")

    assert pyo.value(
        m.fs.valve_1.control_volume.properties_in[24].pressure
    ) == pytest.approx(6e5, rel=1e-5)
    assert pyo.value(
        m.fs.tank.control_volume.properties_out[24].pressure
    ) == pytest.approx(3e5, rel=1e-5)
    assert pyo.value(m.fs.valve_1.valve_opening[24]) == pytest.approx(
        0.543422, rel=1e-5
    )
