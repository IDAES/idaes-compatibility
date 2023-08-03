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
import os
import numpy as np
import pandas as pd
from io import StringIO
import sys

from pyomo.environ import (
    ConcreteModel,
    SolverFactory,
    value,
    Var,
    Constraint,
    Set,
    Objective,
    maximize,
)
from pyomo.common.timing import TicTocTimer

from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.alamopy import AlamoTrainer, AlamoSurrogate, alamo
from idaes.core.surrogate.plotting.sm_plotter import (
    surrogate_scatter2D,
    surrogate_parity,
    surrogate_residual,
)
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core import FlowsheetBlock
from idaes.core.util.convergence.convergence_base import _run_ipopt_with_stats

from unittest.mock import patch
import matplotlib.pyplot as plt
from pyomo.common.tempfiles import TempfileManager


path = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(path, "reformer-data.csv")


@pytest.mark.skipif(not alamo.available(), reason="ALAMO not available")
@patch("matplotlib.pyplot.show")
def test_example(mock_show):
    np.set_printoptions(precision=6, suppress=True)

    csv_data = pd.read_csv(data_file)  # 2800 data points
    data = csv_data.sample(n=100)  # randomly sample points for training/validation

    input_data = data.iloc[:, :2]
    output_data = data.iloc[:, 2:]

    # Define labels, and split training and validation data
    input_labels = input_data.columns
    output_labels = output_data.columns

    n_data = data[input_labels[0]].size
    data_training, data_validation = split_training_validation(
        data, 0.8, seed=n_data
    )  # seed=100

    # capture long output (not required to use surrogate API)
    stream = StringIO()
    oldstdout = sys.stdout
    sys.stdout = stream

    with TempfileManager as tf:
        # Create ALAMO trainer object
        trainer = AlamoTrainer(
            input_labels=input_labels,
            output_labels=output_labels,
            training_dataframe=data_training,
        )

        # Set ALAMO options
        trainer.config.constant = True
        trainer.config.linfcns = True
        trainer.config.multi2power = [1, 2]
        trainer.config.monomialpower = [2, 3]
        trainer.config.ratiopower = [1, 2]
        trainer.config.maxterms = [10] * len(output_labels)  # max for each surrogate
        tmpalm = tf.create_tempfile(suffix=".alm")
        trainer.config.filename = os.path.join(tmpalm)
        trainer.config.overwrite_files = True

        # Train surrogate (calls ALAMO through IDAES ALAMOPy wrapper)
        success, alm_surr, msg = trainer.train_surrogate()

        # save model to JSON
        tmpjson = tf.create_tempfile(suffix=".json")
        model = alm_surr.save_to_file(tmpjson, overwrite=True)

        # create callable surrogate object
        surrogate_expressions = trainer._results["Model"]
        input_labels = trainer._input_labels
        output_labels = trainer._output_labels
        xmin, xmax = [0.1, 0.8], [0.8, 1.2]
        input_bounds = {
            input_labels[i]: (xmin[i], xmax[i]) for i in range(len(input_labels))
        }

        alm_surr = AlamoSurrogate(
            surrogate_expressions, input_labels, output_labels, input_bounds
        )

        # revert back to normal output capture
        sys.stdout = oldstdout

        # display first 50 lines and last 50 lines of output
        celloutput = stream.getvalue().split("\n")
        for line in celloutput[:50]:
            print(line)
        print(".")
        print(".")
        print(".")
        for line in celloutput[-50:]:
            print(line)

        # visualize with IDAES surrogate plotting tools
        fname = tf.create_tempfile(suffix=".pdf")
        surrogate_scatter2D(alm_surr, data_training, filename=fname)
        fname = tf.create_tempfile(suffix=".pdf")
        surrogate_parity(alm_surr, data_training, filename=fname)
        fname = tf.create_tempfile(suffix=".pdf")
        surrogate_residual(alm_surr, data_training, filename=fname)

        # visualize with IDAES surrogate plotting tools
        fname = tf.create_tempfile(suffix=".pdf")
        surrogate_scatter2D(alm_surr, data_validation, filename=fname)
        fname = tf.create_tempfile(suffix=".pdf")
        surrogate_parity(alm_surr, data_validation, filename=fname)
        fname = tf.create_tempfile(suffix=".pdf")
        surrogate_residual(alm_surr, data_validation, filename=fname)

    # create the IDAES model and flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # create flowsheet input variables
    m.fs.bypass_frac = Var(
        initialize=0.80, bounds=[0.1, 0.8], doc="natural gas bypass fraction"
    )
    m.fs.ng_steam_ratio = Var(
        initialize=0.80, bounds=[0.8, 1.2], doc="natural gas to steam ratio"
    )

    # create flowsheet output variables
    m.fs.steam_flowrate = Var(initialize=0.2, doc="steam flowrate")
    m.fs.reformer_duty = Var(initialize=10000, doc="reformer heat duty")
    m.fs.AR = Var(initialize=0, doc="AR fraction")
    m.fs.C2H6 = Var(initialize=0, doc="C2H6 fraction")
    m.fs.C3H8 = Var(initialize=0, doc="C3H8 fraction")
    m.fs.C4H10 = Var(initialize=0, doc="C4H10 fraction")
    m.fs.CH4 = Var(initialize=0, doc="CH4 fraction")
    m.fs.CO = Var(initialize=0, doc="CO fraction")
    m.fs.CO2 = Var(initialize=0, doc="CO2 fraction")
    m.fs.H2 = Var(initialize=0, doc="H2 fraction")
    m.fs.H2O = Var(initialize=0, doc="H2O fraction")
    m.fs.N2 = Var(initialize=0, doc="N2 fraction")
    m.fs.O2 = Var(initialize=0, doc="O2 fraction")

    # create input and output variable object lists for flowsheet
    inputs = [m.fs.bypass_frac, m.fs.ng_steam_ratio]
    outputs = [
        m.fs.steam_flowrate,
        m.fs.reformer_duty,
        m.fs.AR,
        m.fs.C2H6,
        m.fs.C4H10,
        m.fs.C3H8,
        m.fs.CH4,
        m.fs.CO,
        m.fs.CO2,
        m.fs.H2,
        m.fs.H2O,
        m.fs.N2,
        m.fs.O2,
    ]

    # create the Pyomo/IDAES block that corresponds to the surrogate
    # ALAMO
    surrogate = AlamoSurrogate.load_from_file("alamo_surrogate.json")
    m.fs.surrogate = SurrogateBlock()
    m.fs.surrogate.build_model(surrogate, input_vars=inputs, output_vars=outputs)

    # fix input values and solve flowsheet
    m.fs.bypass_frac.fix(0.5)
    m.fs.ng_steam_ratio.fix(1)

    solver = SolverFactory("ipopt")
    [status_obj, solved, iters, time] = _run_ipopt_with_stats(m, solver)

    print("Model status: ", status_obj)
    print("Solution optimal: ", solved)
    print("IPOPT iterations: ", iters)
    print("IPOPT runtime: ", time)

    print()
    print("Steam flowrate = ", value(m.fs.steam_flowrate))
    print("Reformer duty = ", value(m.fs.reformer_duty))
    print("Mole Fraction Ar = ", value(m.fs.AR))
    print("Mole Fraction C2H6 = ", value(m.fs.C2H6))
    print("Mole Fraction C3H8 = ", value(m.fs.C3H8))
    print("Mole Fraction C4H10 = ", value(m.fs.C4H10))
    print("Mole Fraction CH4 = ", value(m.fs.CH4))
    print("Mole Fraction CO = ", value(m.fs.CO))
    print("Mole Fraction CO2 = ", value(m.fs.CO2))
    print("Mole Fraction H2 = ", value(m.fs.H2))
    print("Mole Fraction H2O = ", value(m.fs.H2O))
    print("Mole Fraction N2 = ", value(m.fs.N2))
    print("Mole Fraction O2 = ", value(m.fs.O2))

    # unfix input values and add the objective/constraint to the model
    m.fs.bypass_frac.unfix()
    m.fs.ng_steam_ratio.unfix()
    m.fs.obj = Objective(expr=m.fs.H2, sense=maximize)
    m.fs.con = Constraint(expr=m.fs.N2 <= 0.34)

    # solve the model
    tmr = TicTocTimer()
    status = solver.solve(m, tee=True)
    solve_time = tmr.toc("solve")

    # print and check results
    assert abs(value(m.fs.H2) - 0.33) <= 0.01
    assert value(m.fs.N2 <= 0.4 + 1e-8)
    print("Model status: ", status)
    print("Solve time: ", solve_time)
    for var in inputs:
        print(var.name, ": ", value(var))
    for var in outputs:
        print(var.name, ": ", value(var))
