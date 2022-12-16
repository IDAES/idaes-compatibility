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
    ConcreteModel,
    Constraint,
    value,
    units,
    assert_optimal_termination,
)

import idaes.logger as idaeslog
from idaes.core import FlowsheetBlock
from idaes.models_extra.column_models.properties.MEA_solvent import (
    configuration as aqueous_mea,
)
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver
from idaes.models.unit_models.heat_exchanger_ntu import HeatExchangerNTU as HXNTU


def test_example():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.hotside_properties = GenericParameterBlock(**aqueous_mea)
    m.fs.coldside_properties = GenericParameterBlock(**aqueous_mea)

    m.fs.heat_exchanger = HXNTU(
        hot_side_name="shell",
        cold_side_name="tube",
        shell={
            "property_package": m.fs.hotside_properties,
            "has_pressure_change": True,
        },
        tube={
            "property_package": m.fs.coldside_properties,
            "has_pressure_change": True,
        },
    )

    DOF_initial = degrees_of_freedom(m)
    print("The initial DOF is {0}".format(DOF_initial))
    assert DOF_initial == 15

    m.fs.heat_exchanger.hot_side_inlet.flow_mol[0].fix(60.54879)  # mol/s
    m.fs.heat_exchanger.hot_side_inlet.temperature[0].fix(392.23)  # K
    m.fs.heat_exchanger.hot_side_inlet.pressure[0].fix(202650)  # Pa
    m.fs.heat_exchanger.hot_side_inlet.mole_frac_comp[0, "CO2"].fix(
        0.0158
    )  # dimensionless
    m.fs.heat_exchanger.hot_side_inlet.mole_frac_comp[0, "H2O"].fix(
        0.8747
    )  # dimensionless
    m.fs.heat_exchanger.hot_side_inlet.mole_frac_comp[0, "MEA"].fix(
        0.1095
    )  # dimensionless

    m.fs.heat_exchanger.cold_side_inlet.flow_mol[0].fix(63.01910)  # mol/s
    m.fs.heat_exchanger.cold_side_inlet.temperature[0].fix(326.36)  # K
    m.fs.heat_exchanger.cold_side_inlet.pressure[0].fix(202650)  # Pa
    m.fs.heat_exchanger.cold_side_inlet.mole_frac_comp[0, "CO2"].fix(
        0.0414
    )  # dimensionless
    m.fs.heat_exchanger.cold_side_inlet.mole_frac_comp[0, "H2O"].fix(
        0.8509
    )  # dimensionless
    m.fs.heat_exchanger.cold_side_inlet.mole_frac_comp[0, "MEA"].fix(
        0.1077
    )  # dimensionless

    DOF_initial = degrees_of_freedom(m)
    print("The DOF is {0}".format(DOF_initial))

    m.fs.heat_exchanger.area.fix(100)  # m2
    m.fs.heat_exchanger.heat_transfer_coefficient.fix(200)  # W/m2/K
    m.fs.heat_exchanger.effectiveness.fix(0.7)  # dimensionless

    m.fs.heat_exchanger.hot_side.deltaP.fix(-2000)  # Pa
    m.fs.heat_exchanger.cold_side.deltaP.fix(-2000)  # Pa

    DOF_final = degrees_of_freedom(m)
    print("The DOF is {0}".format(DOF_final))
    assert DOF_final == 0

    m.fs.heat_exchanger.initialize(outlvl=idaeslog.INFO)
    opt = get_solver()
    solve_status = opt.solve(m, tee=True)

    m.fs.heat_exchanger.report()

    assert_optimal_termination(solve_status)
    assert value(m.fs.heat_exchanger.hot_side_outlet.temperature[0]) == pytest.approx(
        344.00, abs=1e-2
    )
    assert value(m.fs.heat_exchanger.cold_side_outlet.temperature[0]) == pytest.approx(
        374.33, abs=1e-2
    )

    m.fs.heat_exchanger.effectiveness.unfix()
    m.fs.heat_exchanger.area.unfix()
    m.fs.heat_exchanger.hot_side_outlet.temperature.fix(344.0)

    DOF_final = degrees_of_freedom(m)
    print("The DOF is {0}".format(DOF_final))
    assert DOF_final == 0

    result = opt.solve(m)
    print(result)
    m.fs.heat_exchanger.report()

    assert_optimal_termination(result)
    assert value(m.fs.heat_exchanger.area) == pytest.approx(100, abs=1e-2)
    assert value(m.fs.heat_exchanger.effectiveness[0]) == pytest.approx(0.7, abs=1e-2)
    assert value(m.fs.heat_exchanger.cold_side_outlet.temperature[0]) == pytest.approx(
        374.33, rel=1e-3
    )
