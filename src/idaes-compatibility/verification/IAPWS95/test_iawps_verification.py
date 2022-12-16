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

from pyomo.environ import ConcreteModel, value, units

from idaes.core import FlowsheetBlock
import idaes.models.properties.iapws95 as iapws95


def test_iawps95_verification():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.iapws = iapws95.Iapws95ParameterBlock(amount_basis=iapws95.AmountBasis.MASS)
    m.fs.state = m.fs.iapws.build_state_block(m.fs.time, defined_state=True)

    m.fs.state[0].flow_mass.fix(1)
    m.fs.state[0].pressure.fix(22.065*units.MPa)
    m.fs.state[0].enth_mass.fix(25.576*units.kJ/units.kg)

    m.fs.state[0].temperature.display()
    assert False