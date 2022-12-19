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
    exp,
    Param,
    Set,
    units as pyunits,
    Var,
    ConcreteModel,
    TerminationCondition,
    SolverStatus,
    value,
)
from pyomo.util.check_units import assert_units_consistent

from idaes.core.solvers import get_solver
from idaes.models.unit_models import CSTR
from .thermophysical_property_example import HDAParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core import (
    FlowsheetBlock,
    declare_process_block_class,
    MaterialFlowBasis,
    ReactionParameterBlock,
    ReactionBlockDataBase,
    ReactionBlockBase,
)
from idaes.core.util.constants import Constants as const
import idaes.logger as idaeslog


def test_example():
    units_metadata = {
        "time": pyunits.s,
        "length": pyunits.m,
        "mass": pyunits.kg,
        "amount": pyunits.mol,
        "temperature": pyunits.K,
    }

    properties_metadata = {
        "k_rxn": {"method": None},
        "k_eq": {"method": None},
        "reaction_rate": {"method": None},
    }

    def define_kinetic_reactions(self):
        # Rate Reaction Index
        self.rate_reaction_idx = Set(initialize=["R1"])

        # Rate Reaction Stoichiometry
        self.rate_reaction_stoichiometry = {
            ("R1", "Vap", "benzene"): 1,
            ("R1", "Vap", "toluene"): -1,
            ("R1", "Vap", "hydrogen"): -1,
            ("R1", "Vap", "methane"): 1,
            ("R1", "Vap", "diphenyl"): 0,
        }

    def define_equilibrium_reactions(self):
        # Equilibrium Reaction Index
        self.equilibrium_reaction_idx = Set(initialize=["E1"])

        # Equilibrium Reaction Stoichiometry
        self.equilibrium_reaction_stoichiometry = {
            ("E1", "Vap", "benzene"): -2,
            ("E1", "Vap", "toluene"): 0,
            ("E1", "Vap", "hydrogen"): 1,
            ("E1", "Vap", "methane"): 0,
            ("E1", "Vap", "diphenyl"): 1,
        }

    def define_parameters(self):
        # Arrhenius Constant
        self.arrhenius = Param(
            default=1.25e-9,
            doc="Arrhenius constant",
            units=pyunits.mol / pyunits.m**3 / pyunits.s / pyunits.Pa**2,
        )

        # Activation Energy
        self.energy_activation = Param(
            default=3800, doc="Activation energy", units=pyunits.J / pyunits.mol
        )

    @declare_process_block_class("HDAReactionParameterBlock")
    class HDAReactionParameterData(ReactionParameterBlock):
        """
        Reaction Parameter Block Class
        """

        def build(self):
            """
            Callable method for Block construction.
            """
            super(HDAReactionParameterData, self).build()

            self._reaction_block_class = HDAReactionBlock

            define_kinetic_reactions(self)
            define_equilibrium_reactions(self)
            define_parameters(self)

        @classmethod
        def define_metadata(cls, obj):
            obj.add_properties(properties_metadata)
            obj.add_default_units(units_metadata)

    def define_variables_and_parameters(self):
        self.k_rxn = Var(
            initialize=7e-10,
            doc="Rate constant",
            units=pyunits.mol / pyunits.m**3 / pyunits.s / pyunits.Pa**2,
        )

        self.reaction_rate = Var(
            self.params.rate_reaction_idx,
            initialize=0,
            doc="Rate of reaction",
            units=pyunits.mol / pyunits.m**3 / pyunits.s,
        )

        self.k_eq = Param(initialize=10000, doc="Equlibrium constant", units=pyunits.Pa)

    def define_rate_expression(self):
        self.arrhenius_equation = Constraint(
            expr=self.k_rxn
            == self.params.arrhenius
            * exp(
                -self.params.energy_activation
                / (const.gas_constant * self.state_ref.temperature)
            )
        )

        def rate_rule(b, r):
            return b.reaction_rate[r] == (
                b.k_rxn
                * b.state_ref.mole_frac_comp["toluene"]
                * b.state_ref.mole_frac_comp["hydrogen"]
                * b.state_ref.pressure**2
            )

        self.rate_expression = Constraint(self.params.rate_reaction_idx, rule=rate_rule)

    def define_equilibrium_expression(self):
        self.equilibrium_constraint = Constraint(
            expr=self.k_eq
            * self.state_ref.mole_frac_comp["benzene"]
            * self.state_ref.pressure
            == self.state_ref.mole_frac_comp["diphenyl"]
            * self.state_ref.mole_frac_comp["hydrogen"]
            * self.state_ref.pressure**2
        )

    class _HDAReactionBlock(ReactionBlockBase):
        def initialize(blk, outlvl=idaeslog.NOTSET, **kwargs):
            init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")
            init_log.info("Initialization Complete.")

    @declare_process_block_class("HDAReactionBlock", block_class=_HDAReactionBlock)
    class HDAReactionBlockData(ReactionBlockDataBase):
        def build(self):

            super(HDAReactionBlockData, self).build()

            define_variables_and_parameters(self)
            define_rate_expression(self)
            define_equilibrium_expression(self)

        def get_reaction_rate_basis(b):
            return MaterialFlowBasis.molar

    m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.thermo_params = HDAParameterBlock()
    m.fs.reaction_params = HDAReactionParameterBlock(
        property_package=m.fs.thermo_params
    )

    m.fs.reactor = CSTR(
        property_package=m.fs.thermo_params,
        reaction_package=m.fs.reaction_params,
        has_equilibrium_reactions=True,
    )

    print("Degrees of Freedom: ", degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 9

    m.fs.reactor.inlet.flow_mol.fix(100)
    m.fs.reactor.inlet.temperature.fix(500)
    m.fs.reactor.inlet.pressure.fix(350000)
    m.fs.reactor.inlet.mole_frac_comp[0, "benzene"].fix(0.1)
    m.fs.reactor.inlet.mole_frac_comp[0, "toluene"].fix(0.4)
    m.fs.reactor.inlet.mole_frac_comp[0, "hydrogen"].fix(0.4)
    m.fs.reactor.inlet.mole_frac_comp[0, "methane"].fix(0.1)
    m.fs.reactor.inlet.mole_frac_comp[0, "diphenyl"].fix(0.0)

    m.fs.reactor.volume.fix(1)

    print("Degrees of Freedom: ", degrees_of_freedom(m))

    assert degrees_of_freedom(m) == 0

    m.fs.reactor.initialize(
        state_args={
            "flow_mol": 100,
            "mole_frac_comp": {
                "benzene": 0.15,
                "toluene": 0.35,
                "hydrogen": 0.35,
                "methane": 0.15,
                "diphenyl": 0.01,
            },
            "temperature": 600,
            "pressure": 350000,
        }
    )

    solver = get_solver()
    results = solver.solve(m, tee=True)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    m.fs.reactor.report()

    assert value(m.fs.reactor.outlet.flow_mol[0]) == pytest.approx(100, abs=1e-3)
    assert value(m.fs.reactor.outlet.temperature[0]) == pytest.approx(790.212, abs=1e-3)
    assert value(m.fs.reactor.outlet.mole_frac_comp[0, "benzene"]) == pytest.approx(
        0.159626, abs=1e-6
    )

    assert_units_consistent(m)
