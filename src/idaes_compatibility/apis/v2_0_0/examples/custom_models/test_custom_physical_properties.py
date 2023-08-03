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
    Expression,
    Reference,
    Param,
    units as pyunits,
    Var,
    ConcreteModel,
    value,
)
from pyomo.util.check_units import assert_units_consistent

from idaes.core import (
    FlowsheetBlock,
    declare_process_block_class,
    MaterialFlowBasis,
    PhysicalParameterBlock,
    StateBlockData,
    StateBlock,
    MaterialBalanceType,
    EnergyBalanceType,
    Component,
    VaporPhase,
)
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import (
    fix_state_vars,
    revert_state_vars,
    solve_indexed_blocks,
)
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_unfixed_variables,
)
from idaes.core.util.constants import Constants as const
import idaes.logger as idaeslog
from idaes.core import VaporPhase, Component
from idaes.models.properties.modular_properties.state_definitions import FTPx
from idaes.models.properties.modular_properties.eos.ideal import Ideal
import idaes.models.properties.modular_properties.pure.RPP3 as RPP


def test_example():
    units_metadata = {
        "time": pyunits.s,
        "length": pyunits.m,
        "mass": pyunits.kg,
        "amount": pyunits.mol,
        "temperature": pyunits.K,
    }

    properties_metadata = {
        "flow_mol": {"method": None},
        "mole_frac_comp": {"method": None},
        "temperature": {"method": None},
        "pressure": {"method": None},
        "mw_comp": {"method": None},
        "dens_mol": {"method": None},
        "enth_mol": {"method": "_enth_mol"},
    }

    def define_components_and_phases(self):
        # Define Component objects for all species
        self.benzene = Component()
        self.toluene = Component()
        self.methane = Component()
        self.hydrogen = Component()
        self.diphenyl = Component()

        # Define Phase objects for all phases
        self.Vap = VaporPhase()

    def define_basic_parameters(self):
        # Thermodynamic reference state
        self.pressure_ref = Param(
            mutable=True, default=101325, units=pyunits.Pa, doc="Reference pressure"
        )
        self.temperature_ref = Param(
            mutable=True, default=298.15, units=pyunits.K, doc="Reference temperature"
        )

        self.mw_comp = Param(
            self.component_list,
            mutable=False,
            initialize={
                "benzene": 78.1136e-3,
                "toluene": 92.1405e-3,
                "hydrogen": 2.016e-3,
                "methane": 16.043e-3,
                "diphenyl": 154.212e-4,
            },
            units=pyunits.kg / pyunits.mol,
            doc="Molecular weight",
        )

    def define_specific_heat_parameters(self):
        # Constants for specific heat capacity, enthalpy
        self.cp_mol_ig_comp_coeff_A = Var(
            self.component_list,
            initialize={
                "benzene": -3.392e1,
                "toluene": -2.435e1,
                "hydrogen": 2.714e1,
                "methane": 1.925e1,
                "diphenyl": -9.707e1,
            },
            units=pyunits.J / pyunits.mol / pyunits.K,
            doc="Parameter A for ideal gas molar heat capacity",
        )
        self.cp_mol_ig_comp_coeff_A.fix()

        self.cp_mol_ig_comp_coeff_B = Var(
            self.component_list,
            initialize={
                "benzene": 4.739e-1,
                "toluene": 5.125e-1,
                "hydrogen": 9.274e-3,
                "methane": 5.213e-2,
                "diphenyl": 1.106e0,
            },
            units=pyunits.J / pyunits.mol / pyunits.K**2,
            doc="Parameter B for ideal gas molar heat capacity",
        )
        self.cp_mol_ig_comp_coeff_B.fix()

        self.cp_mol_ig_comp_coeff_C = Var(
            self.component_list,
            initialize={
                "benzene": -3.017e-4,
                "toluene": -2.765e-4,
                "hydrogen": -1.381e-5,
                "methane": -8.855e-4,
                "diphenyl": -8.855e-4,
            },
            units=pyunits.J / pyunits.mol / pyunits.K**3,
            doc="Parameter C for ideal gas molar heat capacity",
        )
        self.cp_mol_ig_comp_coeff_C.fix()

        self.cp_mol_ig_comp_coeff_D = Var(
            self.component_list,
            initialize={
                "benzene": 7.130e-8,
                "toluene": 4.911e-8,
                "hydrogen": 7.645e-9,
                "methane": -1.132e-8,
                "diphenyl": 2.790e-7,
            },
            units=pyunits.J / pyunits.mol / pyunits.K**4,
            doc="Parameter D for ideal gas molar heat capacity",
        )
        self.cp_mol_ig_comp_coeff_D.fix()

        self.enth_mol_form_vap_comp_ref = Var(
            self.component_list,
            initialize={
                "benzene": -82.9e3,
                "toluene": -50.1e3,
                "hydrogen": 0,
                "methane": -75e3,
                "diphenyl": -180e3,
            },
            units=pyunits.J / pyunits.mol,
            doc="Standard heat of formation at reference state",
        )
        self.enth_mol_form_vap_comp_ref.fix()

    @declare_process_block_class("HDAParameterBlock")
    class HDAParameterData(PhysicalParameterBlock):
        CONFIG = PhysicalParameterBlock.CONFIG()

        def build(self):
            """
            Callable method for Block construction.
            """
            super(HDAParameterData, self).build()

            self._state_block_class = HDAStateBlock

            define_components_and_phases(self)
            define_basic_parameters(self)
            define_specific_heat_parameters(self)

        @classmethod
        def define_metadata(cls, obj):
            """Define properties supported and units."""
            obj.add_properties(properties_metadata)

            obj.add_default_units(units_metadata)

    def add_state_variables(self):
        self.flow_mol = Var(
            initialize=1,
            bounds=(1e-8, 1000),
            units=pyunits.mol / pyunits.s,
            doc="Molar flow rate",
        )
        self.mole_frac_comp = Var(
            self.component_list,
            initialize=0.2,
            bounds=(0, None),
            units=pyunits.dimensionless,
            doc="Component mole fractions",
        )
        self.pressure = Var(
            initialize=101325,
            bounds=(101325, 400000),
            units=pyunits.Pa,
            doc="State pressure",
        )
        self.temperature = Var(
            initialize=298.15,
            bounds=(298.15, 1500),
            units=pyunits.K,
            doc="State temperature",
        )

    def return_state_var_dict(self):
        return {
            "flow_mol": self.flow_mol,
            "mole_frac_comp": self.mole_frac_comp,
            "temperature": self.temperature,
            "pressure": self.pressure,
        }

    def add_molecular_weight_and_density(self):
        self.mw_comp = Reference(self.params.mw_comp)

        self.dens_mol = Var(
            initialize=1, units=pyunits.mol / pyunits.m**3, doc="Mixture density"
        )

        self.ideal_gas_eq = Constraint(
            expr=self.pressure == const.gas_constant * self.temperature * self.dens_mol
        )

    def add_enth_mol(self):
        def enth_rule(b):
            params = self.params
            T = self.temperature
            Tr = params.temperature_ref
            return sum(
                self.mole_frac_comp[j]
                * (
                    (params.cp_mol_ig_comp_coeff_D[j] / 4) * (T**4 - Tr**4)
                    + (params.cp_mol_ig_comp_coeff_C[j] / 3) * (T**3 - Tr**3)
                    + (params.cp_mol_ig_comp_coeff_B[j] / 2) * (T**2 - Tr**2)
                    + params.cp_mol_ig_comp_coeff_A[j] * (T - Tr)
                    + params.enth_mol_form_vap_comp_ref[j]
                )
                for j in self.component_list
            )

        self.enth_mol = Expression(rule=enth_rule)

    def add_mole_fraction_constraint(self):
        if self.config.defined_state is False:
            self.mole_fraction_constraint = Constraint(
                expr=1e3
                == sum(1e3 * self.mole_frac_comp[j] for j in self.component_list)
            )

    def prepare_state(blk, state_args, state_vars_fixed):
        # Fix state variables if not already fixed
        if state_vars_fixed is False:
            flags = fix_state_vars(blk, state_args)
        else:
            flags = None

        # Deactivate sum of mole fractions constraint
        for k in blk.keys():
            if blk[k].config.defined_state is False:
                blk[k].mole_fraction_constraint.deactivate()

        # Check that degrees of freedom are zero after fixing state vars
        for k in blk.keys():
            if degrees_of_freedom(blk[k]) != 0:
                raise Exception(
                    "State vars fixed but degrees of freedom "
                    "for state block is not zero during "
                    "initialization."
                )

        return flags

    def initialize_state(blk, solver, init_log, solve_log):
        # Check that there is something to solve for
        free_vars = 0
        for k in blk.keys():
            free_vars += number_unfixed_variables(blk[k])
        if free_vars > 0:
            # If there are free variables, call the solver to initialize
            try:
                with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                    res = solve_indexed_blocks(solver, [blk], tee=True)  # slc.tee)
            except:
                res = None
        else:
            res = None

        init_log.info("Properties Initialized {}.".format(idaeslog.condition(res)))

    def restore_state(blk, flags, hold_state):
        # Return state to initial conditions
        if hold_state is True:
            return flags
        else:
            blk.release_state(flags)

    def unfix_state(blk, flags, outlvl):
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")

        # Reactivate sum of mole fractions constraint
        for k in blk.keys():
            if blk[k].config.defined_state is False:
                blk[k].mole_fraction_constraint.activate()

        if flags is not None:
            # Unfix state variables
            revert_state_vars(blk, flags)

        init_log.info_high("State Released.")

    class _HDAStateBlock(StateBlock):
        def initialize(
            blk,
            state_args=None,
            state_vars_fixed=False,
            hold_state=False,
            outlvl=idaeslog.NOTSET,
            solver=None,
            optarg=None,
        ):

            init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")
            solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="properties")

            # Create solver
            solver_obj = get_solver(solver, optarg)

            flags = prepare_state(blk, state_args, state_vars_fixed)
            initialize_state(blk, solver_obj, init_log, solve_log)
            restore_state(blk, flags, hold_state)

            init_log.info("Initialization Complete")

        def release_state(blk, flags, outlvl=idaeslog.NOTSET):
            unfix_state(blk, flags, outlvl)

    @declare_process_block_class("HDAStateBlock", block_class=_HDAStateBlock)
    class HDAStateBlockData(StateBlockData):
        """
        Example property package for an ideal gas containing benzene, toluene
        hydrogen, methane and diphenyl.
        """

        def build(self):
            """Callable method for Block construction."""
            super(HDAStateBlockData, self).build()

            add_state_variables(self)
            add_mole_fraction_constraint(self)
            add_molecular_weight_and_density(self)

        def _enth_mol(self):
            add_enth_mol(self)

        def define_state_vars(self):
            return return_state_var_dict(self)

        def get_material_flow_terms(self, p, j):
            return self.flow_mol * self.mole_frac_comp[j]

        def get_enthalpy_flow_terms(self, p):
            """Create enthalpy flow terms."""
            return self.flow_mol * self.enth_mol

        def default_material_balance_type(self):
            return MaterialBalanceType.componentPhase

        def default_energy_balance_type(self):
            return EnergyBalanceType.enthalpyTotal

        def get_material_flow_basis(self):
            return MaterialFlowBasis.molar

    m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.thermo_props = HDAParameterBlock()

    m.fs.state = m.fs.thermo_props.build_state_block(
        m.fs.config.time, defined_state=True
    )

    m.fs.state.display()
    print("Degrees of freedom: ", degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 2

    m.fs.state[0].flow_mol.fix(100)
    m.fs.state[0].temperature.fix(500)
    m.fs.state[0].pressure.fix(350000)
    m.fs.state[0].mole_frac_comp["benzene"].fix(0.1)
    m.fs.state[0].mole_frac_comp["toluene"].fix(0.4)
    m.fs.state[0].mole_frac_comp["hydrogen"].fix(0.4)
    m.fs.state[0].mole_frac_comp["methane"].fix(0.1)
    m.fs.state[0].mole_frac_comp["diphenyl"].fix(0.0)

    print("Degrees of freedom: ", degrees_of_freedom(m))

    m.fs.state.initialize()

    m.fs.state[0].dens_mol.display()

    assert value(m.fs.state[0].dens_mol) == pytest.approx(84.191, abs=1e-3)

    m.fs.state[0].enth_mol.display()
    assert value(m.fs.state[0].enth_mol) == pytest.approx(-22170, abs=1e-1)

    assert_units_consistent(m)

    # Build configuration dictionary
    configuration = {
        # Specifying components
        "components": {
            "benzene": {
                "type": Component,
                "enth_mol_ig_comp": RPP,
                "parameter_data": {
                    "mw": (78.1136e-3, pyunits.kg / pyunits.mol),
                    "cp_mol_ig_comp_coeff": {
                        "A": (-3.392e1, pyunits.J / pyunits.mol / pyunits.K),
                        "B": (4.739e-1, pyunits.J / pyunits.mol / pyunits.K**2),
                        "C": (-3.017e-4, pyunits.J / pyunits.mol / pyunits.K**3),
                        "D": (7.130e-8, pyunits.J / pyunits.mol / pyunits.K**4),
                    },
                    "enth_mol_form_vap_comp_ref": (82.9e3, pyunits.J / pyunits.mol),
                },
            },
            "toluene": {
                "type": Component,
                "enth_mol_ig_comp": RPP,
                "parameter_data": {
                    "mw": (92.1405e-3, pyunits.kg / pyunits.mol),
                    "cp_mol_ig_comp_coeff": {
                        "A": (-2.435e1, pyunits.J / pyunits.mol / pyunits.K),
                        "B": (5.125e-1, pyunits.J / pyunits.mol / pyunits.K**2),
                        "C": (-2.765e-4, pyunits.J / pyunits.mol / pyunits.K**3),
                        "D": (4.911e-8, pyunits.J / pyunits.mol / pyunits.K**4),
                    },
                    "enth_mol_form_vap_comp_ref": (50.1e3, pyunits.J / pyunits.mol),
                },
            },
            "hydrogen": {
                "type": Component,
                "enth_mol_ig_comp": RPP,
                "parameter_data": {
                    "mw": (2.016e-3, pyunits.kg / pyunits.mol),
                    "cp_mol_ig_comp_coeff": {
                        "A": (2.714e1, pyunits.J / pyunits.mol / pyunits.K),
                        "B": (9.274e-3, pyunits.J / pyunits.mol / pyunits.K**2),
                        "C": (-1.381e-5, pyunits.J / pyunits.mol / pyunits.K**3),
                        "D": (7.645e-9, pyunits.J / pyunits.mol / pyunits.K**4),
                    },
                    "enth_mol_form_vap_comp_ref": (0, pyunits.J / pyunits.mol),
                },
            },
            "methane": {
                "type": Component,
                "enth_mol_ig_comp": RPP,
                "parameter_data": {
                    "mw": (16.043e-3, pyunits.kg / pyunits.mol),
                    "cp_mol_ig_comp_coeff": {
                        "A": (1.925e1, pyunits.J / pyunits.mol / pyunits.K),
                        "B": (5.213e-2, pyunits.J / pyunits.mol / pyunits.K**2),
                        "C": (-8.855e-4, pyunits.J / pyunits.mol / pyunits.K**3),
                        "D": (-1.132e-8, pyunits.J / pyunits.mol / pyunits.K**4),
                    },
                    "enth_mol_form_vap_comp_ref": (-75e3, pyunits.J / pyunits.mol),
                },
            },
            "diphenyl": {
                "type": Component,
                "enth_mol_ig_comp": RPP,
                "parameter_data": {
                    "mw": (154.212e-4, pyunits.kg / pyunits.mol),
                    "cp_mol_ig_comp_coeff": {
                        "A": (-9.707e1, pyunits.J / pyunits.mol / pyunits.K),
                        "B": (1.106e0, pyunits.J / pyunits.mol / pyunits.K**2),
                        "C": (-8.855e-4, pyunits.J / pyunits.mol / pyunits.K**3),
                        "D": (2.790e-7, pyunits.J / pyunits.mol / pyunits.K**4),
                    },
                    "enth_mol_form_vap_comp_ref": (-180e3, pyunits.J / pyunits.mol),
                },
            },
        },
        # Specifying phases
        "phases": {"Vap": {"type": VaporPhase, "equation_of_state": Ideal}},
        # Set base units of measurement
        "base_units": {
            "time": pyunits.s,
            "length": pyunits.m,
            "mass": pyunits.kg,
            "amount": pyunits.mol,
            "temperature": pyunits.K,
        },
        # Specifying state definition
        "state_definition": FTPx,
        "state_bounds": {
            "flow_mol": (1e-8, 1, 1000, pyunits.mol / pyunits.s),
            "temperature": (298.15, 298.15, 1500, pyunits.K),
            "pressure": (101325, 101325, 400000, pyunits.Pa),
        },
        "pressure_ref": (101325, pyunits.Pa),
        "temperature_ref": (298.15, pyunits.K),
    }
