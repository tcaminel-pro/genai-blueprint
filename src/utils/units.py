"""
This module
- handles initialization of Pint functionality
- define units related to Emission Factors
- provide a convenient singleton for UNIT_REGISTRY

Copyright (C) 2024 Eviden. All rights reserved
"""

# NOT TESTED

import pint
from loguru import logger
from pint.registry import ApplicationRegistry
from pydantic import BaseModel, ConfigDict
from pydantic_pint import PydanticPintQuantity
from typing_extensions import Annotated

from src.utils.singleton import once

Density = Annotated[pint.Quantity, PydanticPintQuantity("kg/m^3")]


class UnitRegistry(BaseModel):
    reg: ApplicationRegistry

    @once()
    def instance():
        """
        Singleton access to the Pint Unit Registry.  We also define some application specific units and aliases.
        """

        logger.info("register units")
        r = pint.get_application_registry()
        r.default_format = ".3f"
        try:
            r.define("unit = [] = Unit = item = Item")

            r.define("cm2 = cm^2")
            r.define("cm3 = cm^3")
            r.define("m2 = m^2")
            r.define("m3 = m^3")
            r.define("Kg = kg")

            # see also : https://github.com/IAMconsortium/units/blob/main/iam_units/data/
            r.define("CO2e = [GWP] = CO2 = CO2eq = CO2_eq")
            r.define("kgCO2e = [GWPxmass] = kgCO2 = kgCO2eq = kgCO2_eq")
            r.define("fraction = [] = frac")
        except pint.errors.RedefinitionError:
            pass

        return UnitRegistry(reg=r)

    def from_str(self, str_unit: str) -> pint.Quantity:
        """Convert string unit to Pint object"""
        match str_unit:
            case "metric ton*km":
                new_unit = "metric_ton*km"
            case "1 mail", "Line":
                new_unit = "item"
            case "kg CO2e/$USD":
                new_unit = "kg.CO2e/USD"
            case _:
                new_unit = str_unit
        return self.reg(new_unit)

    def get_unit_base(self, unit: str) -> str:
        """Get base unit from unit registry"""

        unit_obj = self.reg(unit)
        if unit_obj.check("[mass]"):
            base_unit = self.reg.get_symbol(str(unit_obj.to(self.reg.gram).units))
        elif unit_obj.check("[volume]"):
            base_unit = self.reg.get_symbol(str(unit_obj.to(self.reg.liter).units))
        elif unit_obj.check("[length]"):
            base_unit = self.reg.get_symbol(str(unit_obj.to(self.reg.meter).units))
        elif unit_obj.check("[]"):
            base_unit = unit
        else:
            logger.error(f"cannot get unit base, not type found for :{unit}")
            base_unit = unit
        return base_unit

    model_config = ConfigDict(arbitrary_types_allowed=True)
