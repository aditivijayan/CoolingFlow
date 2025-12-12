from constants import *
from functions import *

class Units:
    def __init__(self, runit=None, rho_unit=None, vunit=None):
        # Default values (if not provided)
        if runit is None:
            runit = kpc
        if rho_unit is None:
            rho_unit = mp
        if vunit is None:
            vunit = 10.0 * kmps

        # Derived quantities
        self.runit = runit
        self.rho_unit = rho_unit
        self.vunit = vunit
        self.tunit = runit / vunit
        self.mass_unit = rho_unit * runit**3
        self.mdot_unit = self.mass_unit / self.tunit
        self.press_unit = rho_unit * vunit**2
