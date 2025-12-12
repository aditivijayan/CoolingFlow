# from constants import *
# from config import *

# def c(Mhalo):
#     h_inv = H0/100.0
#     log10_c = 1.025 - 0.097 * np.log10(Mhalo/1.e12/Msun/h_inv)
#     return 10.**log10_c

# def Rvir(Mhalo):
#     R_vir = 260.0  * (Mhalo/1.e12/Msun)**(1./3.) * kpc
#     return R_vir

# def NFW(r, Mhalo):
#     Rvir_ = Rvir(Mhalo)
#     Rs = Rvir_/c(Mhalo)
#     rho_r = 1/(r/Rs)/(1+r/Rs)**2
#     return rho_r * 4. * math.pi * r * r

# def NFW_rho(r, Mhalo):
#     Rs = Rvir(Mhalo)/c(Mhalo)
#     rho_r = 1/(r/Rs)/(1+r/Rs)**2
#     return rho_r 

# def mass_enc_halo(r, Mhalo):
#     Rs = Rvir(Mhalo)/c(Mhalo) 
#     I_NFW = integrate.quad(NFW, 0, Rvir(Mhalo), args=(Mhalo))
#     rho0 = Mhalo/I_NFW[0]
#     rho_r = rho0/(r/Rs)/(1+r/Rs)**2
#     dm = 4. * math.pi * r * r * rho_r
#     return dm

# def R_half(Mhalo):
#     r = 3.0 * (Mhalo/1.e12/Msun) **(1./3.) * kpc
#     return r

# def M_star_enc(r, Mhalo):
#     Mstar = 0.01 * Mhalo
#     enc_stellar_mass = Mstar * r/(r + R_half(Mhalo))
#     return enc_stellar_mass


# def mass_enc_outer(r, Mhalo):
#     R200 = 200. * (Mhalo/1.e12/Msun) ** (1./3.) * kpc
#     rho_m = 2.88e-30 #g/cm3
#     be = 1.0
#     se = 1.5
#     rho = rho_m * (be * (r/(5.0*R200))**(-se) + 1)
#     dm = (4. * math.pi * r * r * rho)
#     return dm 


# def tot_mass_enclosed(r,Mhalo):
#     Rvir_ = Rvir(Mhalo)
#     I_NFW = integrate.quad(NFW, 0, Rvir_, args=(Mhalo))
#     central_rho = Mhalo/I_NFW[0]
#     mass_nfw = integrate.quad(mass_enc_halo, 0.00001*kpc, r, args=(Mhalo))[0]
#     mass_enclosed_outer = integrate.quad(mass_enc_outer, 0.00001*kpc, r, args=(Mhalo))[0]
#     mass_stellar = M_star_enc(r, Mhalo)
#     total_mass = mass_nfw + mass_enclosed_outer + mass_stellar
#     return total_mass


# def v_c(r, Mhalo):
#     Mgrav = tot_mass_enclosed(r, Mhalo)
#     return np.sqrt(G*Mgrav/r)


# def T_vir(Mhalo):
#     Tvir = 6.e5 * (Mhalo/(1.e12*Msun))**(2./3.)
#     return Tvir




from constants import *
from config import *

def c(Mhalo):
    h_inv = 1./0.67
    log10_c = 1.025 - 0.097 * np.log10(Mhalo/(1e12*Msun)/h_inv)
    return 10**log10_c

def Rvir(Mhalo):
    return 260.0 * (Mhalo/1e12/Msun)**(1/3) * kpc

def NFW_shape(r, Rs):
    x = r / Rs
    return 1.0 / (x * (1 + x)**2)

def NFW_density(r, Mhalo):
    Rvir_ = Rvir(Mhalo)
    Rs = Rvir_ / c(Mhalo)

    # normalization
    I = integrate.quad(lambda r: 4*np.pi*r*r*NFW_shape(r, Rs),
                       1e-5*kpc, Rvir_, args=())[0]
    rho0 = Mhalo / I

    return rho0 * NFW_shape(r, Rs)

def NFW_mass_enclosed(r, Mhalo):
    Rvir_ = Rvir(Mhalo)
    Rs = Rvir_ / c(Mhalo)

    # normalization
    I = integrate.quad(lambda r: 4*np.pi*r*r*NFW_shape(r, Rs),
                       1e-5*kpc, Rvir_)[0]
    rho0 = Mhalo / I

    # enclosed mass
    M = integrate.quad(lambda r: 4*np.pi*r*r*rho0*NFW_shape(r, Rs),
                       1e-5*kpc, r)[0]
    return M

def M_star_enc(r, Mhalo, Ms_Mh):
    Mstar = Ms_Mh * Mhalo
    Rhalf = 3.0*(Mhalo/1e12/Msun)**(1/3)*kpc
    return Mstar * r / (r + Rhalf)

def mass_outer_enc(r, Mhalo):
    R200 = 200.0*(Mhalo/1e12/Msun)**(1/3)*kpc
    rho_m = 2.88e-30
    be, se = 1.0, 1.5

    rho = rho_m * (be*(r/(5*R200))**(-se) + 1)
    M = integrate.quad(lambda r: 4*np.pi*r*r*rho,
                       1e-5*kpc, r)[0]
    return M

def tot_mass_enclosed(r, Mhalo, Ms_Mh):
    return (NFW_mass_enclosed(r, Mhalo)
            + mass_outer_enc(r, Mhalo)
            + M_star_enc(r, Mhalo, Ms_Mh))

def v_c(r, Mhalo, Ms_Mh):
    M = tot_mass_enclosed(r, Mhalo, Ms_Mh)
    return np.sqrt(G * M / r)

def T_vir(Mhalo):
    return 6e5 * (Mhalo/(1e12*Msun))**(2/3)

