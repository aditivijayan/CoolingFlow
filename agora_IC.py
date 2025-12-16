from constants import *
from functions import *
import numpy as np
from scipy.integrate import solve_ivp
import math
from units import *
from scipy.optimize import root_scalar
from numba import njit
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator

plt.rcParams['font.size']=22
plt.rcParams['axes.linewidth']=2
plt.rcParams['xtick.major.size']=10
plt.rcParams['xtick.minor.size']=5
plt.rcParams['xtick.major.width']=2
plt.rcParams['xtick.minor.width']=1
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.major.size']=10
plt.rcParams['ytick.minor.size']=5
plt.rcParams['ytick.major.width']=2
plt.rcParams['ytick.minor.width']=1
plt.rcParams['ytick.direction']='in'


filename = "CloudyData_UVB=HM2012.h5"
Z = 0.3
with h5py.File(filename, "r") as f:
    cloudy_table = f["CoolingRates/Metals/Cooling"][:,0,:]
    mmw = f["CoolingRates/Primordial/MMW"][:,0,:]

cloudy_table *= Z
log10nH_arr = np.linspace(-10.,4, cloudy_table.shape[0])
log10temp_arr = np.linspace(1.,9, cloudy_table.shape[1])

log10lambda_cloudy = interpolate.RegularGridInterpolator(
    (log10nH_arr, log10temp_arr),
    np.log10(cloudy_table),
    bounds_error=False,
    fill_value=None
)

mmw_cloudy = interpolate.RegularGridInterpolator(
    (log10nH_arr, log10temp_arr),
    mmw,
    bounds_error=False,
    fill_value=None
)




rho_arr = 10.**log10nH_arr
temp_arr =10**log10temp_arr
lnT   = np.log(temp_arr)
lnrho = np.log(rho_arr)
lnLambda = np.log(cloudy_table)

dlnLambda_dlnrho, dlnLambda_dlnT = np.gradient(
    lnLambda,
    lnrho,
    lnT,
    edge_order=2
)

interp_dlnLambda_dlnrho = RegularGridInterpolator(
    (lnrho, lnT),
    dlnLambda_dlnrho,
    bounds_error=False,
    fill_value=None
)

interp_dlnLambda_dlnT = RegularGridInterpolator(
    (lnrho, lnT),
    dlnLambda_dlnT,
    bounds_error=False,
    fill_value=None
)


X = 0.75
mu = 0.61

Mhalo = 1.074e12* Msun
Ms_Mh = 3.438e10*Msun/Mhalo


data = np.loadtxt('vcirc_agora.txt')
rad = data[:, 0]
vcirc = data[:, 1]
vcirc_interp = interp1d(rad,vcirc)

r_grid = np.logspace(np.log10(0.001*kpc), np.log10(25*Rvir(Mhalo)), 1000)
vc_grid = np.zeros(r_grid.shape[0])
i = 0
for r in r_grid:
    r=r/kpc
    if(r>np.amax(rad)):
        vc_grid[i] = vcirc[-1]*kpc
        
    elif(r<np.amin(rad)):
        vc_grid[i] = vcirc[0]*kpc
        
    else:
        vc_grid[i] = vcirc_interp(r)*kmps
    i+=1

vc_interp = interp1d(r_grid,vc_grid)



lnvc_grid = np.log(vc_grid)
lnr_grid  = np.log(r_grid)

dlnvc_dlnr_grid = np.gradient(lnvc_grid, lnr_grid)
dlnvc_dlnr_grid_interp =  interp1d(r_grid,dlnvc_dlnr_grid)



def soln_at_esp(v_rsonic, T_rsonic, rho_rsonic, rsonic, eps):

    dlnvc_dlnr_at_Rsonic = dlnvc_dlnr_grid_interp(rsonic)
    
    vc_rsonic  = vc_interp(rsonic)
    
    cs_rsonic  = v_rsonic
    x = (vc_rsonic/cs_rsonic)**2/2.
    a = 1.0
    
    lnrho = np.log(rho_rsonic)
    lnT = np.log(T_rsonic)
                 
    constant1 = interp_dlnLambda_dlnT((lnrho, lnT)) + 1.5 *  interp_dlnLambda_dlnrho((lnrho, lnT))
    b = 29.0* x/6 - 17.0/6.0 + ((1.-x)/3.) * constant1
    
    constant2 = interp_dlnLambda_dlnrho((lnrho, lnT))
    c = ((2./3.) * x * dlnvc_dlnr_at_Rsonic) +  (5*x*x) - (13.*x/3.0) + (2./3.)  - (5.*(1-x)**2/3.)*constant2
    
    if ((b*b - 4*a*c)<0):
        print('No transsonic soln!')
        return None, None, None, None
    else:
        sol1 = (-b - np.sqrt(b*b - 4*a*c))/(2.*a*c)
        sol2 = (-b + np.sqrt(b*b - 4*a*c))/(2.*a*c)
        sol = np.asarray([sol1,sol2])
        dlnv_dlnr_at_Rsonic = -1.5* sol + 3 - 5*x
        dlnM_dlnr = dlnv_dlnr_at_Rsonic - 0.5 * sol
        
        if(dlnM_dlnr[0]<0.0):   
            dlnT_dlnr_at_Rsonic = sol1
        elif(dlnM_dlnr[1]<0.0):
            dlnT_dlnr_at_Rsonic = sol2
        
        dlnv_dlnr_at_Rsonic = -1.5* dlnT_dlnr_at_Rsonic + 3 - 5*x
        
        delT = 1. + dlnT_dlnr_at_Rsonic * eps
        delv = 1. + dlnv_dlnr_at_Rsonic * eps
        delrho = 1. - (dlnv_dlnr_at_Rsonic + 2.) * eps
        
        T_ini = delT * T_rsonic
        v_ini = delv * v_rsonic
        rho_ini = delrho * rho_rsonic
        r_ini = (1.0 + eps) * rsonic
        cs_ini = np.sqrt(gamma * kb * T_ini / mu / mp)
        Mach_ini = v_ini/cs_ini

        return r_ini, v_ini, T_ini, rho_ini, Mach_ini



 def find_init_at_Rsonic_varying_lambda(rsonic, T):

    # Sonic velocity
    cs = np.sqrt(gamma * kb * T / mu / mp)
    v  = cs
    vc = vc_interp(rsonic)

    # Flow time
    tflow = rsonic / v

    # ---- Build a grid in nH ----
    nHs = 10.0**np.linspace(-10.,4, cloudy_table.shape[0])

    # Pressures
    Ps = (nHs * kb * T) / (X * mu)

    log10nH = np.log10(nHs)
    log10T = np.log10(T)
    lambda_cool = 10.**(log10lambda_cloudy((log10nH, log10T))) 
    tcools = Ps/(gamma-1)/ (nHs**2 * lambda_cool)

    # Compute f(nH):
    # f = 2 - (vc/v)^2 - tflow/(gamma*tcool)
    fvals_np = 2.0  - (vc/v)**2 - (tflow / (gamma * tcools))

    sign_changes = np.where(
        np.sign(fvals_np[:-1]) * np.sign(fvals_np[1:]) < 0)[0]

    if len(sign_changes) != 1:
        raise RuntimeError(
            f"Could not find unique root for nH; number of sign changes = {len(sign_changes)}"
        )

    i = sign_changes[0]

    # ---- Interpolate to find nH where f = 0 ----
    nH_low  = nHs[i]
    nH_high = nHs[i+1]
    f_low   = fvals_np[i]
    f_high  = fvals_np[i+1]
    
    # linear interpolation in log space for stability
    lnH = np.interp(
        0.0,
        [f_low, f_high],
        [np.log10(nH_low), np.log10(nH_high)]
    )

    best_nH = (10.0**lnH)

    # ---- Compute rho and Mdot ----
    rho = best_nH * mp / X

    Mdot = 4.0 * np.pi * rho * rsonic**2 * v
    Mach = v/cs

    return v, rho, Mdot, Mach



def flow_equations_lnr(lnr, ini_val, Mdot):
    
    logT, logrho = ini_val
    rho_max = 10.*mp
    rho_min = 1.e-8*mp
    T_max = 1.e7
    T_min = 1.e3
    
    if(logT>np.log(T_max)):
        logT = np.log(T_max)
    
    if(logT<np.log(T_min)):
        logT = np.log(T_min)

    if(logrho>np.log(rho_max)):
        logrho = np.log(rho_max)
    
    if(logrho<np.log(rho_min)):
        logrho = np.log(rho_min)
    
    
    T = np.exp(logT)
    rho = np.exp(logrho)
    r   = np.exp(lnr)
    
    nH_max = X*rho_max/mp    
    v   = Mdot/(4.*math.pi*r*r*rho)   
    
    
    nH = X * rho/mp
    cs = np.sqrt(gamma * kb * T/(mu * mp))
    lambda_cool = 10.**(log10lambda_cloudy((np.log10(nH), np.log10(T)))) 
    
    tcool = (kb*T/mu/X)/((gamma-1) * nH *  lambda_cool)
    
    Mach =  np.abs(v)/cs
    tflow = r/np.abs(v)
    tratio =  tflow/tcool
    
    vc_over_cs = (vc_interp(r)/cs)**2

    dlnrho_dlnr =  (-tratio/gamma - vc_over_cs + 2*Mach**2)  / (1-Mach**2)
    dlnT_dlnr = tratio + dlnrho_dlnr*(gamma-1)


    return dlnT_dlnr, dlnrho_dlnr


def find_sol(v_rsonic, T_rsonic, rho_rsonic, Mdot, rsonic, eps):
    rtol = eps**2
    
    rmin  =  rsonic*(1.+eps)
    # rmax  =  100.* kpc # Rvir(Mhalo)
    rmax  =  20.* Rvir(Mhalo)
    
    r_ini, v_ini, T_ini, rho_ini, M = soln_at_esp(v_rsonic, T_rsonic, rho_rsonic, rsonic, eps)
   
    #Solve equation in lnr
    ini_value = [T_ini, rho_ini]
    
    log_ini_value = np.log(ini_value)
    lnr_span = (np.log(r_ini), np.log(rmax))
    lnr_eval = np.linspace(lnr_span[0], lnr_span[1], 1000)

    def mach_gt1_event(lnr, y, Mdot):
        logT, logrho = y
        T = np.exp(logT)
        rho = np.exp(logrho)
        r = np.exp(lnr) 
        v = Mdot / (4 * np.pi * r**2 * rho)
        cs = np.sqrt(gamma * kb * T / (mu * mp))
        Mach = abs(v) / cs
        
        return Mach - 1 if r/kpc > 100 else -1  

    mach_gt1_event.terminal = True
    mach_gt1_event.direction = 1  # only trigger when Mach is increasing
    
    sol = solve_ivp(flow_equations_lnr, lnr_span, log_ini_value, t_eval=lnr_eval, method='LSODA',args=(Mdot,), max_step=0.1,\
                 atol=1e-6,rtol=rtol, events=mach_gt1_event)
   
    lnT_sol, lnrho_sol = sol.y
    return sol, Mdot

rsonic = 0.5*kpc
vc = vc_interp(rsonic)
T_rsonic  =  mu * mp * vc**2/gamma/kb
print(T_rsonic)
T_rsonic  =500000 

eps=1.e-1
solution, Mdot1 =find_sol(v_rsonic, T_rsonic, rho_rsonic, Mdot, rsonic, eps)
if solution.t_events[0].size > 0:
    print(f"Stopped at r = {np.exp(solution.t_events[0][0])/kpc:.2f} kpc because Mach>1")
else:
    print("Solver finished normally")

T = np.exp(solution.y[0])
rho = np.exp(solution.y[1])
lnr = (solution.t)
cs = np.sqrt(gamma * kb * T/(0.6 * mp))
rkpc = np.exp(lnr)/kpc
r = rkpc*kpc
v = Mdot/(4*math.pi*r*r*rho)
Mach = v/cs
tflow = np.exp(lnr)/v
tff = np.sqrt(2) * r/v

tcool = np.zeros(rkpc.shape[0])
lambda_cool = np.zeros(rkpc.shape[0])
for i in range(tcool.shape[0]):
    P = kb * rho[i] * T[i]/0.6/mp
    log10nH = np.log10(X * rho[i]/mp)
    log10T  = np.log10(T[i])
    cloudy = 10.**((log10lambda_cloudy((log10nH, log10T))))
    lambda_cool[i] = cloudy
    # cloudy = 0.6e-22
    tcool[i] = P/(nH*nH*cloudy*(gamma-1))


# lambda_cool = 0.6e-22
tratio = tcool/tflow
Mach_anl = 0.11 * (Mhalo/(1.e12*Msun))**(-0.72) * (Mdot/(Msun/yr_to_sec))**(0.5) * (lambda_cool/1.e-22)**(0.5) * (r/100/kpc)**(-0.3) 
nH_anl   = 1.6e-5 * (Mhalo/(1.e12*Msun))**(0.36) * (lambda_cool/1.e-22)**(-0.5) * (r/100/kpc)**(-1.6)
tcool_anl = 7.2 * (Mhalo/(1.e12*Msun))**(0.36) * (Mdot/(Msun/yr_to_sec))**(-0.5) * (lambda_cool/1.e-22)**(-0.5) * (r/100/kpc)**(1.4) 
T_anl = X * mu * tcool_anl * (gamma-1) * nH_anl * lambda_cool * 1.e9 * yr_to_sec/kb 
Rsonic = 0.06 * (Mhalo/(1.e12*Msun))**(-2.4) * (Mdot/(Msun/yr_to_sec))**(1.67) * (lambda_cool/1.e-22)**(0.36) * kpc
Rhalf = 3.0*(Mhalo/1e12/Msun)**(1/3)*kpc


fig, ax = plt.subplots(4, 1, gridspec_kw = {'wspace':0.1, 'hspace':0.0},figsize=(6, 18))

ylabel = [r'$\rho/m_p$', r'$T$ [K]', r'$M$', r'$t_{\rm cool}$']

ax[0].plot(rkpc, rho*X/mp)
ax[0].plot(rkpc, nH_anl)
ax[1].plot(rkpc, T)
ax[1].plot(rkpc, T_anl)
ax[2].plot(rkpc, Mach)
ax[2].plot(rkpc, Mach_anl)
ax[3].plot(rkpc, tcool/yr_to_sec/1.e9)
ax[3].plot(rkpc, tcool_anl)


ax[-1].set_xlabel('r [kpc]')
for i in range(4):
    ax[i].set_ylabel(ylabel[i])
plt.setp(ax, 'xscale', ('log'))
plt.setp(ax, 'yscale', ('log'))
# plt.setp(ax, 'xlim', (0.5,2.))
ax[0].set_ylim(1.e-6, 4.e-1)
ax[1].set_ylim(1.e3, 4.e6)
ax[2].set_ylim(1.e-2,1.e1)
ax[3].set_ylim(1.e-3,1.e3)

ax[0].tick_params(axis='x', which='both', labelbottom=False, top=True, bottom=True)
ax[1].tick_params(axis='x', which='both', labelbottom=False, top=True, bottom=True)
ax[-2].tick_params(axis='x', which='both', top=True, bottom=True, labelbottom=False)
ax[-1].tick_params(axis='x', which='both', top=True, bottom=True, labelbottom=True)
ax[0].set_title(r'$\dot{M}$=%.1e'%(Mdot*yr_to_sec/Msun) + r' $M_{\odot} \ \rm{yr}^{-1}$')
ax[0].text(0.6, 0.8, r'$M_{\rm halo}$=%.1e'%(Mhalo/Msun), transform=ax[0].transAxes)
# ax[0].text(0.6, 0.7, r'$M_{\rm gas}$=%.1e'%(tot_gas_mass/Msun), transform=ax[0].transAxes)
# ax[0].text(0.6, 0.7, r'$\rho_{\rm ini}$=%.1e'%(rho0/mp), transform=ax[0].transAxes)
# ax[0].text(0.6, 0.6, r'$T_{\rm ini}$=%.1e'%(T), transform=ax[0].transAxes)

image_name = 'Figures/solution_%.2f'%(Mdot*yr_to_sec/Msun) + '_eps_' + str(eps) + '_rsonic_' + str(rsonic/kpc)   +'.jpeg'
plt.savefig(image_name, bbox_inches='tight')


T = np.exp(solution.y[0])
rho = np.exp(solution.y[1])
lnr = (solution.t)

cs = np.sqrt(gamma * kb * T/(mu * mp))

cs_mid = 0.5*(cs[:-1] + cs[1:])
rho_mid = 0.5 * (rho[:-1] + rho[1:])

r_grid = np.exp(lnr)
r_mid = 0.5 * (r_grid[:-1] + r_grid[1:])
dr_grid = np.diff(r_grid)

v_mid = Mdot/(4*math.pi*r_mid**2*rho_mid)

dphi = vc_interp(r_mid) **2 * dr_grid/r_mid
phi_int = -np.cumsum(dphi)
phi = -phi_int + phi_int[-1]
Bern_param = v_mid*v_mid/2.0 + cs_mid*cs_mid/(gamma-1) + phi

Rmax = 2000. * kpc
idx = np.argmin(np.abs(r - Rmax))
plt.plot(r_mid/kpc, (Bern_param/1.e5/kmps/kmps))
plt.axhline(0.0, color='black')
# plt.xlim(1.e-1, 1.e3)

plt.xscale('log')
print(Bern_param[-1]/1.e5/kmps/kmps)

image_name = 'Figures/bern_%.2f'%(Mdot*yr_to_sec/Msun) + '_eps_' + str(eps) + '_rsonic_' + str(rsonic/kpc)   +'.jpeg'
plt.savefig(image_name, bbox_inches='tight')
