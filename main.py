import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Parâmetros (como no seu patch)
# ------------------------------
Omega_m_0 = 0.301
Omega_S_0 = 0.649
Omega_k_0 = -0.069
Gamma_S = 0.958
lambda_phi = 1.18
xi = 0.102
H0 = 70.0            # km/s/Mpc
sigma8_0 = 0.810

# ------------------------------
# Integrador RK4 (clássico)
# ------------------------------
def rk4_step(N, Y, dN, f):
    k1 = f(N, Y)
    k2 = f(N + 0.5*dN, Y + 0.5*dN*k1)
    k3 = f(N + 0.5*dN, Y + 0.5*dN*k2)
    k4 = f(N + dN, Y + dN*k3)
    return Y + (dN/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ------------------------------
# Sistema DUT (versão com proteções)
# ------------------------------
def dut_rhs(N, Y):
    # Estados (sem clipping agressivo)
    x, y, u, z = Y

    # limites amplos para evitar overflow numérico extremo
    x = np.clip(x, -1e6, 1e6)
    y = np.clip(y, -1e6, 1e6)
    u = np.clip(u, -1e8, 1e8)
    z = np.clip(z, -1e6, 1e6)

    # quantidades auxiliares
    a = np.exp(N)
    x2 = x**2
    y2 = y**2

    Om_m = u / a**3                 # densidade de matéria "bruta" (não normalizada por H^2)
    Om_k = Omega_k_0 / a**2

    # H^2/H0^2 normalizado (garantir não-negativo)
    H2_over_H02 = Om_m + x2 + y2 + z*(1 - Gamma_S) + Om_k
    H2_over_H02 = np.maximum(H2_over_H02, 1e-18)

    # simplificação para Ricci-like (mantemos sinal)
    R_over_6H02 = H2_over_H02 + 0.5*(x2 - y2)

    combo = x2 - y2 + z*(1 - Gamma_S)

    # Equações (formas conforme seu enunciado; mantenho coeficientes)
    dxdN = -3.0*x + (np.sqrt(6.0)/2.0) * lambda_phi * y2 + 1.5*x*combo
    dydN = -(np.sqrt(6.0)/2.0) * lambda_phi * x * y + 1.5*y*combo
    dudN = -3.0*u - 1.5*u*combo
    dzdN = xi*(x2 - y2) + 6.0*xi*z*R_over_6H02

    return np.array([dxdN, dydN, dudN, dzdN], dtype=float)

# ------------------------------
# Grades e condições iniciais
# ------------------------------
N_points = 5000
N_start = -9.0
N_end = 20.0
N = np.linspace(N_start, N_end, N_points)
dN = N[1] - N[0]

# Para garantir Ω_m(N_start) = Omega_m_0, calculamos u_ini consistentemente:
a_start = np.exp(N_start)   # a(N_start)
# assumimos H2_over_H02_init ≈ Omega_m_0 + Omega_S_0 + Omega_k_0 (aprox. consistente)
# mas é mais robusto resolver u_ini tal que Om_m/H2 = Omega_m_0:
# tomamos inicialmente H2_over_H02_init ~ 1 (H/H0 ~ 1) como aproximação e corrigimos se necessário.
# mais rigoroso: iterar para encontrar u tal que (u/a^3)/H2 = Omega_m_0. Para simplicidade:
u_ini = Omega_m_0 * 1.0 * a_start**3   # se H2/H0^2 ~ 1 no início; é uma boa primeira aproximação

# outros estados: campo escalar congelado inicialmente
x_ini = 1e-6
y_ini = np.sqrt(Omega_S_0)
z_ini = xi * 1e-10

Y = np.array([x_ini, y_ini, u_ini, z_ini], dtype=float)

# Integrar
sol = np.zeros((N_points, 4), dtype=float)
sol[0] = Y
for i in range(1, N_points):
    sol[i] = rk4_step(N[i-1], sol[i-1], dN, dut_rhs)
    # proteção básica: se solução divergir, corta (evita Inf)
    if not np.isfinite(sol[i]).all():
        sol = sol[:i]
        N = N[:i]
        N_points = i
        break

# ------------------------------
# Observáveis e pós-processamento
# ------------------------------
x, y, u, z = sol.T
a = np.exp(N)
zc = 1.0/a - 1.0

Om_m_v = u / a**3
Om_k_v = Omega_k_0 / a**2
H2_oH0 = Om_m_v + x**2 + y**2 + z*(1 - Gamma_S) + Om_k_v
H2_oH0 = np.maximum(H2_oH0, 1e-18)
H = H0 * np.sqrt(H2_oH0)

# índices para hoje e CMB
idx0 = np.argmin(np.abs(zc - 0.0))
idx_cmb = np.argmin(np.abs(zc - 1100.0)) if np.any(np.isfinite(zc)) else 0

# H0 local (calibração verbo)
H0_loc = np.clip(H[idx0] * np.sqrt(1.08), 50.0, 120.0)

# w_eff (evita divisão por zero)
w_eff = (x**2 - y**2 + z*(1 - Gamma_S)/3.0) / H2_oH0
w_eff0 = np.clip(w_eff[idx0], -10.0, 10.0)

# G_eff e f
G_eff = 1.0 / np.maximum(1.0 + xi*z/3.0, 1e-12)
# fração de matéria normalizada localmente: Omega_m_fraction = (rho_m)/(H^2/H0^2)
Om_frac = Om_m_v / H2_oH0
Om_frac = np.clip(Om_frac, 0.0, 1.0)

f_growth = (Om_frac**0.55) * np.sqrt(np.clip(G_eff, 0.0, 1e6))

# sigma8: integramos a variação logarítmica do crescimento (modelo aproximado usado no original)
# definimos um vetor sigma8 com o mesmo tamanho de N
sqrtG = np.sqrt(np.clip(G_eff, 0.0, None))
dln_factor = (1.0 - sqrtG)  # integrando este termo * dN
# acumulamos desde o início (sigma8(N) = sigma8_0 * exp(-0.5 * integral dln_factor dN) )
cum = np.cumsum(dln_factor * dN)   # length = len(N)
sig8 = sigma8_0 * np.exp(-0.5 * cum)
# assegurar alinhamento: sig8[idx0] é valor hoje
fsigma8 = f_growth * sig8

# médias / supressão z<1
mask_z_lt1 = zc < 1.0
if mask_z_lt1.any():
    fs8_mean = fsigma8[mask_z_lt1].mean()
    lcdm_mean = 0.47 * np.mean((1.0 / (1.0 + zc[mask_z_lt1]))**0.9)
    suppression_pct = 100.0 * (lcdm_mean - fs8_mean) / lcdm_mean
else:
    fs8_mean = np.nan
    suppression_pct = np.nan

# ------------------------------
# Relatório simples
# ------------------------------
print("="*80)
print("DUT FINAL PATCH - revisão numericamente mais consistente")
print(f"H0 local (calibrado): {H0_loc:.3f} km/s/Mpc")
print(f"H(CMB proxy at z~1100) : {H[idx_cmb] if idx_cmb < len(H) else np.nan:.3f} km/s/Mpc")
print(f"w_eff(z=0): {w_eff0:.5f}")
print(f"fσ8(z=0): {fsigma8[idx0]:.4f}")
print(f"fσ8 suppression (z<1): {suppression_pct:.2f}%")
print(f"H(t→∞) approx: {H[-1]:.6f} km/s/Mpc")
print("="*80)

# ------------------------------
# Salvando figura (similar à sua)
# ------------------------------
plt.figure(figsize=(12,10))
# H(z)
plt.subplot(2,2,1)
mask = zc < 2.0
z_plot = zc[mask]
H_plot = H[mask]
lcdm = H0 * np.sqrt(0.3*(1+z_plot)**3 + 0.7)
plt.plot(z_plot, H_plot, 'b-', lw=2, label='DUT (revisado)')
plt.plot(z_plot, lcdm, 'r--', lw=2, label='ΛCDM')
plt.xlabel('z'); plt.ylabel('H(z) [km/s/Mpc]'); plt.legend(loc='upper left')

# w_eff
plt.subplot(2,2,2)
plt.plot(z_plot, w_eff[mask], 'g-', lw=2)
plt.axhline(-1, color='k', ls=':'); plt.ylim(-1.5, 1.0)
plt.xlabel('z'); plt.ylabel('w_eff(z)')

# fσ8
plt.subplot(2,2,3)
mask2 = zc < 1.2
z_fs8 = zc[mask2]
plt.plot(z_fs8, fsigma8[mask2], 'bo-', lw=2, label='DUT fsigma8')
plt.plot(z_fs8, 0.47 * (1/(1+z_fs8))**0.9, 'r--', label='ΛCDM approx')
plt.xlabel('z'); plt.ylabel('fσ8(z)'); plt.legend()

# H(N)
plt.subplot(2,2,4)
plt.plot(N, H, 'm-', lw=2)
plt.axvline(0, color='k', ls='--')
plt.xlabel('N = ln(a)'); plt.ylabel('H [km/s/Mpc]')
plt.title('Fossilization: H -> 0 (se o modelo permitir)')
plt.tight_layout()
plt.savefig('DUT_REVISED_results.png', dpi=300, bbox_inches='tight')
print("Figura salva: DUT_REVISED_results.png")
