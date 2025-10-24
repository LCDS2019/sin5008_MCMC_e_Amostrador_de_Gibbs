# z_hmc_rosenbrock.py
# ------------------------------------------------------------
# Hamiltonian Monte Carlo (HMC) com animação e salvamento
# Alvo 2D: densidade ∝ exp( - U(x,y)/2 ), onde
#   U(x,y) = (1 - x)^2 + 100*(y - x**2)**2   (potencial de Rosenbrock)
# Gera:
#  - GIF animado (contornos + trajetória e estados aceitos)
#  - PNG final com métricas
#  - TXT com taxa de aceitação, energia média e ESS aproximado
# ------------------------------------------------------------

import datetime
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm.auto import tqdm

# ------------------------------------------------------------
# Início
# ------------------------------------------------------------
start = datetime.datetime.now()
os.system('cls' if os.name == 'nt' else 'clear')

print(80 * '=')
print(" Hamiltonian Monte Carlo (HMC) — Rosenbrock ".center(80, '='))
print(80 * '=')

# ------------------------------------------------------------
# Parâmetros
# ------------------------------------------------------------
N_STEPS       = 3000     # iterações de HMC
STEP_ANIM     = 5        # espaçamento entre frames no GIF
EPS           = 0.02     # tamanho de passo do leapfrog
L_STEPS       = 25       # nº de passos leapfrog por proposta
MASS          = 1.0      # massa (matriz identidade * MASS)
SEED          = 123
X0, Y0        = -1.5, 1.5  # estado inicial

rng = np.random.default_rng(SEED)

print(f"\nIterações (N_STEPS): {N_STEPS:,}")
print(f"Leapfrog: L={L_STEPS} passos, eps={EPS}")
print(f"Massa: {MASS} (identidade)")
print(f"Estado inicial: ({X0}, {Y0})\n")

# ------------------------------------------------------------
# Alvo: log densidade e gradiente (Rosenbrock)
#   U(x,y) = (1 - x)^2 + 100*(y - x^2)**2
#   log π(x,y) = -U/2  (constante ignorada)
#   ∇U = [ 2(x-1) - 400x(y - x^2),  200*(y - x**2) ]
#   ∇logπ = -0.5 * ∇U
# ------------------------------------------------------------
def U(x, y):
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

def grad_U(x, y):
    dUx = 2.0*(x - 1.0) - 400.0*x*(y - x**2)
    dUy = 200.0*(y - x**2)
    return dUx, dUy

def logpi(x, y):
    return -0.5 * U(x, y)

def grad_logpi(x, y):
    dUx, dUy = grad_U(x, y)
    return -0.5*dUx, -0.5*dUy

# ------------------------------------------------------------
# HMC — leapfrog + aceitação de Metropolis
# ------------------------------------------------------------
def leapfrog(q, p, eps, L, mass=1.0):
    """q=(x,y), p=(px,py); retorna (q_new, p_new)"""
    x, y = q
    px, py = p

    # meia etapa de momento
    gx, gy = grad_logpi(x, y)
    px = px + 0.5*eps*gx
    py = py + 0.5*eps*gy

    # L etapas completas
    for _ in range(L):
        # posição
        x = x + eps * (px / mass)
        y = y + eps * (py / mass)
        # força (exceto na última, mas fazemos e corrigimos depois)
        gx, gy = grad_logpi(x, y)
        if _ < L - 1:
            px = px + eps * gx
            py = py + eps * gy

    # meia etapa final de momento
    px = px + 0.5*eps*gx
    py = py + 0.5*eps*gy

    return (x, y), (px, py)

def hamiltonian(q, p, mass=1.0):
    x, y = q
    px, py = p
    K = 0.5*((px**2 + py**2)/mass)      # energia cinética
    V = -logpi(x, y)                    # V = -log π
    return K + V

def hmc_sampler(n_steps, eps, L, q0, mass=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    xs = np.empty(n_steps)
    ys = np.empty(n_steps)
    accepted = np.zeros(n_steps, dtype=bool)
    energy_err = np.empty(n_steps)

    xs[0], ys[0] = q0
    for t in range(1, n_steps):
        q = (xs[t-1], ys[t-1])
        # momento ~ N(0, M)
        p = rng.normal(0.0, np.sqrt(mass), size=2)
        H0 = hamiltonian(q, p, mass=mass)

        # leapfrog
        q_new, p_new = leapfrog(q, p, eps, L, mass=mass)
        H1 = hamiltonian(q_new, p_new, mass=mass)

        # aceitação
        log_alpha = - (H1 - H0)  # min(1, exp(log_alpha))
        if np.log(rng.random()) < log_alpha:
            xs[t], ys[t] = q_new
            accepted[t] = True
        else:
            xs[t], ys[t] = q
            accepted[t] = False

        energy_err[t] = H1 - H0

    acc_rate = accepted[1:].mean()
    return xs, ys, accepted, energy_err, acc_rate

xs, ys, accepted, dH, acc_rate = hmc_sampler(
    N_STEPS, EPS, L_STEPS, q0=(X0, Y0), mass=MASS, rng=rng
)

# ------------------------------------------------------------
# ESS aproximado (por autocorrelação positiva)
# ------------------------------------------------------------
def autocorr(a, lag):
    a = np.asarray(a)
    a = a - a.mean()
    denom = np.dot(a, a)
    if denom == 0 or lag >= len(a):
        return 0.0
    return np.dot(a[:-lag], a[lag:]) / denom

def rough_ess(a, max_lag=1000):
    n = len(a)
    rho_sum = 0.0
    for k in range(1, min(max_lag, n-1)):
        r = autocorr(a, k)
        if r <= 0:
            break
        rho_sum += 2.0*r
    return int(n / (1.0 + rho_sum))

ESS_X = rough_ess(xs, max_lag=800)
ESS_Y = rough_ess(ys, max_lag=800)

print(f"Taxa de aceitação (aprox.): {acc_rate:.3f}")
print(f"ΔH médio (|ΔH|): {np.mean(np.abs(dH[1:])):.4e}")
print(f"ESS aproximado — X: {ESS_X} | Y: {ESS_Y}")

# ------------------------------------------------------------
# Preparando contornos da densidade (para o painel esquerdo)
# ------------------------------------------------------------
def joint_unnorm_pdf(X, Y):
    # proporcional a exp(-U/2)
    return np.exp(-0.5 * ((1.0 - X)**2 + 100.0*(Y - X**2)**2))

grid_lim = 3.5
grid_points = 250
gx = np.linspace(-grid_lim, grid_lim, grid_points)
gy = np.linspace(-grid_lim, grid_lim, grid_points)
GX, GY = np.meshgrid(gx, gy)
GZ = joint_unnorm_pdf(GX, GY)

# ------------------------------------------------------------
# Figura com dois painéis: (1) contornos + estados, (2) traces
# ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
plt.subplots_adjust(wspace=0.3, bottom=0.18)

# Painel esquerdo
cs = ax1.contour(GX, GY, GZ, levels=12)
path_scatter = ax1.plot([], [], linestyle="None", marker="o", markersize=2, alpha=0.6, label="Estados aceitos")[0]
curr_point  = ax1.plot([], [], marker="o", markersize=7, label="Estado atual")[0]
ax1.set_xlim(-grid_lim, grid_lim)
ax1.set_ylim(-grid_lim, grid_lim)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("HMC: exploração (Rosenbrock)")
ax1.legend(loc="upper right", frameon=False)

# Painel direito
(line_x,) = ax2.plot([], [], linewidth=1.5, label="X")
(line_y,) = ax2.plot([], [], linewidth=1.0, label="Y")
ax2.set_xlim(0, N_STEPS)
ax2.set_ylim(-grid_lim, grid_lim)
ax2.set_xlabel("Iterações")
ax2.set_ylabel("Valor")
ax2.set_title("Traces")
ax2.legend(loc="upper right", frameon=False)

info_text = fig.text(
    0.5, 0.06, "", fontsize=11, family="monospace", va="top", ha="center"
)

def update(frame_idx):
    t = (frame_idx + 1) * STEP_ANIM
    t = min(t, N_STEPS - 1)

    # só pontos aceitos (visual mais limpo)
    acc_mask = accepted[:t+1]
    path_scatter.set_data(xs[:t+1][acc_mask], ys[:t+1][acc_mask])
    curr_point.set_data([xs[t]], [ys[t]])

    line_x.set_data(np.arange(t+1), xs[:t+1])
    line_y.set_data(np.arange(t+1), ys[:t+1])

    info_text.set_text(
        f"iter: {t:5d} | (x_t, y_t)=({xs[t]: .3f}, {ys[t]: .3f}) | acc≈{acc_rate:.3f} | "
        f"ESS_X≈{ESS_X} | ESS_Y≈{ESS_Y} | |ΔH| médio≈{np.mean(np.abs(dH[1:t+1])):.2e}"
    )
    return path_scatter, curr_point, line_x, line_y, info_text

frames = (N_STEPS + STEP_ANIM - 1) // STEP_ANIM
anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

# ------------------------------------------------------------
# Salvando: GIF, PNG e TXT
# ------------------------------------------------------------
out_dir = Path.cwd() / "hmc_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

gif_path = out_dir / "hmc_rosenbrock_anim.gif"
png_path = out_dir / "hmc_rosenbrock_final.png"
txt_path = out_dir / "hmc_rosenbrock_stats.txt"

writer = PillowWriter(fps=15)

print("")
with tqdm(total=frames, desc="Gerando GIF", unit="frame", colour="green", ncols=80) as pbar:
    anim.save(gif_path, writer=writer, progress_callback=lambda i, n: pbar.update(1))

# frame final com título e métricas
update(frames - 1)
fig.suptitle(
    f"HMC — Rosenbrock | acc≈{acc_rate:.3f} | ESS_X≈{ESS_X} | ESS_Y≈{ESS_Y} | eps={EPS}, L={L_STEPS}",
    x=0.5, y=0.98, fontsize=12, weight="bold"
)
plt.savefig(png_path, dpi=220, bbox_inches="tight")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"Iterações: {N_STEPS}\n")
    f.write(f"Leapfrog: eps={EPS}, L={L_STEPS}\n")
    f.write(f"Massa: {MASS}\n")
    f.write(f"Taxa de aceitação (aprox.): {acc_rate:.6f}\n")
    f.write(f"Delta H médio (|ΔH|): {np.mean(np.abs(dH[1:])):.6e}\n")
    f.write(f"ESS aproximado X: {ESS_X}\n")
    f.write(f"ESS aproximado Y: {ESS_Y}\n")
    f.write(f"Semente: {SEED}\n")

print(f"\nArquivos salvos em: {out_dir}")
print(f"GIF : {gif_path}")
print(f"PNG : {png_path}")
print(f"TXT : {txt_path}")

# ------------------------------------------------------------
# Tempo total
# ------------------------------------------------------------
end = datetime.datetime.now()
elapsed = end - start
print(f"\nTempo total de execução: {elapsed}")
