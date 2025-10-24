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
print(" Gibbs Sampler — Bivariate Normal ".center(80, '='))
print(80 * '=')

# ------------------------------------------------------------
# Parâmetros
# ------------------------------------------------------------
N_STEPS      = 4000     # total de iterações da cadeia
STEP_ANIM    = 10       # passo entre frames (p/ reduzir tamanho do GIF)
RHO          = 0.9      # correlação entre X e Y
X0, Y0       = 0.0, 0.0 # estado inicial
SEED         = 123

rng = np.random.default_rng(SEED)

print(f"\nIterações totais (N_STEPS): {N_STEPS:,}")
print(f"Passo entre frames (STEP_ANIM): {STEP_ANIM}")
print(f"Correlação alvo (rho): {RHO}")
print(f"Estado inicial: (x0, y0)=({X0}, {Y0})\n")

# ------------------------------------------------------------
# Alvo: Normal bivariada N(0, Sigma) com var(X)=var(Y)=1 e corr=RHO
# Condicionais:
#   X | Y=y ~ N( mu_x|y = rho*y,  var_x|y = 1 - rho^2 )
#   Y | X=x ~ N( mu_y|x = rho*x,  var_y|x = 1 - rho^2 )
# ------------------------------------------------------------
VAR_COND = 1.0 - RHO**2
SD_COND  = np.sqrt(VAR_COND)

def gibbs_sampler(n_steps, x0=0.0, y0=0.0, rho=0.9):
    x = np.empty(n_steps)
    y = np.empty(n_steps)
    x[0], y[0] = x0, y0
    for t in range(1, n_steps):
        # X | Y=y_{t-1}
        mu_x = rho * y[t-1]
        x[t] = rng.normal(mu_x, SD_COND)
        # Y | X=x_t
        mu_y = rho * x[t]
        y[t] = rng.normal(mu_y, SD_COND)
    return x, y

x, y = gibbs_sampler(N_STEPS, x0=X0, y0=Y0, rho=RHO)

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

def rough_ess(a, max_lag=2000):
    n = len(a)
    rho_sum = 0.0
    for k in range(1, min(max_lag, n-1)):
        r = autocorr(a, k)
        if r <= 0:
            break
        rho_sum += 2.0 * r
    return int(n / (1.0 + rho_sum))

ESS_X = rough_ess(x, max_lag=1000)
ESS_Y = rough_ess(y, max_lag=1000)
print(f"ESS aproximado — X: {ESS_X} | Y: {ESS_Y}")

# ------------------------------------------------------------
# Preparando contornos da densidade conjunta (para o painel esquerdo)
# pdf conjunta (até constante) de N(0, Sigma) com var=1 e corr=rho:
#  f(x,y) ∝ exp( -1/(2*(1-rho^2)) * (x^2 - 2*rho*x*y + y^2) )
# Usaremos apenas para visual (contornos)
# ------------------------------------------------------------
def joint_pdf_unnorm(X, Y, rho=RHO):
    z = (X**2 - 2.0*rho*X*Y + Y**2) / (2.0*(1.0 - rho**2))
    return np.exp(-z)

grid_lim = 4.0
grid_points = 200
gx = np.linspace(-grid_lim, grid_lim, grid_points)
gy = np.linspace(-grid_lim, grid_lim, grid_points)
GX, GY = np.meshgrid(gx, gy)
GZ = joint_pdf_unnorm(GX, GY, rho=RHO)

# ------------------------------------------------------------
# Figura com dois painéis: (1) nuvem + contornos, (2) traces X e Y
# ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
plt.subplots_adjust(wspace=0.3, bottom=0.18)

# Painel esquerdo: contornos + amostras
cs = ax1.contour(GX, GY, GZ, levels=8)  # contornos da densidade conjunta
path_scatter = ax1.plot([], [], linestyle="None", marker="o", markersize=2, alpha=0.6, label="Amostras (X,Y)")[0]
curr_point  = ax1.plot([], [], marker="o", markersize=7, label="Estado atual")[0]
ax1.set_xlim(-grid_lim, grid_lim)
ax1.set_ylim(-grid_lim, grid_lim)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("Gibbs: exploração da Normal bivariada")
ax1.legend(loc="upper left", frameon=False)

# Painel direito: traces de X e Y
(line_x,) = ax2.plot([], [], linewidth=1.5, label="X (estado)")
(line_y,) = ax2.plot([], [], linewidth=1.0, label="Y (estado)")
ax2.set_xlim(0, N_STEPS)
ax2.set_ylim(-grid_lim, grid_lim)
ax2.set_xlabel("Iterações")
ax2.set_ylabel("Valor")
ax2.set_title("Trajetórias (traces) de X e Y")
ax2.legend(loc="upper right", frameon=False)

info_text = fig.text(
    0.5, 0.06, "", fontsize=11, family="monospace", va="top", ha="center"
)

def update(frame_idx):
    t = (frame_idx + 1) * STEP_ANIM
    t = min(t, N_STEPS - 1)

    path_scatter.set_data(x[:t+1], y[:t+1])
    curr_point.set_data([x[t]], [y[t]])

    line_x.set_data(np.arange(t+1), x[:t+1])
    line_y.set_data(np.arange(t+1), y[:t+1])

    info_text.set_text(
        f"iter: {t:5d} | (x_t, y_t)=({x[t]: .3f}, {y[t]: .3f}) | ESS_X≈{ESS_X} | ESS_Y≈{ESS_Y} | rho={RHO}"
    )
    return path_scatter, curr_point, line_x, line_y, info_text

frames = (N_STEPS + STEP_ANIM - 1) // STEP_ANIM
anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

# ------------------------------------------------------------
# Salvando: GIF, PNG final e TXT com métricas
# ------------------------------------------------------------
out_dir = Path.cwd() / "gibbs_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

gif_path = out_dir / "gibbs_bivariate_anim.gif"
png_path = out_dir / "gibbs_bivariate_final.png"
txt_path = out_dir / "gibbs_bivariate_stats.txt"

writer = PillowWriter(fps=15)

print("")
with tqdm(total=frames, desc="Gerando GIF", unit="frame", colour="green", ncols=80) as pbar:
    anim.save(gif_path, writer=writer, progress_callback=lambda i, n: pbar.update(1))

# frame final com título e métricas
update(frames - 1)
fig.suptitle(
    f"Gibbs — Normal bivariada | rho={RHO} | ESS_X≈{ESS_X} | ESS_Y≈{ESS_Y}",
    x=0.5, y=0.98, fontsize=12, weight="bold"
)
plt.savefig(png_path, dpi=220, bbox_inches="tight")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"Correlação (rho): {RHO}\n")
    f.write(f"Iterações totais: {N_STEPS}\n")
    f.write(f"Semente RNG: {SEED}\n")
    f.write(f"ESS aproximado X: {ESS_X}\n")
    f.write(f"ESS aproximado Y: {ESS_Y}\n")

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
