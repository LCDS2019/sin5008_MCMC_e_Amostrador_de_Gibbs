import datetime
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm.auto import tqdm

# -------------------------------------------------
# Início
# -------------------------------------------------
start = datetime.datetime.now()
os.system('cls' if os.name == 'nt' else 'clear')

print(80 * '=')
print(" Metropolis Random-Walk (MCMC) ".center(80, '='))
print(80 * '=')

# -------------------------------------------------
# Parâmetros
# -------------------------------------------------
N_STEPS = 1200         # total de iterações
STEP_ANIM = 3          # passo entre frames (p/ reduzir tamanho do GIF)
PROPOSAL_SIGMA = 1.0   # desvio da proposta RW: x' ~ N(x, sigma^2)
X0 = 0.0               # estado inicial
SEED = 42

rng = np.random.default_rng(SEED)

print(f"\nIterações totais (N_STEPS): {N_STEPS:,}")
print(f"Passo entre frames (STEP_ANIM): {STEP_ANIM}")
print(f"Sigma da proposta: {PROPOSAL_SIGMA}")
print(f"Estado inicial: {X0}\n")

# -------------------------------------------------
# Densidade alvo (mistura bimodal)
# -------------------------------------------------
def normal_pdf(x, mu, sigma):
    return (1.0/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)

def target_pdf(x):
    return 0.3*normal_pdf(x, -2.0, 0.5) + 0.7*normal_pdf(x,  2.0, 1.0)

def target_logpdf(x):
    # log-sum-exp da mistura para estabilidade numérica
    a = np.log(0.3) - 0.5*np.log(2*np.pi*0.5**2) - 0.5*((x+2.0)/0.5)**2
    b = np.log(0.7) - 0.5*np.log(2*np.pi*1.0**2) - 0.5*((x-2.0)/1.0)**2
    m = np.maximum(a, b)
    return m + np.log(np.exp(a-m) + np.exp(b-m))

# -------------------------------------------------
# Metropolis (Random-Walk) - proposta simétrica
# -------------------------------------------------
def metropolis_rw(logpdf, x0, n_steps=1000, proposal_sigma=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = np.empty(n_steps)
    prop = np.empty(n_steps)
    accepted = np.zeros(n_steps, dtype=bool)

    x[0] = x0
    prop[0] = x0
    for t in range(1, n_steps):
        proposal = rng.normal(x[t-1], proposal_sigma)
        log_alpha = logpdf(proposal) - logpdf(x[t-1])  # q cancela (simétrica)
        if np.log(rng.random()) < log_alpha:
            x[t] = proposal
            accepted[t] = True
        else:
            x[t] = x[t-1]
            accepted[t] = False
        prop[t] = proposal

    acc_rate = accepted[1:].mean()  # ignora t=0 no cálculo
    return x, prop, accepted, acc_rate

# roda a cadeia
x, prop, accepted, acc_rate = metropolis_rw(
    target_logpdf, X0, n_steps=N_STEPS, proposal_sigma=PROPOSAL_SIGMA, rng=rng
)

# -------------------------------------------------
# ESS aproximado (simples, via autocorrelação positiva)
# -------------------------------------------------
def autocorr(x, lag):
    x = np.asarray(x)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0 or lag >= len(x):
        return 0.0
    return np.dot(x[:-lag], x[lag:]) / denom

def rough_ess(x, max_lag=1000):
    n = len(x)
    rho_sum = 0.0
    for k in range(1, min(max_lag, n-1)):
        r = autocorr(x, k)
        if r <= 0:
            break
        rho_sum += 2.0 * r
    return n / (1.0 + rho_sum)

ESS_est = int(rough_ess(x, max_lag=300))

print(f"Taxa de aceitação (aprox.): {acc_rate:.3f}")
print(f"ESS (tamanho efetivo de amostra, aprox.): {ESS_est}")

# -------------------------------------------------
# Figura: dois painéis (densidade com estados + trace)
# -------------------------------------------------
xs = np.linspace(-5, 6, 600)
ys = target_pdf(xs)
ymax = float(ys.max() * 1.15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
plt.subplots_adjust(wspace=0.3, bottom=0.18)

# Painel 1: densidade + pontos visitados + proposta atual
ax1.plot(xs, ys, linewidth=2, label="Densidade alvo")
scatter_path = ax1.plot([], [], linestyle="None", marker="o", markersize=3, alpha=0.5, label="Estados visitados")[0]
point_curr = ax1.plot([], [], marker="o", markersize=8, label="Estado atual")[0]
point_prop = ax1.plot([], [], marker="x", markersize=8, label="Proposta")[0]
ax1.set_xlim(xs.min(), xs.max())
ax1.set_ylim(0, ymax)
ax1.set_xlabel("x")
ax1.set_ylabel("Densidade")
ax1.set_title("Metropolis RW — exploração da densidade")
ax1.legend(loc="upper left", frameon=False)

# Painel 2: trace (trajetória de x_t)
(line_trace,) = ax2.plot([], [], linewidth=1.5, label="x (estado)")
ax2.set_xlim(0, N_STEPS)
ax2.set_ylim(xs.min(), xs.max())
ax2.set_xlabel("Iterações")
ax2.set_ylabel("x")
ax2.set_title("Trajetória (trace) das amostras")
ax2.legend(loc="upper left", frameon=False)

info_text = fig.text(
    0.5, 0.06, "", fontsize=11, family="monospace", va="top", ha="center"
)

def update(frame_idx):
    t = (frame_idx + 1) * STEP_ANIM
    t = min(t, N_STEPS - 1)

    visited_x = x[:t+1]
    visited_y = target_pdf(visited_x)
    scatter_path.set_data(visited_x, visited_y)
    point_curr.set_data([x[t]], [target_pdf(x[t])])
    point_prop.set_data([prop[t]], [target_pdf(prop[t])])

    line_trace.set_data(np.arange(t+1), x[:t+1])

    status = "ACEITO " if accepted[t] else "REJEITADO"
    info_text.set_text(
        f"iter: {t:4d} | estado: {x[t]: .3f} | proposta: {prop[t]: .3f} | {status}"
        f" | acc≈{acc_rate:.3f} | ESS≈{ESS_est}"
    )
    return scatter_path, point_curr, point_prop, line_trace, info_text

frames = (N_STEPS + STEP_ANIM - 1) // STEP_ANIM
anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

# -------------------------------------------------
# Salvando: GIF, PNG final e TXT com métricas
# -------------------------------------------------
out_dir = Path.cwd() / "mcmc_metropolis_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

gif_path = out_dir / "metropolis_rw_anim.gif"
png_path = out_dir / "metropolis_rw_final.png"
txt_path = out_dir / "metropolis_rw_stats.txt"

writer = PillowWriter(fps=15)

print("")
with tqdm(total=frames, desc="Gerando GIF", unit="frame", colour="green", ncols=80) as pbar:
    anim.save(gif_path, writer=writer, progress_callback=lambda i, n: pbar.update(1))

# frame final com título e métricas
update(frames - 1)
fig.suptitle(
    f"Metropolis RW — acc≈{acc_rate:.3f} | ESS≈{ESS_est} | σ_proposta={PROPOSAL_SIGMA}",
    x=0.5, y=0.98, fontsize=12, weight="bold"
)
plt.savefig(png_path, dpi=220, bbox_inches="tight")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"Taxa de aceitação (aprox.): {acc_rate:.6f}\n")
    f.write(f"ESS aproximado: {ESS_est}\n")
    f.write(f"Sigma da proposta: {PROPOSAL_SIGMA}\n")
    f.write(f"Iterações totais: {N_STEPS}\n")
    f.write(f"Semente RNG: {SEED}\n")

print(f"\nArquivos salvos em: {out_dir}")
print(f"GIF : {gif_path}")
print(f"PNG : {png_path}")
print(f"TXT : {txt_path}")

# -------------------------------------------------
# Tempo total
# -------------------------------------------------
end = datetime.datetime.now()
elapsed = end - start
print(f"\nTempo total de execução: {elapsed}")
