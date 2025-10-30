import datetime
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Início
# ---------------------------------------------------------------------------

start = datetime.datetime.now()
os.system('cls' if os.name == 'nt' else 'clear')

print(80 * '=')
print(" Pi value calculation ".center(80, '='))
print(80 * '=')

# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------

N = 12000
step = 50
np.random.seed(42)

print(f"\nNúmero total de pontos (N): {N:,}")
print(f"Passo da animação (step): {step}\n")

# ---------------------------------------------------------------------------
# Geração dos pontos
# ---------------------------------------------------------------------------

x = np.random.uniform(0, 1, size=N)
y = np.random.uniform(0, 1, size=N)
inside = x**2 + y**2 <= 1.0

cum_inside = np.cumsum(inside.astype(int))
indices = np.arange(1, N + 1)
pi_running = 4 * cum_inside / indices

# ---------------------------------------------------------------------------
# Figura com dois gráficos (círculo + convergência)
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(wspace=0.35, bottom=0.18)  

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Estimativa de π por Monte Carlo')

theta = np.linspace(0, np.pi/2, 400)
ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)

scat_in = ax1.scatter([], [], c='blue', s=8, label='Dentro do círculo')
scat_out = ax1.scatter([], [], c='red', s=8, label='Fora do círculo')

ax1.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False
)

ax2.set_xlim(0, N)
ax2.set_ylim(3.0, 3.3)
ax2.set_xlabel('Iterações')
ax2.set_ylabel('Estimativa de π')
ax2.set_title('Convergência da estimativa')
ax2.axhline(np.pi, color='gray', linestyle='--', label='π real')
(line_est,) = ax2.plot([], [], color='blue', linewidth=2, label='Estimativa')
ax2.legend(loc='upper right', frameon=False)

info_text = fig.text(
    0.5, 0.05,
    '',
    fontsize=11,
    family='monospace',
    va='top',
    ha='center'
)

# ---------------------------------------------------------------------------
# Função de atualização da animação
# ---------------------------------------------------------------------------

def update(frame):
    k = min(N, (frame + 1) * step)
    xi, yi = x[:k], y[:k]
    mask = inside[:k]
    scat_in.set_offsets(np.c_[xi[mask], yi[mask]])
    scat_out.set_offsets(np.c_[xi[~mask], yi[~mask]])
    est = pi_running[k - 1]

    line_est.set_data(indices[:k], pi_running[:k])

    info_text.set_text(f'Iterações: {k:,}    π ≈ {est:.6f}')
    return scat_in, scat_out, line_est, info_text

frames = (N + step - 1) // step
anim = FuncAnimation(fig, update, frames=frames, interval=60, blit=True)

# ---------------------------------------------------------------------------
# Salvando os arquivos + π final, SE e IC
# ---------------------------------------------------------------------------

out_dir = Path.cwd() / "monte_carlo_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

gif_path = out_dir / "monte_carlo_pi_convergencia.gif"
png_path = out_dir / "monte_carlo_pi_convergencia_final.png"
txt_path = out_dir / "pi_final.txt"

writer = PillowWriter(fps=15)

print('')
with tqdm(total=frames, desc="Gerando GIF", unit="frame", colour="green", ncols=80) as pbar:
    anim.save(gif_path, writer=writer,
              progress_callback=lambda i, n: pbar.update(1))

update(frames - 1)
pi_final = float(pi_running[-1])
p_hat = pi_final / 4.0
se_pi = 4.0 * np.sqrt(p_hat * (1.0 - p_hat) / N)  # SE(π̂) ≈ 4 * sqrt(p(1-p)/N)
z = 1.96
ci_low, ci_high = pi_final - z * se_pi, pi_final + z * se_pi

fig.suptitle(f"π final ≈ {pi_final:.8f}", x=0.5, y=0.99, fontsize=12, weight='bold')
fig.text(0.5, 0.015,
         f"SE ≈ {se_pi:.6f}   |   95% CI: [{ci_low:.6f}, {ci_high:.6f}]",
         ha='center', va='bottom', fontsize=10, family='monospace')
plt.savefig(png_path, dpi=200, bbox_inches='tight')

print(f"\nπ final estimado (N={N}): {pi_final:.10f}")
print(f"Erro padrão (SE): {se_pi:.10f}")
print(f"IC 95%: [{ci_low:.10f}, {ci_high:.10f}]")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"π final estimado (N={N}): {pi_final:.10f}\n")
    f.write(f"Erro padrão (SE): {se_pi:.10f}\n")
    f.write(f"IC 95%: [{ci_low:.10f}, {ci_high:.10f}]\n")

print(f"\nArquivos salvos em: {out_dir}")
print(f"GIF : {gif_path}")
print(f"PNG : {png_path}")
print(f"TXT : {txt_path}")

# ---------------------------------------------------------------------------
# Tempo total
# ---------------------------------------------------------------------------

end = datetime.datetime.now()
elapsed = end - start
print(f"\nTempo total de execução: {elapsed}")
