import datetime
import os

start = datetime.datetime.now()
os.system('clear')

################################################################################

print(80*'=')
print(" Markov Chain Simulation and Visualization ".center(80, '='))
print(80*'=')


import numpy as np
import matplotlib
matplotlib.use("Agg")  # garante modo não interativo (sem janelas)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =========================================================
# Utilitários
# =========================================================
def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_fig(fig, png_path, pdf_writer=None, dpi=300):
    """Salva a figura em PNG e, se houver PdfPages, também no PDF."""
    if png_path:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if pdf_writer is not None:
        pdf_writer.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Núcleo da cadeia de Markov
# =========================================================
def checar_matriz_transicao(P, tol=1e-9):
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P deve ser uma matriz quadrada.")
    if (P < -tol).any():
        raise ValueError("P possui probabilidades negativas.")
    if not np.allclose(P.sum(axis=1), 1.0, atol=tol):
        raise ValueError("Cada linha de P deve somar 1.")
    return P


def simular_cadeia_markov(P, estados, n_passos, estado_inicial=None, seed=None):
    """
    Simula uma cadeia de Markov discreta.

    P: matriz de transição (n x n) com linhas somando 1
    estados: lista/tupla com rótulos dos estados (len == n)
    n_passos: número de passos (gera trajetória com n_passos+1 incluindo o estado inicial)
    estado_inicial: rótulo do estado inicial (ou None -> aleatório)
    seed: inteiro para reprodutibilidade
    """
    rng = np.random.default_rng(seed)
    P = checar_matriz_transicao(P)
    estados = list(estados)
    n = len(estados)
    idx = {s: i for i, s in enumerate(estados)}

    if estado_inicial is None:
        s_atual = rng.integers(0, n)
    else:
        if estado_inicial not in idx:
            raise ValueError(f"Estado inicial '{estado_inicial}' não está em 'estados'.")
        s_atual = idx[estado_inicial]

    caminho_idx = [s_atual]
    for _ in range(n_passos):
        probs = P[s_atual]
        s_atual = rng.choice(n, p=probs)
        caminho_idx.append(s_atual)

    caminho = [estados[i] for i in caminho_idx]
    counts = np.bincount(caminho_idx, minlength=n)
    freqs = counts / counts.sum()

    return {
        "caminho": caminho,
        "caminho_idx": caminho_idx,
        "contagens": dict(zip(estados, counts)),
        "frequencias": dict(zip(estados, freqs))
    }


def distribuicao_estacionaria(P, tol=1e-12):
    """Resolve πP = π usando autovetor de P^T associado ao autovalor 1."""
    P = np.asarray(P, dtype=float)
    w, v = np.linalg.eig(P.T)
    k = np.argmin(np.abs(w - 1.0))
    pi = np.real(v[:, k])
    pi = np.maximum(pi, 0)
    s = pi.sum()
    if s < tol:
        raise RuntimeError("Falha ao encontrar vetor estacionário.")
    return pi / s


def matriz_transicao_empirica(caminho_idx, n_estados):
    counts = np.zeros((n_estados, n_estados), dtype=int)
    for a, b in zip(caminho_idx[:-1], caminho_idx[1:]):
        counts[a, b] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = counts.sum(axis=1, keepdims=True)
        P_emp = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums != 0)
    return counts, P_emp


# =========================================================
# Visualizações com cores claras
# =========================================================
def fig_heatmap_matriz(P, estados, titulo="Matriz", cmap="YlGnBu"):
    """Heatmap com paleta clara e contraste ajustado."""
    P = np.asarray(P, dtype=float)
    fig = plt.figure(figsize=(4.8, 4.2))
    im = plt.imshow(P, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(estados)), estados)
    plt.yticks(range(len(estados)), estados)
    plt.xlabel("Destino")
    plt.ylabel("Origem")
    plt.title(titulo, fontsize=11, weight="bold")

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            color = "black" if P[i, j] < 0.6 else "white"
            plt.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
    plt.tight_layout()
    return fig


def fig_trajetoria(estados, caminho_idx):
    t = np.arange(len(caminho_idx))
    fig = plt.figure(figsize=(9, 2.6))
    plt.step(t, caminho_idx, where="post", color="#2c7fb8")
    plt.yticks(range(len(estados)), estados)
    plt.xlabel("Passo")
    plt.ylabel("Estado")
    plt.title("Trajetória da cadeia de Markov")
    plt.tight_layout()
    return fig


def fig_frequencias(estados, freq_dict):
    """Mostra a frequência relativa de visitas a cada estado."""
    labels = list(estados)
    vals = [freq_dict[e] for e in labels]
    fig = plt.figure(figsize=(6, 3))
    plt.bar(labels, vals, color="#74a9cf", edgecolor="black")
    plt.ylim(0, 1)
    plt.ylabel("Frequência relativa")
    plt.title("Proporção de visitas a cada estado (frequência empírica)")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v*100:.1f}%", ha="center")
    plt.tight_layout()
    return fig


def fig_convergencia_vs_estacionaria(P, estados, caminho_idx):
    n = len(estados)
    T = len(caminho_idx)
    counts = np.zeros((T, n), dtype=int)
    for t in range(T):
        counts[t, caminho_idx[t]] += 1
    cumul = counts.cumsum(axis=0) / np.arange(1, T + 1)[:, None]
    pi = distribuicao_estacionaria(P)

    fig = plt.figure(figsize=(8, 4))
    for k, nome in enumerate(estados):
        plt.plot(cumul[:, k], label=f"{nome} (cumul.)", lw=1.8)
    for k, nome in enumerate(estados):
        plt.axhline(pi[k], linestyle="--", linewidth=1.2, color="gray")
    plt.ylim(0, 1)
    plt.xlabel("Passo")
    plt.ylabel("Frequência")
    plt.title("Convergência das frequências para a distribuição estacionária")
    plt.legend(ncol=2)
    plt.tight_layout()
    return fig


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    out_dir = "markov_outputs"
    ensure_dir(out_dir)

    # Matriz de transição e estados
    P = [
        [0.6, 0.2, 0.2],  # Feliz -> (Feliz, Neutro, Triste)
        [0.4, 0.4, 0.2],  # Neutro -> ...
        [0.2, 0.3, 0.5]   # Triste -> ...
    ]
    estados = ["Feliz", "Neutro", "Triste"]

    print("Matriz de transição P:")
    print(np.array(P))
    print("\nEstados:", estados)
    print("\nDistribuição estacionária π:")
    pi = distribuicao_estacionaria(P)
    for estado, prob in zip(estados, pi):
        print(f"  {estado}: {prob:.4f}")
    print("")

    # =========================================================
    resultado = simular_cadeia_markov(P, estados, n_passos=500, estado_inicial="Neutro", seed=42)
    # =========================================================


    print("Caminho simulado (início):", resultado["caminho"][:15], "...")
    print("Contagens:", resultado["contagens"])
    print("Frequências:", {k: float(v) for k, v in resultado["frequencias"].items()})

    counts, P_emp = matriz_transicao_empirica(resultado["caminho_idx"], n_estados=len(estados))
    print("\nTransições observadas (counts):\n", counts)
    print("\nMatriz empírica (linhas ~ 1):\n", np.round(P_emp, 3))

    pdf_path = os.path.join(out_dir, "markov_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # Heatmap teórico
        fig = fig_heatmap_matriz(P, estados, titulo="Matriz de transição P (teórica)", cmap="YlGnBu")
        save_fig(fig, os.path.join(out_dir, "01_P_teorica.png"), pdf)

        # Trajetória
        fig = fig_trajetoria(estados, resultado["caminho_idx"])
        save_fig(fig, os.path.join(out_dir, "02_trajetoria.png"), pdf)

        # Frequências empíricas (tempo de permanência)
        fig = fig_frequencias(estados, resultado["frequencias"])
        save_fig(fig, os.path.join(out_dir, "03_frequencias_empiricas.png"), pdf)

        # Convergência x estacionária
        fig = fig_convergencia_vs_estacionaria(P, estados, resultado["caminho_idx"])
        save_fig(fig, os.path.join(out_dir, "04_convergencia_vs_estacionaria.png"), pdf)

        # Heatmap empírico
        fig = fig_heatmap_matriz(P_emp, estados, titulo="Matriz de transição P_emp (empírica)", cmap="PuBuGn")
        save_fig(fig, os.path.join(out_dir, "05_P_empirica.png"), pdf)

    print(f"\nArquivos salvos em: {os.path.abspath(out_dir)}")
    print(f"- PDF consolidado: {pdf_path}")
    print("- PNGs individuais: 01_... 05_...")

################################################################################

print('')
end = datetime.datetime.now()
time = end - start
print(f'Tempo total de execução: {time}')

################################################################################