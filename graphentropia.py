import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

k = 1.380649e-23  # Constante de Boltzmann (J/K)
sigma = 5.670374419e-8  # Constante de Stefan-Boltzmann (W/m²·K⁴)


# 1. Distribuição de Boltzmann
def boltzmann_distribution(E, T):
    coeff = (2 / np.sqrt(np.pi)) * (1 / (k * T) ** (3 / 2))
    return coeff * np.sqrt(E) * np.exp(-E / (k * T))


def show_boltzmann_distribution():
    st.subheader("Distribuição de Boltzmann")
    T = st.slider("Temperatura (K)", 100, 1000, 300)

    E = np.linspace(1e-25, 5e-20, 1000)
    dist = boltzmann_distribution(E, T)

    fig, ax = plt.subplots()
    ax.plot(E, dist, label=f"T = {T} K", lw=2)
    ax.axvline(0.5 * k * T, color='r', linestyle='--', label='Energia mais provável')
    ax.axvline(1.5 * k * T, color='g', linestyle='--', label='Energia média')
    ax.set_xlabel("Energia (J)")
    ax.set_ylabel("Densidade de Probabilidade")
    ax.set_title("Distribuição de Boltzmann")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


# 2. Entropia vs Energia
def show_entropy_plot():
    st.subheader("Entropia vs Energia e Temperatura")
    N = st.slider("Número de partículas (N)", 1, 100, 1)
    E = np.linspace(1e-21, 5e-21, 100)

    log_Omega = (3 * N / 2) * np.log(E)
    S = k * log_Omega
    dSdE = np.gradient(S, E)
    T = 1 / dSdE
    T_teorico = (2 * E) / (3 * N * k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(E, S, 'b-')
    ax1.set_title("Entropia vs Energia")
    ax1.set_xlabel("Energia (J)")
    ax1.set_ylabel("Entropia (J/K)")
    ax1.grid(True)

    ax2.plot(E, T, 'r-', label='Temperatura estimada')
    ax2.plot(E, T_teorico, 'k--', label='Temperatura teórica')
    ax2.set_title("Temperatura vs Energia")
    ax2.set_xlabel("Energia (J)")
    ax2.set_ylabel("Temperatura (K)")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig)


# 3. Fração de partículas
def show_energy_fraction():
    st.subheader("Fração de partículas com energia até E₀")
    T = st.slider("Temperatura (K)", 100, 1000, 300)

    def boltzmann_dist(E, T):
        log_coeff = np.log(2 / np.sqrt(np.pi)) + 1.5 * np.log(1 / (k * T))
        log_val = log_coeff + 0.5 * np.log(E) - E / (k * T)
        return np.exp(log_val)

    E_min = 1e-24
    E_max = 5 * k * T
    E_values = np.logspace(np.log10(E_min), np.log10(E_max), 500)
    f_E = boltzmann_dist(E_values, T)
    total = np.trapezoid(f_E, E_values)

    if total <= 0:
        st.error("Integral inválida. Ajuste os parâmetros.")
        return

    f_E_norm = f_E / total
    cumulative = np.cumsum(f_E_norm[:-1] * np.diff(E_values))
    cumulative = np.insert(cumulative, 0, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.semilogx(E_values, f_E_norm, 'b-', lw=2)
    ax1.set_title("Distribuição de Energia")
    ax1.set_xlabel("Energia (J)")
    ax1.set_ylabel("Densidade de Probabilidade")
    ax1.grid(True, which="both")

    ax2.semilogx(E_values, cumulative, 'r-', lw=2)
    ax2.set_title("Fração Cumulativa de Partículas")
    ax2.set_xlabel("Energia (J)")
    ax2.set_ylabel("Fração com E ≤ E₀")
    ax2.grid(True, which="both")

    for frac in [0.25, 0.5, 0.75, 0.9]:
        idx = np.argmin(np.abs(cumulative - frac))
        ax2.plot(E_values[idx], cumulative[idx], 'ko')
        ax2.annotate(f'{frac*100:.0f}%',
                     xy=(E_values[idx], cumulative[idx]),
                     xytext=(10, 10), textcoords='offset points')

    st.pyplot(fig)


# 4. Lei de Stefan-Boltzmann
def stefan_boltzmann_power(T, A=1.0, emissividade=1.0):
    return emissividade * sigma * A * T ** 4


def energia_total_radiada(T1, T2, A=1.0, emissividade=1.0):
    integrand = lambda T: emissividade * sigma * A * T ** 4
    energia, _ = quad(integrand, T1, T2)
    return energia


def show_stefan_boltzmann():
    st.subheader("Lei de Stefan-Boltzmann")

    emiss = st.slider("Emissividade (ε)", 0.01, 1.0, 0.95)
    area = st.slider("Área (m²)", 0.1, 10.0, 1.0)
    T1 = st.slider("Temperatura Inicial (K)", 100, 1900, 300)
    T2 = st.slider("Temperatura Final (K)", T1+10, 2000, 1000)

    T_range = np.linspace(100, 2000, 1000)
    power = stefan_boltzmann_power(T_range, area, emiss)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(T_range, power, 'r-', lw=2, label="P = εσAT⁴")
    ax.fill_between(T_range, power, where=(T_range >= T1) & (T_range <= T2),
                    color='orange', alpha=0.5, label="Energia irradiada")

    ax.set_xlabel("Temperatura (K)")
    ax.set_ylabel("Potência irradiada (W)")
    ax.set_title("Potência vs Temperatura")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    energia = energia_total_radiada(T1, T2, area, emiss)
    st.success(f"Energia irradiada de {T1}K a {T2}K: {energia:.2f} J")


# Interface principal
st.title("Aplicações do Cálculo na Química e Física Térmica")

option = st.sidebar.radio("Escolha uma visualização:", (
    "1. Distribuição de Boltzmann",
    "2. Entropia e Temperatura",
    "3. Fração de partículas por energia",
    "4. Lei de Stefan-Boltzmann"
))

if "Boltzmann" in option:
    show_boltzmann_distribution()
elif "Entropia" in option:
    show_entropy_plot()
elif "Fração" in option:
    show_energy_fraction()
elif "Stefan-Boltzmann" in option:
    show_stefan_boltzmann()
