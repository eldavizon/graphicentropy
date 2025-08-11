import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import quad

# ===============================
# Configurações iniciais & constantes
# ===============================
st.set_page_config(layout="wide")
k = 1.380649e-23  # Constante de Boltzmann (J/K)

# CSS responsivo mínimo
st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .element-container:has(.stPlotlyChart) { padding-left: 0px !important; padding-right: 0px !important; }
        .stSlider > div { padding: 0px 4px !important; }
    }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===============================
# Distribuição de Boltzmann (energia)
# ===============================
def boltzmann_distribution(E, T):
    """
    f(E) = (2 / sqrt(pi)) * (1 / (kT)^(3/2)) * sqrt(E) * exp(-E/(kT))
    Esta é a distribuição de energia (cinética) resultante da distribuição de Maxwell para velocidades,
    transformada para variáveis de energia E = 1/2 m v^2.
    """
    with np.errstate(all="ignore"):
        coeff = (2 / np.sqrt(np.pi)) * (1 / (k * T) ** (3 / 2))
        return coeff * np.sqrt(E) * np.exp(-E / (k * T))


def plot_boltzmann_distribution():
    st.subheader("Distribuição de Boltzmann")

    T_init = st.slider("Temperatura (K)", 100, 1000, 300, key="boltzmann_temp")
    E = np.linspace(1e-25, 5e-20, 1000)
    y = boltzmann_distribution(E, T_init)

    # energias características
    E_mp = 0.5 * k * T_init  # energia mais provável (kT/2)
    E_mean = 1.5 * k * T_init  # energia média (3/2 kT)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E, y=y, mode="lines", name="Distribuição de Boltzmann", line=dict(width=3)))
    fig.add_trace(
        go.Scatter(
            x=[E_mp],
            y=[boltzmann_distribution(E_mp, T_init)],
            mode="markers+text",
            name="Energia mais provável",
            marker=dict(color="red", size=9, symbol="x"),
            text=["$E_{mp}=\\frac{1}{2}kT$"],
            textposition="top right",
            hovertemplate="E_mp: %{x:.2e} J",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[E_mean],
            y=[boltzmann_distribution(E_mean, T_init)],
            mode="markers+text",
            name="Energia média",
            marker=dict(color="green", size=9, symbol="circle"),
            text=["$\\bar{E}=\\frac{3}{2}kT$"],
            textposition="bottom right",
            hovertemplate="E_mean: %{x:.2e} J",
        )
    )

    fig.update_layout(
        title=f"📊 Distribuição de Boltzmann a T = {T_init} K",
        xaxis=dict(title="Energia (J)", tickformat=".1e", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="Densidade de probabilidade (1/J)", showgrid=True, gridcolor="lightgray"),
        height=500,
        margin=dict(l=30, r=30, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Expander com dedução passo a passo
    with st.expander("Deduções matemáticas — Distribuição de Boltzmann (mostrar)"):
        st.markdown(
            r"""
**Resumo da origem:** partimos da distribuição de Maxwell para velocidades em 3D:
$$
f_v(v) = 4\pi \left(\frac{m}{2\pi k T}\right)^{3/2} v^2 e^{- \frac{m v^2}{2 k T}}.
$$
Usando $E=\tfrac{1}{2} m v^2$ e a mudança de variável $dv = \frac{dE}{m v} = \frac{dE}{\sqrt{2 m E}}$, obtém-se
a distribuição em energia $f_E(E)$ da forma
$$
f_E(E) \propto \sqrt{E}\, e^{-E/(kT)}.
$$

A normalização (constante) leva à forma que usamos:
$$
f_E(E) = \frac{2}{\sqrt{\pi}} \frac{1}{(kT)^{3/2}} \sqrt{E}\, e^{-E/(kT)}.
$$

**Energia mais provável**: derivando $f_E(E)$ e igualando a zero:
$$
\frac{d}{dE}\big(\sqrt{E} e^{-E/(kT)}\big) = 0
\quad\Rightarrow\quad
\frac{1}{2E} - \frac{1}{kT} = 0
\quad\Rightarrow\quad
E_{mp} = \frac{1}{2}kT.
$$

**Energia média** (momento de primeira ordem da distribuição normalizada):
$$
\langle E \rangle = \int_0^\infty E f_E(E)\, dE = \frac{3}{2} kT.
$$

Observações:
- O fator $(kT)^{-3/2}$ garante a correta dimensionalidade e normalização.
- A forma $\sqrt{E}$ aparece devido à densidade de estados (em 3D).
            """
        )


# ===============================
# Entropia e Temperatura
# ===============================
def plot_entropy_temperature():
    st.subheader("Entropia e Temperatura")

    N = 1.0
    E = np.linspace(1e-21, 5e-21, 200)
    # Modelo simples: número de microestados Ω(E) ∝ E^{3N/2} (gás ideal monoatômico)
    # => S = k ln Ω = k * (3N/2) * ln(E) + const
    S = k * (3 * N / 2) * np.log(E)
    dSdE = np.gradient(S, E)
    T = 1.0 / dSdE
    T_teorico = (2 * E) / (3 * N * k)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=E, y=S, mode="lines", name="Entropia S(E)"))
    fig1.update_layout(title="📈 Entropia em função da Energia", xaxis=dict(tickformat=".1e"), height=380)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=E, y=T, mode="lines", name="T = (∂S/∂E)^{-1}"))
    fig2.add_trace(
        go.Scatter(
            x=E, y=T_teorico, mode="lines", name="T = 2E/(3Nk)", line=dict(dash="dash")
        )
    )
    fig2.update_layout(title="📈 Temperatura vs Energia", xaxis=dict(tickformat=".1e"), height=380)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Deduções matemáticas — Entropia e Temperatura (mostrar)"):
        st.markdown(
            r"""
Partimos da hipótese (modelo do gás ideal monoatômico, contagem sem detalhar constantes):
$$
\Omega(E) \propto E^{\frac{3N}{2}}.
$$
Logo a entropia de Boltzmann:
$$
S(E) = k \ln \Omega(E) = k \frac{3N}{2} \ln E + \text{const}.
$$

Definição termodinâmica de temperatura:
$$
\frac{1}{T} = \left(\frac{\partial S}{\partial E}\right)_{V,N}.
$$
Derivando $S(E)$:
$$
\frac{\partial S}{\partial E} = k \frac{3N}{2} \frac{1}{E}
\quad\Rightarrow\quad
T(E) = \frac{1}{\partial S/\partial E} = \frac{2E}{3Nk}.
$$

Observações:
- A constante aditiva em S(E) desaparece ao derivar.
- Esse resultado é consistente com a energia média por partícula $\langle E\rangle = \tfrac{3}{2}kT$.
            """
        )


# ===============================
# Fração de partículas até E0
# ===============================
def plot_energy_fraction():
    st.subheader("Fração de Partículas em Intervalo de Energia")

    T = st.slider("Temperatura (K)", 100, 1000, 300, key="energy_frac_temp")

    # Geração de uma grade logarítmica de energia para cobrir muitas ordens de magnitude
    E_min = 1e-24
    E_max = 5 * k * T
    E_values = np.logspace(np.log10(E_min), np.log10(E_max), 800)
    f_E = boltzmann_distribution(E_values, T)

    # normalização numérica
    total = np.trapz(f_E, E_values)
    if total <= 0 or not np.isfinite(total):
        st.error("Erro na normalização (integral inválida). Ajuste T ou intervalo de energia.")
        return
    f_E_norm = f_E / total

    # fração acumulada (numérica)
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (f_E_norm[1:] + f_E_norm[:-1]) * np.diff(E_values))))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=E_values, y=f_E_norm, mode="lines", name="Distribuição Normalizada"))
    fig1.update_layout(title="🔍 Distribuição de Energia (normalizada)", xaxis=dict(type="log", tickformat=".1e"), height=380)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=E_values, y=cumulative[:-0], mode="lines", name="Fração acumulada"))
    # marcar percentis
    for frac in [0.25, 0.5, 0.75, 0.9]:
        idx = np.argmin(np.abs(cumulative - frac))
        fig2.add_trace(
            go.Scatter(
                x=[E_values[idx]],
                y=[cumulative[idx]],
                mode="markers+text",
                marker=dict(size=8),
                text=[f"{int(frac*100)}%"],
                textposition="top right",
                showlegend=False,
            )
        )
    fig2.update_layout(title="📈 Fração Acumulada de Partículas", xaxis=dict(type="log", tickformat=".1e"), height=380)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Deduções matemáticas — Fração acumulada (mostrar)"):
        st.markdown(
            r"""
A fração de partículas com energia menor ou igual a $E_0$ é:
$$
F(E_0) = \frac{\int_0^{E_0} f_E(E)\, dE}{\int_0^{\infty} f_E(E)\, dE}.
$$
Com a forma de $f_E(E)$ (distribuição de energia),
fazendo a mudança de variável $x=E/(kT)$, obtém-se uma expressão em termos da função gama incompleta:
$$
F(E_0) = \frac{\gamma\left(\tfrac{3}{2}, \, \frac{E_0}{kT}\right)}{\Gamma\left(\tfrac{3}{2}\right)},
$$
onde $\gamma$ é a função gama incompleta inferior e $\Gamma(3/2)=\tfrac{\sqrt{\pi}}{2}$.

No código usamos integração numérica (trapézio) sobre uma malha logarítmica para
capturar corretamente as contribuições de ordens de magnitude distintas.

Observações práticas:
- Para valores muito pequenos/grandes de $E$ pode haver underflow; por isso escolhemos uma grade log.
- A expressão fechada com funções gama é útil para cálculos teóricos/analíticos.
            """
        )


# ===============================
# Lei de Stefan-Boltzmann
# ===============================
def plot_stefan_boltzmann():
    st.subheader("Lei de Stefan-Boltzmann")

    sigma = 5.670374419e-8  # W / (m^2 K^4)

    def stefan_boltzmann_power(T, A=1.0, emissivity=1.0):
        return emissivity * sigma * A * T ** 4

    def energia_total_radiada(T1, T2, A=1.0, emissivity=1.0):
        # usamos integração numérica aqui, mas existe solução analítica simples
        integrand = lambda T: emissivity * sigma * A * T ** 4
        energia, _ = quad(integrand, T1, T2)
        return energia

    with st.expander("Parâmetros de entrada"):
        col1, col2 = st.columns(2)
        with col1:
            emiss = st.slider("Emissividade (ε)", 0.01, 1.0, 0.95, key="emiss")
            area = st.slider("Área (m²)", 0.1, 10.0, 1.0, key="area")
        with col2:
            T1 = st.slider("Temperatura Inicial (K)", 100, 1900, 200, key="T1")
            T2 = st.slider("Temperatura Final (K)", 200, 2000, 1500, key="T2")

    T1, T2 = min(T1, T2), max(T1, T2)
    T_range = np.linspace(100, 2000, 1000)
    power = stefan_boltzmann_power(T_range, area, emiss)
    energia = energia_total_radiada(T1, T2, area, emiss)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T_range, y=power, mode="lines", name="P = ε σ A T⁴", line=dict(color="red", width=2)))
    mask = (T_range >= T1) & (T_range <= T2)
    fig.add_trace(
        go.Scatter(
            x=T_range[mask],
            y=power[mask],
            fill="tozeroy",
            fillcolor="rgba(255,165,0,0.4)",
            name="Intervalo (T1→T2)",
            line=dict(width=0),
        )
    )

    fig.update_layout(
        title="🌞 Lei de Stefan-Boltzmann - Potência irradiada vs Temperatura",
        xaxis=dict(title="Temperatura (K)"),
        yaxis=dict(title="Potência (W)"),
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info(f"**Energia irradiada de {T1:.0f} K a {T2:.0f} K:** {energia:.4e} J")

    with st.expander("Deduções matemáticas — Stefan-Boltzmann (mostrar)"):
        st.markdown(
            r"""
A potência irradiada por uma superfície com emissividade ε e área A é dada por:
$$
P(T) = \varepsilon \sigma A T^4.
$$

A energia total irradiada quando a temperatura varia de $T_1$ a $T_2$ (interpretando $P$ como função de $T$ a ser integrada) é:
$$
E = \int_{T_1}^{T_2} P(T)\, dT = \varepsilon \sigma A \int_{T_1}^{T_2} T^4\, dT.
$$

A integração analítica é direta:
$$
\int T^4 dT = \frac{T^5}{5} \quad\Rightarrow\quad
E = \frac{\varepsilon \sigma A}{5}\left(T_2^5 - T_1^5\right).
$$

No código usamos integração numérica (`quad`) por generalidade, mas a expressão acima mostra a forma fechada.
            """
        )


# ===============================
# Interface principal
# ===============================
def main():
    st.title("📈 Aplicações do Cálculo I na Química — com deduções passo a passo")

    st.markdown("### 📚 Menu de Gráficos")
    st.markdown("Escolha uma visualização; abra o expander correspondente para ver as deduções matemáticas.")

    options = {
        "1. Distribuição de Boltzmann": plot_boltzmann_distribution,
        "2. Entropia e Temperatura": plot_entropy_temperature,
        "3. Fração de partículas": plot_energy_fraction,
        "4. Lei de Stefan-Boltzmann": plot_stefan_boltzmann,
    }

    choice = st.selectbox("Visualização:", list(options.keys()), index=0, label_visibility="collapsed")
    st.markdown("---")
    options[choice]()


if __name__ == "__main__":
    main()
