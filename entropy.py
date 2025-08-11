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
# Funções para Distribuição
# ===============================

def boltzmann_velocity_distribution(v, T, m):
    """
    Distribuição de velocidades de Maxwell-Boltzmann para partículas de massa m (kg) a temperatura T (K).
    f(v) = 4π (m/(2πkT))^{3/2} v^2 exp(-mv^2/(2kT))
    """
    coeff = 4 * np.pi * (m / (2 * np.pi * k * T)) ** 1.5
    return coeff * v ** 2 * np.exp(-m * v ** 2 / (2 * k * T))


def boltzmann_energy_distribution(E, T, m):
    """
    Distribuição de energia cinética para partículas de massa m a temperatura T.
    E = 1/2 m v^2
    f(E) = (2/√π) * (1/(kT)^{3/2}) * sqrt(E) * exp(-E/(kT))
    """
    coeff = (2 / np.sqrt(np.pi)) * (1 / (k * T) ** 1.5)
    return coeff * np.sqrt(E) * np.exp(-E / (k * T))

# ===============================
# Plotagem da Distribuição de Velocidade
# ===============================
def plot_velocity_distribution(mass_u, T):
    m = mass_u * 1.66054e-27  # converte u para kg

    v_max = 3000  # m/s
    v = np.linspace(0, v_max, 1000)
    f_v = boltzmann_velocity_distribution(v, T, m)

    v_mp = np.sqrt(2 * k * T / m)  # velocidade mais provável
    v_mean = np.sqrt(8 * k * T / (np.pi * m))  # velocidade média
    v_rms = np.sqrt(3 * k * T / m)  # velocidade RMS

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v, y=f_v, mode="lines", name="Distribuição de velocidades", line=dict(color="blue", width=3)))
    fig.add_trace(go.Scatter(
        x=[v_mp], y=[boltzmann_velocity_distribution(v_mp, T, m)],
        mode="markers+text",
        name="Velocidade mais provável",
        marker=dict(color="red", size=10, symbol="x"),
        text=[f"v_mp = {v_mp:.1f} m/s"],
        textposition="top right"
    ))
    fig.add_trace(go.Scatter(
        x=[v_mean], y=[boltzmann_velocity_distribution(v_mean, T, m)],
        mode="markers+text",
        name="Velocidade média",
        marker=dict(color="green", size=10, symbol="circle"),
        text=[f"v_mean = {v_mean:.1f} m/s"],
        textposition="top left"
    ))
    fig.add_trace(go.Scatter(
        x=[v_rms], y=[boltzmann_velocity_distribution(v_rms, T, m)],
        mode="markers+text",
        name="Velocidade RMS",
        marker=dict(color="orange", size=10, symbol="diamond"),
        text=[f"v_rms = {v_rms:.1f} m/s"],
        textposition="bottom right"
    ))

    fig.update_layout(
        title=f"📊 Distribuição de Velocidades a T={T} K para massa {mass_u} u ({m:.2e} kg)",
        xaxis_title="Velocidade (m/s)",
        yaxis_title="Densidade de Probabilidade",
        height=500,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", font=dict(size=12))
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    Massa da partícula: **{mass_u:.2f} u** (≈ {m:.2e} kg)  
    Temperatura: **{T} K**

    Velocidade mais provável: **{v_mp:.2f} m/s**  
    Velocidade média: **{v_mean:.2f} m/s**  
    Velocidade RMS: **{v_rms:.2f} m/s**
    """)

    with st.expander("Deduções matemáticas — Distribuição de Velocidade (mostrar)"):
        st.markdown(r"""
A distribuição de velocidades para partículas em um gás ideal a temperatura T é dada pela distribuição de Maxwell-Boltzmann:
$$
f(v) = 4\pi \left(\frac{m}{2\pi k T}\right)^{3/2} v^2 e^{-\frac{m v^2}{2 k T}}.
$$

Aqui, $m$ é a massa da partícula, $k$ é a constante de Boltzmann, e $T$ a temperatura absoluta.

Velocidades características:
- Velocidade mais provável: $v_{mp} = \sqrt{\frac{2 k T}{m}}$  
- Velocidade média: $v_{mean} = \sqrt{\frac{8 k T}{\pi m}}$  
- Velocidade RMS: $v_{rms} = \sqrt{\frac{3 k T}{m}}$

Essas velocidades são obtidas calculando máximos ou momentos da distribuição.
        """)


# ===============================
# Plotagem da Distribuição de Energia Cinética
# ===============================
def plot_energy_distribution(mass_u, T):
    m = mass_u * 1.66054e-27  # converte u para kg

    E_max = 5 * k * T
    E = np.linspace(0, E_max, 1000)
    f_E = boltzmann_energy_distribution(E, T, m)

    E_mp = 0.5 * k * T  # energia mais provável
    E_mean = 1.5 * k * T  # energia média

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E, y=f_E, mode="lines", name="Distribuição de Energia", line=dict(color="blue", width=3)))
    fig.add_trace(go.Scatter(
        x=[E_mp], y=[boltzmann_energy_distribution(E_mp, T, m)],
        mode="markers+text",
        name="Energia mais provável",
        marker=dict(color="red", size=10, symbol="x"),
        text=[f"E_mp = 1/2 kT = {E_mp:.2e} J"],
        textposition="top right"
    ))
    fig.add_trace(go.Scatter(
        x=[E_mean], y=[boltzmann_energy_distribution(E_mean, T, m)],
        mode="markers+text",
        name="Energia média",
        marker=dict(color="green", size=10, symbol="circle"),
        text=[f"E_mean = 3/2 kT = {E_mean:.2e} J"],
        textposition="bottom right"
    ))

    fig.update_layout(
        title=f"📊 Distribuição de Energia Cinética a T={T} K",
        xaxis_title="Energia (J)",
        yaxis_title="Densidade de Probabilidade",
        height=500,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", font=dict(size=12)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    Massa da partícula: **{mass_u:.2f} u** (≈ {m:.2e} kg)  
    Temperatura: **{T} K**

    Energia mais provável: **{E_mp:.2e} J**  
    Energia média: **{E_mean:.2e} J**
    """)

    with st.expander("Deduções matemáticas — Distribuição de Energia Cinética (mostrar)"):
        st.markdown(r"""
A distribuição de energia cinética para partículas em um gás ideal a temperatura T é dada por:
$$
f(E) = \frac{2}{\sqrt{\pi}} \frac{1}{(kT)^{3/2}} \sqrt{E} e^{-E/(kT)}.
$$

Aqui, $E = \frac{1}{2} m v^2$ é a energia cinética, $k$ a constante de Boltzmann, e $T$ a temperatura absoluta.

Energia mais provável (máximo de $f(E)$):
$$
E_{mp} = \frac{1}{2} k T.
$$

Energia média (valor médio esperado):
$$
\langle E \rangle = \frac{3}{2} k T.
$$

Essa distribuição é derivada a partir da distribuição de velocidades de Maxwell-Boltzmann usando a mudança de variável entre $v$ e $E$.
        """)

# ===============================
# Interface principal
# ===============================
def main():
    st.title("📈 Aplicações do Cálculo I na Química — com deduções passo a passo")

    st.markdown("### 📚 Menu de Gráficos")
    st.markdown("Escolha o tipo de gráfico para Distribuição de Boltzmann; abra o expander correspondente para ver as deduções matemáticas.")

    dist_type = st.selectbox("Tipo de gráfico:", ["Velocidade", "Energia cinética"])

    mass_u = st.number_input("Massa da partícula (unidade atômica, u)", min_value=1e-3, max_value=300.0, value=28.0, step=1.0)
    T = st.slider("Temperatura (K)", 100, 1500, 300)

    st.markdown("---")

    if dist_type == "Velocidade":
        plot_velocity_distribution(mass_u, T)
    else:
        plot_energy_distribution(mass_u, T)


if __name__ == "__main__":
    main()
