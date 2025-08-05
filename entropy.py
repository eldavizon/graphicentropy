import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time


# Configurações iniciais
st.set_page_config(layout="wide", page_title="Cálculo I na Química")
k = 1.380649e-23  # Constante de Boltzmann
sigma = 5.670374419e-8  # Constante de Stefan-Boltzmann

# CSS responsivo
st.markdown("""
<style>
@media (max-width: 768px) {
    .element-container:has(.stPlotlyChart) {
        padding: 0px !important;
    }
    .stSlider > div {
        padding: 0px 4px !important;
    }
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =======================================
# Função 1 - Distribuição de Boltzmann
# =======================================
def boltzmann_distribution(E, T):
    with np.errstate(all='ignore'):
        coeff = (2 / np.sqrt(np.pi)) * (1 / (k * T) ** (3 / 2))
        return coeff * np.sqrt(E) * np.exp(-E / (k * T))

def plot_boltzmann_distribution():
    st.subheader("Distribuição de Boltzmann")

    T = st.slider("Temperatura (K)", 100, 1000, 300, key="boltzmann_temp")
    E = np.linspace(1e-25, 5e-20, 1000)
    y = boltzmann_distribution(E, T)

    # Ponto mais provável e ponto médio
    emp = 0.5 * k * T  # Energia mais provável (modo da distribuição)
    eme = 1.5 * k * T  # Energia média

    # Intervalo para integração
    st.markdown("##### Intervalo de energia para cálculo de probabilidade")
    col1, col2 = st.columns(2)
    with col1:
        E1 = st.number_input("Energia mínima (J)", value=1e-24, format="%.1e")
    with col2:
        E2 = st.number_input("Energia máxima (J)", value=1e-21, format="%.1e")

    E1, E2 = min(E1, E2), max(E1, E2)
    prob, _ = quad(lambda E_: boltzmann_distribution(E_, T), E1, E2)

    fig = go.Figure()

    # Curva da distribuição
    fig.add_trace(go.Scatter(
        x=E, y=y,
        mode='lines',
        name='Distribuição de Boltzmann',
        line=dict(width=3, color='royalblue')
    ))

    # Energia mais comum (modo) = 0.5kT
    fig.add_trace(go.Scatter(
        x=[emp], y=[boltzmann_distribution(emp, T)],
        mode='markers+text',
        name='Energia mais provável (Emp = 0.5kT)',
        marker=dict(size=10, color='red'),
        text=["Emp (0.5kT)"],
        textposition="top right"
    ))

    # Energia média = 1.5kT
    fig.add_trace(go.Scatter(
        x=[eme], y=[boltzmann_distribution(eme, T)],
        mode='markers+text',
        name='Energia média (Ē = 1.5kT)',
        marker=dict(size=10, color='green'),
        text=["Ē (1.5kT)"],
        textposition="bottom right"
    ))

    # Área preenchida (probabilidade entre E1 e E2)
    E_mask = (E >= E1) & (E <= E2)
    fig.add_trace(go.Scatter(
        x=E[E_mask], y=y[E_mask],
        fill='tozeroy',
        name=f"Área entre {E1:.1e} J e {E2:.1e} J",
        fillcolor='rgba(255,165,0,0.5)',
        line=dict(width=0)
    ))

    fig.update_layout(
        title=f"📊 Distribuição de Boltzmann a T = {T} K",
        xaxis_title="Energia (J)",
        yaxis_title="Densidade de probabilidade (1/J)",
        height=500,
        font=dict(size=15),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=30, r=30, t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Explicação do significado da área sob a curva
    st.markdown(f"""
    ✅ **Probabilidade** de uma partícula ter energia entre **{E1:.1e} J** e **{E2:.1e} J**:

    $$
    P(E_1 \\leq E \\leq E_2) = \\int_{{E_1}}^{{E_2}} f(E) \\, dE \\approx {prob:.4f}
    $$

    Onde a função densidade de probabilidade \\( f(E) \\) é dada por:

    $$
    f(E) = \\frac{{2}}{{\\sqrt{{\\pi}}}} \\cdot \\frac{{1}}{{(kT)^{{3/2}}}} \\cdot \\sqrt{{E}} \\cdot e^{{-E/(kT)}}
    $$

    Além disso:

    - Energia mais provável (modo da distribuição):

    $$
    E_{{\\text{{mp}}}} = \\frac{{1}}{{2}}kT
    $$

    - Energia média:

    $$
    \\bar{{E}} = \\frac{{3}}{{2}}kT
    $$
    """)


def boltzmann_energy_distribution(n, kT):
    """
    Gera n energias aleatórias com base na distribuição de Boltzmann via amostragem inversa.
    """
    E_max = 10 * kT
    E = np.linspace(0, E_max, 1000)
    f_E = (2 / np.sqrt(np.pi)) * (1 / (kT ** 1.5)) * np.sqrt(E) * np.exp(-E / kT)
    f_E /= np.trapz(f_E, E)

    cdf = np.cumsum(f_E)
    cdf /= cdf[-1]
    rand_vals = np.random.rand(n)
    sampled_energies = np.interp(rand_vals, cdf, E)
    return sampled_energies


def plot_boltzmann_animation():
    st.title("Distribuição de Boltzmann - Simulação de Partículas (Ar Atmosférico)")

    # ---- Parâmetros físicos reais ----
    k_B = 1.380649e-23  # Constante de Boltzmann (J/K)
    molar_mass = 0.02897  # kg/mol (massa molar do ar)
    N_A = 6.02214076e23  # mol^-1
    m = molar_mass / N_A  # massa de uma molécula (~4.8e-26 kg)

    # ---- Interface ----
    n_particles = st.slider("Número de partículas", 50, 500, 300, step=10)
    temperature = st.slider("Temperatura (K)", 100, 1000, 300, step=10)
    real_speed_mode = st.checkbox("🔁 Ativar velocidades reais (sem escala visual)")

    # Dimensões físicas reais da sala (5m x 5m em 2D)
    box_size = 5.0  # metros
    dt = 0.01  # intervalo de tempo em segundos por frame

    st.markdown(r"""
    A velocidade de cada partícula é derivada da energia cinética conforme:

    $$
    v = \sqrt{\frac{2E}{m}}
    $$
    """)

    st.markdown(
        """
        Esta simulação usa a **distribuição de Boltzmann real** para calcular a velocidade de cada molécula de ar.  
        O botão acima ativa/desativa o modo com **velocidade real sem escala visual** (pode parecer rápido demais).
        """
    )

    # ---- Inicialização ----
    positions = np.random.rand(n_particles, 2) * box_size
    angles = np.random.rand(n_particles) * 2 * np.pi

    # Gera energias e calcula velocidades reais
    energies = boltzmann_energy_distribution(n_particles, kT=k_B * temperature)
    speeds = np.sqrt(2 * energies / m)  # m/s

    # Aplica escala visual apenas se modo real não estiver ativado
    if real_speed_mode:
        scaled_speeds = speeds
        st.warning("⚠️ Velocidades reais ativadas: partículas podem atravessar a caixa muito rápido.")
    else:
        # Aplica fator de escala visual (ex: 0.03 * box_size / máx. velocidade)
        scale_factor = 0.03 * box_size / speeds.max()
        scaled_speeds = speeds * scale_factor

    velocities = np.stack((np.cos(angles), np.sin(angles)), axis=1) * scaled_speeds[:, np.newaxis]

    # Layout centralizado
    left, center, right = st.columns([1, 2, 1])
    with center:
        canvas = st.empty()

    for frame in range(200):
        positions += velocities * dt

        # Reflexão nas bordas
        for i in range(n_particles):
            for j in range(2):
                if positions[i, j] < 0 or positions[i, j] > box_size:
                    velocities[i, j] *= -1
                    positions[i, j] = np.clip(positions[i, j], 0, box_size)

        # Coloração por velocidade
        actual_speeds = np.linalg.norm(velocities, axis=1)
        norm_speeds = (actual_speeds - actual_speeds.min()) / np.ptp(actual_speeds)
        colors = plt.cm.plasma(norm_speeds)

        # Gráfico
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('black')
        ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=10, alpha=0.85)
        ax.set_title(f"T = {temperature} K", color="white", fontsize=10)

        with center:
            canvas.pyplot(fig)
        plt.close(fig)
        time.sleep(0.01)
# =======================================
# Função 4 - Lei de Stefan-Boltzmann
# =======================================
# Constante de Stefan-Boltzmann
sigma = 5.67e-8  # W/m²·K⁴

def plot_stefan_boltzmann():
    st.subheader("Lei de Stefan-Boltzmann")

    with st.expander("🔧 Parâmetros"):
        col1, col2 = st.columns(2)
        with col1:
            emiss = st.slider('Emissividade (ε)', 0.01, 1.0, 0.95)
            area = st.slider('Área (m²)', 0.1, 10.0, 1.0)
        with col2:
            T1 = st.slider('Temperatura Inicial (K)', 100, 1900, 200)
            T2 = st.slider('Temperatura Final (K)', 200, 2000, 1500)

    T1, T2 = min(T1, T2), max(T1, T2)

    def P(T): return emiss * sigma * area * T**4
    energia, _ = quad(P, T1, T2)

    T_range = np.linspace(100, 2000, 1000)
    power = P(T_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T_range, y=power, mode='lines', name='Potência irradiada (W)'))
    mask = (T_range >= T1) & (T_range <= T2)
    fig.add_trace(go.Scatter(
        x=T_range[mask], y=power[mask],
        fill='tozeroy', name='Energia total',
        fillcolor='rgba(255,165,0,0.5)', line=dict(width=0)
    ))

    fig.update_layout(
        title='🌞 Potência irradiada vs Temperatura',
        xaxis_title='Temperatura (K)',
        yaxis_title='Potência (W)',
        height=450
    )

    # Layout com duas colunas
    col_grafico, col_texto = st.columns([3, 2])

    with col_grafico:
        st.plotly_chart(fig, use_container_width=True)

    with col_texto:
        st.markdown("### 🔍 Interpretação física")
        st.markdown(r"""
        A **Lei de Stefan-Boltzmann** afirma que a potência irradiada por um corpo negro é proporcional à quarta potência da sua temperatura absoluta:

        $$
        P(T) = \varepsilon \cdot \sigma \cdot A \cdot T^4
        $$

        **Onde:**

        $$
        \varepsilon \quad \text{: emissividade do material (entre 0 e 1)}
        $$

        $$
        \sigma \approx 5.67 \times 10^{-8} \ \text{W/m}^2 \cdot \text{K}^4 \quad \text{: constante de Stefan-Boltzmann}
        $$

        $$
        A \quad \text{: área da superfície emissora (m}^2\text{)}
        $$

        $$
        T \quad \text{: temperatura absoluta (K)}
        $$
        """)

        st.markdown("### 📐 Energia irradiada")
        st.markdown(fr"""
        A curva mostra a potência irradiada em função da temperatura.  
        A **área sob a curva** entre **{T1}K** e **{T2}K** representa a **energia total irradiada**:

        $$
        E = \int_{{T_1}}^{{T_2}} P(T) \, dT = \int_{{T_1}}^{{T_2}} \varepsilon \cdot \sigma \cdot A \cdot T^4 \, dT
        $$
        """)

        st.info(f"🔋 Energia irradiada entre **{T1}K** e **{T2}K**: **{energia:.2f} J**")

        # Comparações energéticas
        st.markdown("### 🔁 Equivalente energético:")

        # Conversões
        tempo_lampada_100W = energia / 100  # segundos
        motores_carro = energia / 150000  # energia ~150kJ por min
        energia_kWh = energia / 3.6e6
        horas_residencia = energia_kWh / 0.5  # consumo médio 0.5 kWh/hora

        st.markdown(f"- 💡 Manter uma **lâmpada de 100W** acesa por **{tempo_lampada_100W:.1f} segundos**")
        st.markdown(f"- 🚗 Equivale à energia liberada por **{motores_carro:.2f} motores de carro** funcionando por 1 minuto")
        st.markdown(f"- 🏘️ Supriria o consumo de uma residência média por **{horas_residencia:.2f} horas**")

        with st.expander("🔎 Como essas estimativas foram feitas"):
            st.markdown("""
            - **Lâmpada de 100W**: 100W = 100J/s  
            - **Motor de carro**: Consumo estimado de ~150 kJ por minuto de funcionamento contínuo  
            - **Residência média**: 0.5 kWh/hora (considerando uma média de 500W de potência média contínua)
            - **1 kWh = 3.6 × 10⁶ J**
            """)

# =======================================
# Função Principal
# =======================================
def main():
    st.title("📈 Aplicações do Cálculo I na Química")
    st.markdown("Explore abaixo algumas visualizações interativas envolvendo cálculo, energia e probabilidade:")

    options = {
        "1. Distribuição de Boltzmann": plot_boltzmann_distribution,
        "2. Distribuição de Boltzmann (animação)": plot_boltzmann_animation,
        "3. Lei de Stefan-Boltzmann": plot_stefan_boltzmann
    }

    choice = st.selectbox("Selecione uma visualização:", list(options.keys()), index=0)
    st.markdown("---")
    options[choice]()  # Executa a função selecionada

if __name__ == "__main__":
    main()
