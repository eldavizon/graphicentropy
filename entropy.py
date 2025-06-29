import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import quad

# Configurações iniciais
st.set_page_config(layout="wide")  # Força layout wide
k = 1.380649e-23  # Constante de Boltzmann em J/K

## 1. Distribuição de Boltzmann - Versão Corrigida
def boltzmann_distribution(E, T):
    """Função da distribuição de Boltzmann"""
    with np.errstate(all='ignore'):
        coeff = (2 / np.sqrt(np.pi)) * (1 / (k * T) ** (3 / 2))
        return coeff * np.sqrt(E) * np.exp(-E / (k * T))

def plot_boltzmann_distribution():
    st.subheader("Distribuição de Boltzmann")
    
    T_init = st.slider("Temperatura (K)", 100, 1000, 300, key="boltzmann_temp")
    E = np.linspace(1e-25, 5e-20, 1000)
    y = boltzmann_distribution(E, T_init)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E, y=y, mode='lines', name='Distribuição', line=dict(width=2)))
    
    emp = 0.5 * k * T_init
    eme = 1.5 * k * T_init
    fig.add_vline(x=emp, line=dict(color='red', dash='dash'), annotation_text="Energia mais provável")
    fig.add_vline(x=eme, line=dict(color='green', dash='dash'), annotation_text="Energia média")
    
    fig.update_layout(
        title='Distribuição de Boltzmann',
        xaxis_title='Energia (J)',
        yaxis_title='Densidade de Probabilidade',
        height=400,  # Altura reduzida
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=12))
    
    st.plotly_chart(fig, use_container_width=True)

## 2. Entropia vs Energia
def entropy(Omega):
    return k * np.log(Omega)

def plot_entropy_temperature():
    st.subheader("Entropia e Temperatura")
    
    N = 1.0
    E = np.linspace(1e-21, 5e-21, 100)
    log_Omega = (3 * N / 2) * np.log(E)
    S = k * log_Omega
    dSdE = np.gradient(S, E)
    T = 1 / dSdE
    T_teorico = (2 * E) / (3 * N * k)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=E, y=S, mode='lines', name='Entropia', line=dict(color='blue')))
    fig1.update_layout(
        title='Entropia vs Energia',
        xaxis_title='Energia (J)',
        yaxis_title='Entropia (J/K)',
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=E, y=T, mode='lines', name='Temperatura', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=E, y=T_teorico, mode='lines', name='Relação teórica', line=dict(color='black', dash='dash')))
    fig2.update_layout(
        title='Temperatura vs Energia',
        xaxis_title='Energia (J)',
        yaxis_title='Temperatura (K)',
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

## 3. Fração de partículas - Versão Corrigida
def plot_energy_fraction():
    st.subheader("Fração de Partículas em Intervalo de Energia")
    
    T = st.slider("Temperatura (K)", 100, 1000, 300, key="energy_frac_temp")

    def boltzmann_distribution(E, T):
        with np.errstate(all='ignore'):
            log_coeff = np.log(2 / np.sqrt(np.pi)) + 1.5 * np.log(1 / (k * T))
            log_val = log_coeff + 0.5 * np.log(E) - E / (k * T)
            return np.exp(log_val)

    E_min = 1e-24
    E_max = 5 * k * T
    E_values = np.logspace(np.log10(E_min), np.log10(E_max), 500)
    f_E = boltzmann_distribution(E_values, T)
    total_integral = np.trapezoid(f_E, E_values)
    f_E_normalized = f_E / total_integral
    cumulative = np.insert(np.cumsum(f_E_normalized[:-1] * np.diff(E_values)), 0, 0)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=E_values, y=f_E_normalized, mode='lines', line=dict(color='blue', width=2)))
    fig1.update_layout(
        title='Distribuição de Energia',
        xaxis_title='Energia (J)',
        yaxis_title='Densidade de Probabilidade',
        xaxis_type='log',
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=E_values, y=cumulative, mode='lines', line=dict(color='red', width=2)))
    
    for frac in [0.25, 0.5, 0.75, 0.9]:
        idx = np.argmin(np.abs(cumulative - frac))
        fig2.add_trace(go.Scatter(x=[E_values[idx]], y=[cumulative[idx]], 
                     mode='markers+text', 
                     marker=dict(color='black', size=8),
                     text=[f'{frac*100:.0f}%'],
                     textposition='top right',
                     showlegend=False))
    
    fig2.update_layout(
        title='Fração Cumulativa',
        xaxis_title='Energia (J)',
        yaxis_title='Fração com E ≤ E₀',
        xaxis_type='log',
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

## 4. Lei de Stefan-Boltzmann - Radiação térmica
def plot_stefan_boltzmann():
    st.subheader("Lei de Stefan-Boltzmann")
    
    sigma = 5.670374419e-8

    def stefan_boltzmann_power(T, A=1.0, emissividade=1.0):
        return emissividade * sigma * A * T ** 4

    def energia_total_radiada(T1, T2, A=1.0, emissividade=1.0):
        integrand = lambda T: emissividade * sigma * A * T ** 4
        energia, _ = quad(integrand, T1, T2)
        return energia

    col1, col2 = st.columns(2)
    with col1:
        emiss = st.slider('Emissividade (ε)', 0.01, 1.0, 0.95, key='emiss')
        area = st.slider('Área (m²)', 0.1, 10.0, 1.0, key='area')
    with col2:
        T1 = st.slider('Temperatura Inicial (K)', 100, 1900, 200, key='T1')
        T2 = st.slider('Temperatura Final (K)', 200, 2000, 1500, key='T2')

    T1, T2 = min(T1, T2), max(T1, T2)
    T_range = np.linspace(100, 2000, 1000)
    power = stefan_boltzmann_power(T_range, area, emiss)
    energia = energia_total_radiada(T1, T2, area, emiss)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T_range, y=power, mode='lines', name="P = εσAT⁴", line=dict(color='red', width=2)))
    
    mask = (T_range >= T1) & (T_range <= T2)
    fig.add_trace(go.Scatter(x=T_range[mask], y=power[mask], fill='tozeroy', 
                          fillcolor='rgba(255,165,0,0.5)', name="Energia irradiada", line=dict(width=0)))
    
    fig.update_layout(
        title='Lei de Stefan-Boltzmann',
        xaxis_title='Temperatura (K)',
        yaxis_title='Potência irradiada (W)',
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"**Energia irradiada de {T1:.0f}K a {T2:.0f}K:** {energia:.2f} J")

## Menu principal
def main():
    st.title("Aplicações do Cálculo I na Química - Visualizações")
    
    options = {
        "1. Distribuição de Boltzmann": plot_boltzmann_distribution,
        "2. Entropia e Temperatura": plot_entropy_temperature,
        "3. Fração de partículas": plot_energy_fraction,
        "4. Lei de Stefan-Boltzmann": plot_stefan_boltzmann
    }
    
    # Menu mais compacto
    with st.sidebar:
        st.markdown("### Menu de Gráficos")
        choice = st.radio("Selecione:", list(options.keys()), label_visibility="collapsed")
    
    # Executa a função selecionada
    options[choice]()

if __name__ == "__main__":
    main()