import numpy as np
import streamlit as st
import plotly.graph_objects as go
from math import pi
from scipy.integrate import quad

# ===============================
# Configurações iniciais & constantes
# ===============================
k = 1.380649e-23  # Constante de Boltzmann (J/K)

def boltzmann_energy_dist(E, T):
    """
    Distribuição de energia (densidade por unidade de energia) para 3 graus de liberdade translacionais:
    f(E) = (2 / sqrt(pi)) * (1 / (k T)^(3/2)) * sqrt(E) * exp(-E/(k T))
    (unidades: 1/J)
    """
    with np.errstate(all="ignore"):
        coeff = (2.0 / np.sqrt(pi)) * (1.0 / (k * T) ** 1.5)
        return coeff * np.sqrt(E) * np.exp(-E / (k * T))


def maxwell_boltzmann_speed_dist(v, T, m):
    """
    Distribuição de velocidades (densidade por v) em 3D:
    f(v) = 4π (m / (2π k T))^(3/2) v^2 exp(- m v^2 / (2 k T))
    (unidades: 1/(m/s))
    """
    with np.errstate(all="ignore"):
        pref = 4.0 * pi * (m / (2.0 * pi * k * T)) ** 1.5
        return pref * v ** 2 * np.exp(-m * v ** 2 / (2.0 * k * T))


def stefan_boltzmann_power(T, A=1.0, emissivity=1.0):
    sigma = 5.670374419e-8
    return emissivity * sigma * A * T ** 4


def energia_total_radiada_numeric(T1, T2, A=1.0, emissivity=1.0):
    integrand = lambda T: emissivity * 5.670374419e-8 * A * T ** 4
    energia, _ = quad(integrand, T1, T2)
    return energia


def main():
    st.set_page_config(layout="wide", page_title="Derivação Interativa — Boltzmann")

    # CSS responsivo mínimo
    st.markdown(
        """
        <style>
        @media (max-width: 768px) {
            .element-container:has(.stPlotlyChart) { padding-left: 0px !important; padding-right: 0px !important; }
            .stSlider > div { padding: 0px 4px !important; }
        }
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        .stExpander > .stMarkdown { font-size: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------
    # Sidebar: parâmetros e opções
    # ---------------------------
    with st.sidebar:
        st.title("Parâmetros")
        T_global = st.slider("Temperatura (K) - controle global", 50, 1500, 300, step=10)
        st.markdown("**Massa molecular por partícula** (em u = amu)")
        mass_u = st.number_input("Massa (u)", value=28.0, step=1.0, format="%.4f")
        u_to_kg = 1.66053906660e-27
        m = mass_u * u_to_kg
        st.markdown(f"Massa por partícula (kg): **{m:.3e}**")
        st.markdown("---")
        st.title("Opções de Visualização")
        show_v2 = st.checkbox("Mostrar efeito do fator geométrico v² (comparar)", value=True)
        show_lagrange = st.checkbox("Mostrar esboço do método de Lagrange", value=True)
        show_normal_const = st.checkbox("Mostrar constantes analíticas de normalização", value=True)
        st.markdown("---")
        st.markdown("Dica: os sliders nas páginas individuais também mudam os gráficos localmente.")

    # ---------------------------
    # Walkthrough interativo (passos)
    # ---------------------------
    st.title("🔬 Ludwig Boltzmann — Dedução interativa (começando de PV = nRT)")
    st.markdown(
        "Escolha um passo abaixo para ver o raciocínio histórico-matemático, as ferramentas usadas e a demonstração numérica."
    )

    steps = [
        "1. PV = nRT → Energia por molécula",
        "2. Teoria cinética: P ↔ energia cinética média",
        "3. Pergunta: como se distribui a energia?",
        "4. Contagem de microestados → Entropia S = k ln Ω",
        "5. Método dos multiplicadores de Lagrange (esboço)",
        "6. Resultados: fator de Boltzmann e interpretação",
        "7. Jacobiano / densidade de estados (v²)",
        "8. Normalização (integrais gaussianas e função gama)",
        "9. Experimento numérico: comparar candidatos",
    ]

    step = st.selectbox("Selecione o passo:", steps, index=0)

    if step == steps[0]:
        st.header("1 — PV = nRT → Energia por molécula")
        st.write("Partimos da lei dos gases ideais:")
        st.latex(r"PV = nRT.")
        st.write(r"Com \(n = N/N_A\) e \(R = k N_A\) obtemos:")
        st.latex(r"PV = N k T.")
        st.write(r"Dividindo por \(N\):")
        st.latex(r"\frac{P V}{N} = k T.")
        st.write("Interpretação: \(kT\) tem dimensão de energia — liga temperatura à energia média por partícula.")
        st.info("Ferramenta: álgebra básica e interpretação dimensional — conecta grandezas macroscópicas com energia microscópica.")

    elif step == steps[1]:
        st.header("2 — Teoria cinética: pressão e energia cinética média")
        st.latex(r"P = \frac{1}{3} \frac{N m \langle v^2 \rangle}{V}.")
        st.write("Comparando com \(P = \frac{N k T}{V}\) resulta:")
        st.latex(r"\frac{1}{2} m \langle v^2 \rangle = \frac{3}{2} k T.")
        st.write("Ou seja, a energia cinética média por partícula é \(\\tfrac{3}{2}kT\).")
        st.info("Ferramenta: esperança matemática (média) e manipulação algébrica para ligar temperatura a energia cinética.")

    elif step == steps[2]:
        st.header("3 — A pergunta central")
        st.write("Se conhecemos a média \(\\langle E\\rangle = \\tfrac{3}{2} kT\), como as energias individuais se distribuem?")
        st.write("Precisamos de uma **função de densidade de probabilidade** \(f(v)\) ou \(f(E)\) que descreva essa dispersão.")
        st.info("Ferramenta: formulação probabilística — passamos do determinístico para o estatístico.")

    elif step == steps[3]:
        st.header("4 — Contagem de microestados e entropia")
        st.latex(r"S = k \ln \Omega.")
        st.write("Procuramos a distribuição \(f\) que maximize \(\Omega\) (ou \(S\)) sob restrições (normalização e energia média fixa).")
        st.info("Ferramenta: combinatória e logaritmos — simplificam a maximização.")

    elif step == steps[4]:
        st.header("5 — Método dos multiplicadores de Lagrange (esboço)")
        st.write("Queremos maximizar \(S[f]\) sujeito a:")
        st.latex(r"\int f(\mathbf{v})\, d^3v = 1")
        st.latex(r"\int \tfrac{1}{2} m v^2 f(\mathbf{v})\, d^3v = \tfrac{3}{2}kT")
        st.write("Definimos o funcional:")
        st.latex(r"\mathcal{L}[f] = \ln\Omega[f] - \alpha\left(\int f - 1\right) - \beta\left(\int \tfrac{1}{2}mv^2 f - \tfrac{3}{2}kT\right)")
        st.write(r"A condição \(\delta \mathcal{L}/\delta f = 0\) leva a:")
        st.latex(r"f(\mathbf{v}) \propto e^{-\beta \tfrac{1}{2} m v^2}.")
        st.write(r"Identificando \(\beta = 1/(kT)\) obtemos o exponencial de Boltzmann.")
        if show_lagrange:
            st.write("**Por que usar Lagrange?**")
            st.write("- É a ferramenta sistemática para otimização com restrições; transforma um problema restrito em um sem restrições.")
            st.write("- Em linguagem física, estamos escolhendo *a distribuição mais provável* (maior número de microestados) que satisfaça as condições macroscopicamente observadas.")
        st.info("Ferramenta: cálculo variacional e multiplicadores de Lagrange — essencial para derivar densidades de probabilidade condicionadas.")

    elif step == steps[5]:
        st.header("6 — Do exponencial em v ao fator de Boltzmann em E")
        st.write(r"A solução funcional exponencial depende da energia cinética: \(E=\tfrac{1}{2}m v^2\).")
        st.write("Escrevendo em termos de energia:")
        st.latex(r"f(E) \propto e^{-E/(kT)}.")
        st.write("Esse é o **fator de Boltzmann**: estados com energia maior têm probabilidade exponencialmente menor.")
        st.info("Ferramenta: identificação de constantes (β→1/kT) e mudança de variável energia↔velocidade.")

    elif step == steps[6]:
        st.header("7 — O Jacobiano e o termo \(v^2\) (densidade de estados)")
        st.write("Quando passamos de \(f(\mathbf{v})\) (densidade no espaço vetorial 3D) para \(f(v)\) (densidade do módulo da velocidade),")
        st.write("temos de multiplicar pelo elemento de volume em coordenadas esféricas:")
        st.latex(r"d^3v = 4\pi v^2 dv.")
        st.write("Portanto:")
        st.latex(r"f(v)\,dv \propto 4\pi v^2 e^{-m v^2/(2kT)} dv.")
        st.write("O termo \(4\pi v^2\) é o **Jacobiano** (mudança de variável) — representa a densidade de estados geométrica.")
        if show_v2:
            st.info("Ferramenta: cálculo multivariado — mudança de variáveis e interpretação geométrica (mais microestados a velocidades maiores).")

    elif step == steps[7]:
        st.header("8 — Normalização (integrais gaussianas e função gama)")
        st.write("Para que \(f(v)\) seja probabilidade válida, impomos:")
        st.latex(r"\int_0^\infty f(v) dv = 1.")
        st.write("Isso define a constante:")
        st.latex(r"A = 4\pi \left(\frac{m}{2\pi k T}\right)^{3/2}.")
        st.write("A normalização envolve integrais do tipo \(\int_0^\infty v^2 e^{-a v^2} dv\) que se resolvem com transformações que levam à função gama.")
        if show_normal_const:
            st.write("Relação útil (para referência):")
            st.latex(r"\int_0^\infty x^{n} e^{-a x^2} dx = \frac{1}{2} a^{-(n+1)/2} \Gamma\!\left(\frac{n+1}{2}\right).")
            st.write("Usando isso com \(n=2\) recuperamos a forma do prefator analítico.")
        st.info("Ferramenta: técnicas de integração (gaussiana) e conhecimento de funções especiais (Γ).")

    elif step == steps[8]:
        st.header("9 — Experimento numérico: comparar candidatos")
        st.write("Construímos candidatos simples e verificamos se cumprem normalização e a restrição de energia média.")
        st.write("- Candidato A: \(e^{-E/(kT)}\)")
        st.write("- Candidato B: \(\sqrt{E}\, e^{-E/(kT)}\)  (simula densidade de estados)")
        st.write("A normalização e a energia média mostram por que a forma final combina ambos os fatores.")
        st.info("Ferramenta: integração numérica para testar hipóteses e validar escolhas analíticas.")

    # ---------------------------
    # Menu principal com funções de plot
    # ---------------------------
    st.markdown("---")
    st.header("Menu de visualizações")
    options = {
        "Distribuição de Boltzmann (energia) — gráfico + dedução": "boltzmann",
        "Entropia e Temperatura — S(E) e T(E)": "entropy",
        "Fração de partículas até E₀": "cdf",
        "Potência radiada (Stefan-Boltzmann)": "stefan_boltzmann",
    }

    choice = st.selectbox("Escolha a visualização:", list(options.keys()))

    if options[choice] == "boltzmann":
        st.subheader("Distribuição de energia (Boltzmann)")
        T = st.slider("Temperatura (K)", 50, 1500, T_global, step=10)
        E_vals = np.linspace(0, 10 * k * T, 500)
        fE = boltzmann_energy_dist(E_vals, T)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=E_vals / (k * T), y=fE, mode="lines", name=r"$f(E)$")
        )
        fig.update_layout(
            title="Distribuição de energia de Boltzmann",
            xaxis_title=r"$E / (kT)$",
            yaxis_title=r"$f(E)$ (densidade por energia)",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif options[choice] == "entropy":
        st.subheader("Entropia e Temperatura (qualitativo)")
        st.latex(r"S = k \ln \Omega")
        st.latex(r"\frac{1}{T} = \frac{\partial S}{\partial E}")
        st.write("Esta visualização é conceitual e não mostra cálculos numéricos precisos.")
        st.info("É uma demonstração visual da relação entre entropia, energia e temperatura.")

    elif options[choice] == "cdf":
        st.subheader("Fração acumulada de partículas até energia E₀")
        T = st.slider("Temperatura (K)", 50, 1500, T_global, step=10, key="cdf_T")
        E0 = st.slider("Energia limite E₀ (em kT)", 0.1, 10.0, 3.0, step=0.1)
        integral = quad(
            lambda E: boltzmann_energy_dist(E, T), 0, E0 * k * T
        )[0]
        st.latex(
            r"F(E_0) = \int_0^{E_0} f(E) dE = "
            + f"{integral:.4f}"
        )
        st.write(
            f"A fração acumulada de partículas com energia até {E0:.1f} kT é aproximadamente {integral:.4f}."
        )

    elif options[choice] == "stefan_boltzmann":
        st.subheader("Potência radiada (Lei de Stefan-Boltzmann)")
        T1 = st.number_input("Temperatura inicial T1 (K)", value=300.0)
        T2 = st.number_input("Temperatura final T2 (K)", value=1000.0)
        A = st.number_input("Área (m²)", value=1.0)
        emissivity = st.slider("Emissividade", 0.0, 1.0, 1.0, 0.01)
        potencia = stefan_boltzmann_power(T2, A, emissivity)
        energia = energia_total_radiada_numeric(T1, T2, A, emissivity)
        st.latex(r"P = \varepsilon \sigma A T^4")
        st.write(f"Potência radiada a T2 = {T2} K: {potencia:.3e} W")
        st.write(f"Energia total radiada de {T1} K a {T2} K: {energia:.3e} J")

    # ---------------------------
    # Pequeno laboratório numérico — comparação candidatos
    # ---------------------------
    st.markdown("---")
    st.header("Mini-laboratório: comparação de funções candidatas")

    T_lab = st.slider("Temperatura para o mini-lab (K)", 50, 1500, 300, step=10, key="lab_T")
    E_vals_lab = np.linspace(0, 10 * k * T_lab, 500)

    fA = np.exp(-E_vals_lab / (k * T_lab))
    fB = np.sqrt(E_vals_lab) * np.exp(-E_vals_lab / (k * T_lab))

    # Normalizar
    normA = np.trapz(fA, E_vals_lab)
    normB = np.trapz(fB, E_vals_lab)
    fA_norm = fA / normA
    fB_norm = fB / normB

    # Médias
    meanA = np.trapz(E_vals_lab * fA_norm, E_vals_lab)
    meanB = np.trapz(E_vals_lab * fB_norm, E_vals_lab)

    st.write("Distribuição A: \(f_A(E) \propto e^{-E/(kT)}\)")
    st.write(f"Integral (normalização) = {normA:.4f}")
    st.write(f"Energia média calculada = {meanA / k:.4f} kT (esperado: 3/2)")

    st.write("Distribuição B: \(f_B(E) \propto \sqrt{E} e^{-E/(kT)}\)")
    st.write(f"Integral (normalização) = {normB:.4f}")
    st.write(f"Energia média calculada = {meanB / k:.4f} kT (esperado: 3/2)")

    fig_lab = go.Figure()
    fig_lab.add_trace(go.Scatter(x=E_vals_lab / (k * T_lab), y=fA_norm, mode="lines", name=r"$f_A(E)$"))
    fig_lab.add_trace(go.Scatter(x=E_vals_lab / (k * T_lab), y=fB_norm, mode="lines", name=r"$f_B(E)$"))
    fig_lab.update_layout(
        title="Comparação das funções candidatas (normalizadas)",
        xaxis_title=r"$E / (kT)$",
        yaxis_title="f(E) normalizado",
    )
    st.plotly_chart(fig_lab, use_container_width=True)

    # Deduções matemáticas detalhadas
    st.markdown("---")
    st.header("Deduções matemáticas — passo a passo")

    st.latex(r"""
    f_v(v) = 4\pi\left(\frac{m}{2\pi k T}\right)^{3/2} v^2 e^{- \frac{m v^2}{2 k T}}
    """)

    st.write(r"Usando \(E=\tfrac{1}{2} m v^2\) e a mudança de variável \(dv = dE / (m v)\), substituindo \(v = \sqrt{2E/m}\) obtemos:")

    st.latex(r"""
    f_E(E) \propto \sqrt{E}\, e^{-E/(kT)}.
    """)

    st.write("A normalização analítica leva ao fator:")

    st.latex(r"""
    f_E(E) = \frac{2}{\sqrt{\pi}} \frac{1}{(kT)^{3/2}} \sqrt{E}\, e^{-E/(kT)}.
    """)

    st.write(r"Energia mais provável e média (derivações mostradas anteriormente):")

    st.latex(r"""
    E_{mp} = \tfrac{1}{2}kT \\
    \langle E \rangle = \tfrac{3}{2}kT
    """)

    st.write("Ferramentas matemáticas usadas e porquê:")
    st.write("- Álgebra e manipulação de constantes: para ligar \(PV=nRT\) a energia por partícula.")
    st.write("- Cálculo variacional (Lagrange): para derivar a forma exponencial como a distribuição mais provável.")
    st.write("- Mudança de variáveis e Jacobiano: para transformar de \(f(\mathbf v)\) para \(f(E)\) e contar estados.")
    st.write("- Integração (gaussiana/função gama): para normalizar e calcular médias.")

if __name__ == "__main__":
    main()
