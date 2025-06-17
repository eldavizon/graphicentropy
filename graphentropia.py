import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import quad
import seaborn as sns

# Configurações iniciais
plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid")
sns.set_palette("husl")
k = 1.380649e-23  # Constante de Boltzmann em J/K


## 1. Distribuição de Boltzmann - Versão Corrigida
def boltzmann_distribution(E, T):
    """Função da distribuição de Boltzmann"""
    with np.errstate(all='ignore'):
        coeff = (2 / np.sqrt(np.pi)) * (1 / (k * T) ** (3 / 2))
        return coeff * np.sqrt(E) * np.exp(-E / (k * T))


def plot_boltzmann_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)

    T_init = 300
    E = np.linspace(1e-25, 5e-20, 1000)  # Evitando E=0

    line, = ax.plot(E, boltzmann_distribution(E, T_init), lw=2)
    ax.set_title('Distribuição de Boltzmann')
    ax.set_xlabel('Energia (J)')
    ax.set_ylabel('Densidade de Probabilidade')
    ax.grid(True)

    # Linhas verticais
    emp_line = ax.axvline(0.5 * k * T_init, color='r', linestyle='--', label='Energia mais provável')
    eme_line = ax.axvline(1.5 * k * T_init, color='g', linestyle='--', label='Energia média')
    ax.legend()

    # Slider corrigido
    ax_temp = plt.axes([0.25, 0.1, 0.65, 0.03])
    temp_slider = Slider(
        ax=ax_temp,
        label='Temperatura (K)',
        valmin=100,
        valmax=1000,
        valinit=T_init,
    )

    def update(val):
        T = temp_slider.val
        y = boltzmann_distribution(E, T)
        line.set_ydata(y)

        # Atualiza linhas verticais
        emp_line.set_data([0.5 * k * T, 0.5 * k * T], [0, np.max(y)])
        eme_line.set_data([1.5 * k * T, 1.5 * k * T], [0, np.max(y)])

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    temp_slider.on_changed(update)
    plt.show()


## 2. Entropia vs Energia
def entropy(Omega):
    return k * np.log(Omega)


def plot_entropy_temperature():
    # Configurações iniciais
    k = 1.380649e-23  # Constante de Boltzmann
    N = 1.0  # Vamos trabalhar com N=1 para evitar overflow

    # Valores de energia fisicamente razoáveis
    E = np.linspace(1e-21, 5e-21, 100)  # Intervalo típico para poucas partículas

    # Cálculo CORRETO usando logaritmos para evitar overflow
    log_Omega = (3 * N / 2) * np.log(E)  # ln(Ω) = (3N/2)ln(E)
    S = k * log_Omega  # S = k ln(Ω)

    # Derivada numérica estável
    dSdE = np.gradient(S, E)
    T = 1 / dSdE

    # Plotagem
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1: Entropia vs Energia
    ax1.plot(E, S, 'b-')
    ax1.set_title('Entropia vs Energia\nS proporcional ln(E)')
    ax1.set_xlabel('Energia (J)')
    ax1.set_ylabel('Entropia (J/K)')
    ax1.grid(True)

    # Gráfico 2: Temperatura vs Energia
    ax2.plot(E, T, 'r-')
    ax2.set_title('Temperatura vs Energia\nT = 2E/(3Nk)')
    ax2.set_xlabel('Energia (J)')
    ax2.set_ylabel('Temperatura (K)')
    ax2.grid(True)

    # Linha teórica esperada
    T_teorico = (2 * E) / (3 * N * k)
    ax2.plot(E, T_teorico, 'k--', label='Relação teórica')
    ax2.legend()

    plt.tight_layout()
    plt.show()

## 3. Fração de partículas - Versão Corrigida
def plot_energy_fraction():
    T = 300  # Temperatura em Kelvin
    k = 1.380649e-23  # Constante de Boltzmann

    def boltzmann_distribution(E, T):
        """Distribuição de Boltzmann correta e estável numericamente"""
        with np.errstate(all='ignore'):
            # Calcula em escala logarítmica para melhor estabilidade numérica
            log_coeff = np.log(2 / np.sqrt(np.pi)) + 1.5 * np.log(1 / (k * T))
            log_val = log_coeff + 0.5 * np.log(E) - E / (k * T)
            return np.exp(log_val)

    # Intervalo de energia cuidadosamente escolhido
    E_min = 1e-24  # Valor mínimo de energia (próximo de zero)
    E_max = 5 * k * T  # Valor máximo baseado na temperatura
    E_values = np.logspace(np.log10(E_min), np.log10(E_max), 500)

    # Calcula a distribuição
    f_E = boltzmann_distribution(E_values, T)

    # Calcula a integral total para normalização
    total_integral = np.trapezoid(f_E, E_values)

    # Verifica se a integral é válida
    if total_integral <= 0:
        print("Erro: Integral inválida. Verifique os parâmetros.")
        return

    # Normaliza a distribuição
    f_E_normalized = f_E / total_integral

    # Calcula a fração cumulativa
    cumulative = np.cumsum(f_E_normalized[:-1] * np.diff(E_values))
    cumulative = np.insert(cumulative, 0, 0)  # Adiciona ponto inicial

    # Plotagem
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1: Distribuição de Boltzmann
    ax1.semilogx(E_values, f_E_normalized, 'b-', lw=2)
    ax1.set_title('Distribuição de Energia (Boltzmann)')
    ax1.set_xlabel('Energia (J)')
    ax1.set_ylabel('Densidade de Probabilidade')
    ax1.grid(True, which="both", ls="-")

    # Gráfico 2: Fração cumulativa
    ax2.semilogx(E_values, cumulative, 'r-', lw=2)
    ax2.set_title('Fração Cumulativa de Partículas')
    ax2.set_xlabel('Energia (J)')
    ax2.set_ylabel('Fração com E ≤ $E_0$')
    ax2.grid(True, which="both", ls="-")

    # Adiciona marcações de porcentagem
    for frac in [0.25, 0.5, 0.75, 0.9]:
        idx = np.argmin(np.abs(cumulative - frac))
        ax2.plot(E_values[idx], cumulative[idx], 'ko')
        ax2.annotate(f'{frac * 100:.0f}%',
                     xy=(E_values[idx], cumulative[idx]),
                     xytext=(10, 10), textcoords='offset points')

    plt.tight_layout()
    plt.show()

## Menu principal
def main():
    print("Aplicações do Cálculo I na Química - Visualizações")
    print("1. Distribuição de Boltzmann (interativa)")
    print("2. Entropia e Temperatura")
    print("3. Fração de partículas em intervalo de energia")

    while True:
        choice = input("Escolha o gráfico (1-3) ou 'q' para sair: ")

        if choice == '1':
            plot_boltzmann_distribution()
        elif choice == '2':
            plot_entropy_temperature()
        elif choice == '3':
            plot_energy_fraction()
        elif choice.lower() == 'q':
            break
        else:
            print("Opção inválida. Tente novamente.")


if __name__ == "__main__":
    main()