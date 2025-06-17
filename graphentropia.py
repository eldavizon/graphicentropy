import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import quad
import seaborn as sns

# Configurações iniciais - usando estilo disponível
plt.style.use('seaborn-v0_8')  # Alternativa para versões mais recentes
sns.set_style("whitegrid")
sns.set_palette("husl")
k = 1.380649e-23  # Constante de Boltzmann em J/K


## 1. Distribuição de Boltzmann
def boltzmann_distribution(E, T):
    """Função da distribuição de Boltzmann"""
    return (2 / np.sqrt(np.pi)) * (1 / (k * T) ** (3 / 2)) * np.sqrt(E) * np.exp(-E / (k * T))


def plot_boltzmann_distribution():
    # Configuração da figura
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)

    # Valores iniciais
    T_init = 300  # Temperatura inicial em Kelvin
    E = np.linspace(0, 5e-20, 1000)  # Energia em Joules

    # Plot inicial
    line, = ax.plot(E, boltzmann_distribution(E, T_init), lw=2)
    ax.set_title('Distribuição de Boltzmann')
    ax.set_xlabel('Energia (J)')
    ax.set_ylabel('Densidade de Probabilidade')
    ax.grid(True)

    # Adicionando linhas para energia mais provável e média
    emp_line = ax.axvline(0.5 * k * T_init, color='r', linestyle='--', label='Energia mais provável')
    eme_line = ax.axvline(1.5 * k * T_init, color='g', linestyle='--', label='Energia média')
    ax.legend()

    # Adicionando slider para temperatura
    ax_temp = plt.axes([0.25, 0.1, 0.65, 0.03])
    temp_slider = Slider(
        ax=ax_temp,
        label='Temperatura (K)',
        valmin=100,
        valmax=1000,
        valinit=T_init,
    )

    # Função de atualização
    def update(val):
        T = temp_slider.val
        line.set_ydata(boltzmann_distribution(E, T))
        emp_line.set_xdata(0.5 * k * T)
        eme_line.set_xdata(1.5 * k * T)
        fig.canvas.draw_idle()

    temp_slider.on_changed(update)

    plt.show()


## 2. Entropia vs Energia
def entropy(Omega):
    """Função de entropia"""
    return k * np.log(Omega)


def plot_entropy_temperature():
    # Criando dados para Omega (número de microestados)
    E = np.linspace(1e-21, 1e-19, 100)  # Energia em Joules
    Omega = E ** 3  # Número de microestados (relação simplificada)

    # Calculando entropia
    S = entropy(Omega)

    # Calculando a derivada numérica (1/T)
    dSdE = np.gradient(S, E)
    T = 1 / dSdE

    # Criando a figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Entropia vs Energia
    ax1.plot(E, S, 'b-')
    ax1.set_title('Entropia vs Energia')
    ax1.set_xlabel('Energia (J)')
    ax1.set_ylabel('Entropia (J/K)')
    ax1.grid(True)

    # Plot Temperatura vs Energia
    ax2.plot(E, T, 'r-')
    ax2.set_title('Temperatura vs Energia')
    ax2.set_xlabel('Energia (J)')
    ax2.set_ylabel('Temperatura (K)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


## 3. Fração de partículas em um intervalo de energia
def plot_energy_fraction():
    T = 300  # Temperatura em Kelvin

    # Definindo a função de distribuição normalizada
    def f(E):
        return boltzmann_distribution(E, T)

    # Calculando a integral total para normalização
    total_integral, _ = quad(f, 0, np.inf)

    # Criando valores de energia
    E_values = np.linspace(0, 5e-20, 500)

    # Calculando a fração cumulativa
    cumulative = np.zeros_like(E_values)
    for i, E in enumerate(E_values):
        cumulative[i], _ = quad(f, 0, E)

    cumulative /= total_integral  # Normalizando

    # Criando a figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot da distribuição
    ax1.plot(E_values, f(E_values) / total_integral, 'b-')
    ax1.set_title('Distribuição de Energia')
    ax1.set_xlabel('Energia (J)')
    ax1.set_ylabel('Densidade de Probabilidade')
    ax1.grid(True)

    # Plot da fração cumulativa
    ax2.plot(E_values, cumulative, 'r-')
    ax2.set_title('Fração de Partículas com Energia ≤ E')
    ax2.set_xlabel('Energia (J)')
    ax2.set_ylabel('Fração Cumulativa')
    ax2.grid(True)

    # Adicionando alguns valores específicos
    for frac in [0.25, 0.5, 0.75, 0.9]:
        idx = np.argmin(np.abs(cumulative - frac))
        ax2.plot(E_values[idx], cumulative[idx], 'ko')
        ax2.annotate(f'{frac * 100:.0f}% at {E_values[idx]:.1e} J',
                     xy=(E_values[idx], cumulative[idx]),
                     xytext=(10, 10), textcoords='offset points')

    plt.tight_layout()
    plt.show()


## Menu interativo para selecionar qual gráfico visualizar
def main():
    print("Aplicações do Cálculo I na Química - Visualizações")
    print("1. Distribuição de Boltzmann (interativa)")
    print("2. Entropia e Temperatura")
    print("3. Fração de partículas em intervalo de energia")

    choice = input("Escolha o gráfico (1-3) ou 'q' para sair: ")

    while choice.lower() != 'q':
        if choice == '1':
            plot_boltzmann_distribution()
        elif choice == '2':
            plot_entropy_temperature()
        elif choice == '3':
            plot_energy_fraction()
        else:
            print("Opção inválida. Tente novamente.")

        choice = input("Escolha o gráfico (1-3) ou 'q' para sair: ")


if __name__ == "__main__":
    main()