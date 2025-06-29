# graphicentropy

# Aplicações do Cálculo I na Química - Explicação em 3 Níveis

## 1. Distribuição de Boltzmann (Interativa)

### Nível Físico-Químico
A distribuição de Boltzmann descreve como as moléculas de um gás se distribuem em diferentes energias quando em equilíbrio térmico. Em qualquer temperatura acima do zero absoluto:
- Algumas moléculas têm baixa energia
- Algumas têm alta energia
- A maioria está em energias intermediárias

### Nível Matemático
A distribuição é dada por:

$$
f(E) = \frac{2}{\sqrt{\pi}} (kT)^{-3/2} \sqrt{E} e^{-E/kT}
$$

- **Derivada**: Usada para encontrar o pico da distribuição $$(dP/dE = 0)$$
- **Integral**: Calcula a energia média $$⟨E⟩ = ∫E·f(E)dE = 3kT/2$$
- **kT**: Escala de energia característica do sistema

### O que o Gráfico Mostra
- **Eixo X**: Energia das partículas (Joules)
- **Eixo Y**: Probabilidade relativa
- **Linhas marcantes**:
  - Vermelha: Energia mais provável (1/2 kT) - onde a derivada se anula
  - Verde: Energia média (3/2 kT) - resultado da integração
- **Interatividade**: Ao variar T com o slider:
  - A curva se alarga (mais partículas em altas energias)
  - As linhas marcantes se deslocam proporcionalmente

## 2. Entropia e Temperatura

### Nível Físico-Químico
A entropia (S) mede a desordem molecular. A temperatura (T) está intimamente relacionada a como a entropia muda quando adicionamos energia:
- Sistemas com alta T precisam de muita energia para aumentar S
- Sistemas com baixa T têm S muito sensível a mudanças de E

### Nível Matemático
Relações fundamentais:
$$
S = k \ln \Omega \approx \frac{3Nk}{2}\ln E + C
$$
$$
\frac{1}{T} = \left(\frac{\partial S}{\partial E}\right)_{V,N}
$$

- **Derivada logarítmica**: $$∂(ln E)/∂E = 1/E → T = 2E/(3Nk)$$

- **Integração (para processos irreversíveis)**:$$ΔS = ∫(1/T)dE $$

### O que os Gráficos Mostram
**Gráfico 1 (Entropia vs Energia)**:
- Curva côncava para baixo (derivada decrescente)
- Inclinação = 1/T (diminui com E crescente)

**Gráfico 2 (Temperatura vs Energia)**:
- Linha reta confirmando T ∝ E
- Comparação entre derivada numérica e teoria

## 3. Fração de Partículas em Intervalo de Energia

### Nível Físico-Químico
Nem todas as moléculas têm a mesma energia! Esta análise mostra:
- Qual porcentagem das moléculas está abaixo de certa energia
- Como a energia térmica kT divide a distribuição

### Nível Matemático
Fração cumulativa:
$$
F(E \leq E_0) = \int_0^{E_0} f(E)dE
$$
- **Integração numérica**: Calculada por soma de trapézios
- **Normalização**: $$∫f(E)dE = 1 (100% das partículas)$$
- **Escala log**: Necessária para capturar variações em múltiplas ordens de grandeza

### O que os Gráficos Mostram
**Gráfico 1 (Distribuição)**:
- Pico em $$E = kT/2$$
- Cauda longa para altas energias (raras mas importantes)

**Gráfico 2 (Fração Cumulativa)**:
- Pontos notáveis:
  - 50% em E ≈ 0.7kT
  - 90% em E ≈ 2.3kT
- Curva sigmoide típica de distribuições de probabilidade

## 4. Lei de Stefan-Boltzmann

### Nível Físico-Químico
Todos os corpos emitem radiação térmica:
- Potência irradiada depende dramaticamente da temperatura $$(∝ T⁴)$$
- A energia total emitida é a integral da potência no tempo

### Nível Matemático
Lei de Stefan-Boltzmann:
$$P = \epsilon \sigma A T^4$$
Energia total:
$$
E = \int_{t_1}^{t_2} P(T) dt = \epsilon \sigma A \int T^4 dt
$$
- **Derivada (sensibilidade térmica)**: $$dP/dT = 4εσAT³ $$
- **Integral**: Área sob a curva P(T)

### O que o Gráfico Mostra
- Curva superlinear (domínio do termo T⁴)
- Área sombreada:
  - Representa energia total irradiada
  - Cresce rapidamente com T
- Sliders demonstram:
  - Emissividade (ε): 0 (perfeito espelho) a 1 (corpo negro ideal)
  - Efeito da área superficial
  - Dependência não-linear com T

## Conexões com Cálculo I

1. **Derivadas**:
   - Taxas de variação: $$dS/dE = 1/T$$
   - Otimização (encontrar máximos como E_mp)
   - Sensibilidade $$(dP/dT)$$

2. **Integrais**:
   - Somas contínuas (probabilidades acumuladas)
   - Cálculo de valores médios (⟨E⟩)
   - Energias totais (áreas sob curvas)

3. **Funções**:
   - Exponenciais: decaimento de probabilidades
   - Logarítmicas: transformação de relações multiplicativas
   - Polinomiais: leis de escala (T⁴)

Cada visualização foi projetada para conectar:
1. Os fenômenos físicos observáveis
2. As formulações matemáticas precisas
3. As representações gráficas intuitivas
