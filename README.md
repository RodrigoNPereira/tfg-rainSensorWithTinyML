# Pluviômetro Sem Partes Móveis: Aplicação de Modelos CNN para Detecção Sonora de Chuvas em Dispositivos de Borda

**Trabalho Final de Graduação - Engenharia da Computação**  
**Universidade Federal de Itajubá (UNIFEI)**

---

## 📋 Resumo

Este projeto apresenta o desenvolvimento de um pluviômetro sem partes móveis baseado na detecção sonora de chuva utilizando modelos de redes neurais convolucionais (CNN) implementados em dispositivos de borda. O sistema utiliza técnicas de Machine Learning com TinyML para classificar diferentes intensidades de chuva através da análise de sinais de áudio.

### Características Principais
- **Detecção sem partes móveis**: Sistema baseado puramente em análise de áudio
- **Classificação de intensidade**: Capaz de distinguir entre "Sem chuva", "Chuva baixa", "Chuva média" e "Chuva alta"
- **Implementação em borda**: Utilizando TinyML para processamento local
- **Baixo consumo**: Otimizado para dispositivos embarcados

---

## 🔬 Metodologia

### Processamento de Sinais de Áudio
O projeto implementa uma pipeline completa de processamento de sinais de áudio que inclui:

1. **Pré-processamento**
   - Taxa de amostragem: 16 kHz
   - Duração das janelas: 6 segundos
   - Aplicação de janela Hamming

2. **Extração de Características**
   - **Mel Filter Banks (MFE)**: 41 filtros mel
   - **Spectrogramas**: Análise tempo-frequência
   - **MFCCs**: Coeficientes cepstrais em escala mel
   - **Framing**: Divisão em quadros com sobreposição

### Classes de Classificação
- **No Rain**: Ausência de chuva
- **Low Rain**: Chuva de baixa intensidade
- **Medium Rain**: Chuva de intensidade moderada
- **High Rain**: Chuva de alta intensidade

---

## 📁 Estrutura do Projeto

```
tfg-rainSensorWithTinyML/
├── README.md
├── Audios/                               # Conjunto de dados de áudio
│   ├── No.wav                           # Áudio sem chuva
│   ├── Low.wav                          # Áudio chuva baixa
│   ├── Medium.wav                       # Áudio chuva média
│   ├── High.wav                         # Áudio chuva alta
│   └── signalAmplitudeForDifferentClasses.png
├── images/                              # Visualizações e gráficos
│   ├── hamming.png                      # Janela Hamming
│   ├── mfe.png                          # Mel Filter Banks
│   ├── signalAmplitudeForDifferentClasses.png
│   ├── spectogram6seconds.png           # Espectrograma
│   └── spectogramForDifferntClasses.png
├── TFG-ChuvaClassifier_inferencing/     # Biblioteca Edge Impulse
│   ├── library.properties
│   └── src/
│       ├── TFG-ChuvaClassifier_inferencing.h
│       ├── edge-impulse-sdk/           # SDK Edge Impulse
│       └── model-parameters/           # Parâmetros do modelo
├── featureEngineering.ipynb            # Engenharia de características
├── MFE.ipynb                           # Implementação Mel Filter Banks
├── SpectogramsMFCCs.ipynb              # Análise de espectrogramas e MFCCs
├── sem43.ipynb                         # Processamento e análise principal
├── testeDataset.ipynb                  # Teste do conjunto de dados
└── Pluviometro_Sem_Partes_Moveis__Aplicacao_de_modelos_CNN_para_Deteccao_Sonora_de_Chuvas_em_Dispositivos_de_Borda.pdf
```

---

## 📓 Notebooks Jupyter

### 1. `featureEngineering.ipynb`
**Descrição**: Implementa a pipeline principal de extração de características dos sinais de áudio.

**Funcionalidades**:
- Carregamento de arquivos de áudio usando `librosa`
- Visualização de amplitude do sinal no domínio do tempo
- Geração de espectrogramas para diferentes classes de chuva
- Análise comparativa entre janelas de 30 segundos e 6 segundos

**Principais funções**:
```python
# Carregamento de áudio
No_rain, sample_rate = librosa.load("Audios/No.wav", sr=16000, duration=30)
Low, sample_rate = librosa.load("Audios/Low.wav", sr=16000, duration=30)

# Geração de espectrogramas
frequencies, times, Sxx = spectrogram(No_rain, sample_rate)
```

### 2. `MFE.ipynb`
**Descrição**: Implementação completa dos Mel Filter Banks (MFE) para extração de características.

**Características técnicas**:
- 41 filtros mel
- Pré-ênfase com α = 0.97
- FFT de 256 pontos
- Frequência de corte inferior: 80 Hz
- Normalização das energias dos bancos de filtros

**Função principal**:
```python
def calc_plot_mfe_features(audio, sample_rate, alpha=0.97, NFFT=256, 
                          low_freq_cut=80, nfilt=41, noise_floor_dB=-64,
                          frame_size=0.08, frame_stride=0.025):
    # Implementação completa dos MFE
    # Retorna matriz de características (num_frames x nfilt)
```

### 3. `SpectogramsMFCCs.ipynb`
**Descrição**: Análise detalhada de espectrogramas e coeficientes MFCC com múltiplas visualizações.

**Funcionalidades**:
- Análise de frames individuais
- Comparação entre diferentes janelas temporais
- Implementação de MFCC
- Visualizações 2x2 de diferentes classes de chuva

### 4. `sem43.ipynb`
**Descrição**: Notebook principal com processamento de dados completo e preparação para o modelo.

**Funcionalidades**:
- Estruturas de dados para manipulação de arquivos
- Processamento em lote de dados de áudio
- Preparação de dados para treinamento
- Integração com Edge Impulse

### 5. `testeDataset.ipynb`
**Descrição**: Validação e teste do conjunto de dados preparado.

**Funcionalidades**:
- Verificação da qualidade dos dados
- Testes de consistência
- Validação das características extraídas

---

## 🔧 Implementação TinyML

### Edge Impulse - Plataforma de Desenvolvimento

O projeto utiliza a plataforma **Edge Impulse** como ferramenta principal para criar, treinar e otimizar o modelo de Machine Learning para dispositivos embarcados. Edge Impulse é uma plataforma end-to-end que facilita o desenvolvimento de soluções TinyML.

#### Pipeline de Desenvolvimento no Edge Impulse

1. **Coleta e Upload de Dados**
   - Upload dos arquivos de áudio das 4 classes (No Rain, Low, Medium, High)
   - Divisão automática em conjuntos de treino e teste
   - Visualização e análise dos dados importados

2. **Design do Impulse**
   - **Bloco de Processamento**: Configuração dos parâmetros de áudio
     - Janela de tempo: 6000ms
     - Taxa de amostragem: 16 kHz
   - **Extração de Características**: Mel Filter Banks (MFE)
     - 41 coeficientes MFE
     - Configuração de frame size e frame stride
   - **Bloco de Aprendizado**: Rede Neural Convolucional (CNN)
     - Arquitetura otimizada para classificação de 4 classes
     - Camadas convolucionais e pooling

3. **Treinamento do Modelo**
   - Configuração de hiperparâmetros (épocas, taxa de aprendizado)
   - Treinamento com validação cruzada
   - Visualização de métricas (acurácia, loss, matriz de confusão)
   - Análise de performance por classe

4. **Otimização para Dispositivos de Borda**
   - **Quantização**: Conversão do modelo para INT8
   - **EON Compiler**: Otimização automática de memória e velocidade
   - Análise de recursos (RAM, Flash, Latência)
   - Teste de performance em diferentes targets

5. **Geração da Biblioteca Arduino**
   - Exportação como biblioteca C++ otimizada
   - Biblioteca gerada: `TFG-ChuvaClassifier_inferencing`
   - Compatível com Arduino IDE
   - Inclui toda a pipeline de inferência

#### Biblioteca Edge Impulse Gerada

**Nome**: `TFG-ChuvaClassifier_inferencing`
- **Versão**: 1.0.1
- **Tamanho do Modelo**: Otimizado para < 64KB RAM
- **Compatibilidade**: Arduino Nano 33 BLE Sense, Portenta H7, Nicla Vision
- **Requisitos Mínimos**: ARM Cortex-M4, 64KB RAM, 512KB Flash

**Estrutura da Biblioteca**:
```
TFG-ChuvaClassifier_inferencing/
├── library.properties              # Metadados da biblioteca
├── src/
│   ├── TFG-ChuvaClassifier_inferencing.h  # Header principal
│   ├── edge-impulse-sdk/          # SDK completo do Edge Impulse
│   │   ├── CMSIS/                 # Bibliotecas CMSIS-DSP
│   │   ├── dsp/                   # Processamento de sinais
│   │   ├── classifier/            # Motor de classificação
│   │   └── porting/               # Adaptações de hardware
│   └── model-parameters/          # Parâmetros do modelo treinado
│       ├── model_metadata.h       # Metadados do modelo
│       └── model_variables.h      # Pesos e configurações
```

**Características do Modelo**:
- **Entrada**: Características MFE (41 coeficientes × N frames)
- **Arquitetura**: CNN otimizada para dispositivos embarcados
- **Saída**: 4 classes de classificação com scores de confiança
- **Otimizações Aplicadas**:
  - Quantização INT8 para redução de 75% no tamanho
  - Otimização de operações com CMSIS-NN
  - Uso eficiente de memória com alocação estática

#### Vantagens do Edge Impulse

✅ **Facilidade de Uso**: Interface visual para todo o pipeline de ML  
✅ **Otimização Automática**: EON Compiler otimiza para o hardware alvo  
✅ **Suporte Multiplataforma**: Exportação para Arduino, STM32, ESP32, etc.  
✅ **Análise de Performance**: Métricas detalhadas de uso de recursos  
✅ **Versionamento**: Controle de versões do modelo e datasets  
✅ **Testing em Tempo Real**: Teste do modelo diretamente na plataforma  

### Configuração do Hardware

**Exemplo de código Arduino**:
```cpp
#include <TFG-ChuvaClassifier_inferencing.h>
#include <PDM.h>

// Buffer para armazenar amostras de áudio
static int16_t audio_buffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT];

void setup() {
    Serial.begin(115200);
    
    // Inicializa microfone PDM
    PDM.onReceive(pdm_data_ready_inference_callback);
    PDM.begin(1, EI_CLASSIFIER_FREQUENCY);
}

void loop() {
    // Executa inferência
    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &get_audio_signal_data;
    
    ei_impulse_result_t result = {0};
    run_classifier(&signal, &result, false);
    
    // Exibe resultados
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        Serial.print(result.classification[ix].label);
        Serial.print(": ");
        Serial.println(result.classification[ix].value);
    }
}
```

**Dependências Arduino**:
- `Arduino_LSM9DS1`: Sensor IMU (se utilizado)
- `PDM`: Biblioteca de microfone PDM
- `Arduino_OV767X`: Câmera (se utilizada para outros sensores)

---

## 📊 Processamento de Sinais

### Pipeline de Extração de Características

1. **Pré-processamento**:
   ```python
   # Pré-ênfase
   audio = np.append(audio[0], audio[1:] - alpha * audio[:-1])
   
   # Framing
   frame_length = int(round(frame_size * sample_rate))
   frame_step = int(round(frame_stride * sample_rate))
   ```

2. **Windowing**:
   ```python
   # Aplicação da janela Hamming
   frames *= np.hamming(frame_length)
   ```

3. **FFT e Espectro de Potência**:
   ```python
   # FFT e cálculo do espectro de potência
   mag_frames = np.absolute(fft.fft(frames, NFFT))
   pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
   ```

4. **Mel Filter Banks**:
   ```python
   # Aplicação dos filtros mel
   filter_banks = np.dot(pow_frames, fbank.T)
   filter_banks = 20 * np.log10(filter_banks)  # Conversão para dB
   ```

5. **Normalização**:
   ```python
   # Normalização Z-score
   filter_banks -= np.mean(filter_banks, axis=0)
   filter_banks /= np.std(filter_banks, axis=0)
   ```

---

## 🎯 Resultados e Visualizações

### Análise Temporal
- **Gráficos de amplitude**: Visualização das diferenças entre classes no domínio do tempo
- **Comparação entre classes**: Identificação de padrões específicos para cada intensidade

### Análise Frequencial
- **Espectrogramas**: Representação tempo-frequência para cada classe
- **Mel Filter Banks**: Características otimizadas para percepção auditiva
- **Colorbar visualization**: Mapas de calor para visualização das energias

### Visualizações Implementadas
```python
# Subplot 2x2 para comparação entre classes
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Mel Filter Banks', fontsize=18)

# Adição de colorbars individuais
cax0 = axs[0, 0].imshow(no_rain.T, aspect='auto', cmap='coolwarm')
fig.colorbar(cax0, ax=axs[0, 0], label='Filter Bank Value')
```

---

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install librosa matplotlib numpy scipy soundfile ffmpeg-python pandas
```

### Execução dos Notebooks
1. **Feature Engineering**: Execute `featureEngineering.ipynb` para análise inicial
2. **MFE Processing**: Execute `MFE.ipynb` para extração de características Mel
3. **Spectrograms Analysis**: Execute `SpectogramsMFCCs.ipynb` para análise detalhada
4. **Main Processing**: Execute `sem43.ipynb` para processamento principal
5. **Dataset Testing**: Execute `testeDataset.ipynb` para validação

### Implementação em Hardware
1. Instale a biblioteca Arduino IDE
2. Importe `TFG-ChuvaClassifier_inferencing`
3. Configure o hardware (Arduino Nano 33 BLE Sense recomendado)
4. Carregue o código de inferência

---

## 📈 Características Técnicas

### Parâmetros de Processamento
- **Taxa de amostragem**: 16 kHz
- **Duração da janela**: 6 segundos
- **Tamanho do frame**: 0.08 segundos (80ms)
- **Stride do frame**: 0.025 segundos (25ms)
- **FFT**: 256 pontos
- **Filtros Mel**: 41 filtros
- **Faixa de frequência**: 80 Hz - 8 kHz

### Otimizações para TinyML
- **Quantização**: Redução da precisão para 8-bits
- **Compressão de modelo**: Técnicas de pruning
- **Otimização de memória**: Uso eficiente de RAM
- **Processamento em tempo real**: Latência otimizada

---

## 🔬 Contribuições Científicas

### Inovações do Projeto
1. **Pluviômetro sem partes móveis**: Eliminação de componentes mecânicos
2. **Classificação multi-classe**: Detecção de diferentes intensidades
3. **Implementação embarcada**: TinyML para dispositivos de baixo consumo
4. **Pipeline otimizada**: Processamento eficiente de sinais de áudio

### Aplicações Potenciais
- **Agricultura de precisão**: Monitoramento automatizado de chuva
- **Smart cities**: Redes de sensores urbanos
- **Meteorologia**: Complemento a estações meteorológicas
- **IoT ambiental**: Integração em sistemas de monitoramento

---

## 📚 Tecnologias Utilizadas

### Software
- **Python**: Linguagem principal para processamento
- **Librosa**: Processamento de áudio
- **NumPy/SciPy**: Computação científica
- **Matplotlib**: Visualizações
- **Edge Impulse**: Plataforma TinyML
- **Jupyter Notebooks**: Desenvolvimento e documentação

### Hardware
- **Arduino Nano 33 BLE Sense**: Plataforma de desenvolvimento
- **Microfone MEMS**: Captura de áudio
- **ARM Cortex-M4**: Processador embarcado

---

## 🎓 Contexto Acadêmico

**Instituição**: Universidade Federal de Itajubá (UNIFEI)  
**Curso**: Engenharia da Computação  
**Modalidade**: Trabalho Final de Graduação (TFG)  
**Área de Concentração**: Machine Learning, TinyML, Processamento de Sinais

### Objetivos Educacionais
- Aplicação prática de conceitos de Machine Learning
- Desenvolvimento de sistemas embarcados
- Processamento digital de sinais
- Integração hardware-software
- Pesquisa e inovação tecnológica

---

## 📄 Documentação Adicional

O documento completo do TFG está disponível em:
`Pluviometro_Sem_Partes_Moveis__Aplicacao_de_modelos_CNN_para_Deteccao_Sonora_de_Chuvas_em_Dispositivos_de_Borda.pdf`

### Estrutura do Documento
- Resumo e Abstract
- Introdução e Objetivos
- Fundamentação Teórica
- Metodologia
- Resultados e Discussão
- Conclusões e Trabalhos Futuros
- Referências Bibliográficas

---

## 🤝 Contribuições

Este projeto representa uma contribuição significativa para as áreas de:
- **TinyML**: Implementação de ML em dispositivos de borda
- **Agricultura de Precisão**: Monitoramento ambiental automatizado
- **Processamento de Sinais**: Técnicas avançadas de análise de áudio
- **IoT Ambiental**: Sensores inteligentes para monitoramento climático

---

## 📞 Contato

- **Desenvolvido por**: Rodrigo Pereira, Felipe Fernandes, Bruno Batista
- **Orientação**: José Alberto Ferreira Filho, Marcelo José Rovai
- **Universidade Federal de Itajubá (UNIFEI)**  
- **Curso de Engenharia da Computação**

Sinta-se a vontade para entrar em contato comigo caso haja alguma dúvida ou gostaria de conversar sobre o projeto por meio do email `rodrigonpgma@gmail.com`

---

*Este README fornece uma visão abrangente do projeto TFG desenvolvido na UNIFEI, demonstrando a aplicação prática de técnicas de Machine Learning e TinyML para solução de problemas reais de monitoramento ambiental.*