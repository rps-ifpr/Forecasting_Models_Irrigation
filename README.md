# Transformer Models and Recurrent Neural Networks Applied to Meteorological Data with NeuralForecast
![](https://i.imgur.com/jYDN7PL.png)

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/cubos-academy/academy-template-readme-projects?color=%2304D361">

  <img alt="Repository size" src="https://img.shields.io/github/repo-size/cubos-academy/academy-template-readme-projects">
  
  <a href="https://github.com/cubos-academy/academy-template-readme-projects/commits/main">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/cubos-academy/academy-template-readme-projects">
  </a>
  
  <!-- <img alt="License" src="https://img.shields.io/badge/license-MIT-brightgreen"> -->
  
   <a href="https://github.com/cubos-academy/academy-template-readme-projects/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/cubos-academy/academy-template-readme-projects?style=social">
  </a>
  
<h4 align="center"> 
	🚧 Forecasting_Models_Irrigation 🚧
</h4>

<p align="center">
	<!--<img alt="Status Em Desenvolvimento" src="https://img.shields.io/badge/STATUS-EM%20DESENVOLVIMENTO-green"> -->
	<img alt="Status Concluído" src="https://img.shields.io/badge/STATUS-CONCLU%C3%8DDO-brightgreen">  
</p>

<p align="center">
 <a href="#-sobre-o-projeto">Sobre</a> •
 <a href="#-funcionalidades">Funcionalidades</a> •
 <a href="#-layout">Layout</a> • 
 <a href="#-como-executar-o-projeto">Como executar</a> • 
 <a href="#-tecnologias">Tecnologias</a> • 
 <a href="#-contribuidores">Contribuidores</a> • 
 <a href="#-autor">Autor</a> • 
 <a href="#user-content--licença">Licença</a>
</p>

## 💻 Sobre o projeto

Com o aumento da demanda por previsões meteorológicas precisas, este projeto explora o uso de redes neurais, especialmente modelos Transformer e RNN, para capturar padrões complexos em séries temporais meteorológicas. Foram avaliados 14 modelos com a biblioteca [NeuralForecast](https://github.com/Nixtla/neuralforecast), utilizando métricas como RMSE, RMSPE, Max Abs Error, Mean Abs Error e Median Abs Error.

A estrutura do projeto é organizada conforme descrito abaixo.

## Estrutura de Diretórios Versão 1

```plaintext
.
├── .venv/                   # Ambiente virtual Python
├── data/                    # Dados brutos e processados
├── img/                     # Imagens geradas durante a execução dos modelos
├── src/                     # Código-fonte principal
│   └── output/              # Scripts e saídas dos modelos
│       ├── 01_model.py                      # Script base do modelo inicial
│       ├── 02_train_model.py                # Script de treinamento do modelo
│       ├── 03_predict_test_data.py          # Previsão com dados de teste
│       ├── 04_mod_Transfor_TFT.py           # Modelo Temporal Fusion Transformer (TFT)
│       ├── 05_mod_RNN_AutoRNN.py            # Modelo RNN com automação
│       ├── 06_mod_RNN_AutoLSTM.py           # Modelo LSTM com automação
│       ├── 07_mod_RNN_AutoGRU.py            # Modelo GRU com automação
│       ├── 08_mod_RNN_AutoTCN.py            # Modelo TCN (Temporal Convolutional Network)
│       ├── 09_mod_RNN_DeepAR.py             # Modelo DeepAR
│       ├── 10_mod_RNN_AutoDilatedRNN.py     # Modelo RNN com dilatação
│       ├── 11_mod_RNN_AutoBiTCN.py          # Modelo BiTCN
│       ├── 12_mod_Transfor_AutoVanilla.py   # Modelo Transformer Vanilla
│       ├── 13_mod_Transfor_AutoInformer.py  # Modelo Informer
│       ├── 14_mod_Transfor_Autoformer.py    # Modelo Autoformer
│       ├── 15_mod_Transfor_AutoFEDformer.py # Modelo FEDformer
│       ├── 16_mod_Transfor_AutoPatchTST.py  # Modelo PatchTST
│       ├── 17_mod_Transfor_AutoTransformer.py # Modelo Transformer genérico
│       ├── app_plot_result_checkpoints.py   # Script para visualização de checkpoints
│       ├── app_plot_result_light.py         # Script para visualização de resultados simplificados
│       └── model_metrics_summary.csv        # Resumo das métricas dos modelos

Esta é a segunda versão do projeto, com melhorias significativas em relação à primeira. A principal diferença é a implementação completa da validação cruzada para todos os modelos, além de outras mudanças estruturais.

## Estrutura de Diretórios da Versão 2

```plaintext
.
├── src_v2/                       # Código-fonte atualizado
│   ├── lightning_logs/           # Logs de treinamento e validação dos modelos
│   └── output/                   # Scripts e saídas dos modelos
│       ├── app4.py               # Modelo 4 atualizado
│       ├── app5.py               # Modelo 5 atualizado
│       ├── app6.py               # Modelo 6 atualizado
│       ├── app7.py               # Modelo 7 atualizado
│       ├── app8.py               # Modelo 8 atualizado
│       ├── app9.py               # Modelo 9 atualizado
│       ├── app10.py              # Modelo 10 atualizado
│       ├── app11.py              # Modelo 11 atualizado
│       ├── app12.py              # Modelo 12 atualizado
│       ├── app13.py              # Modelo 13 atualizado
│       ├── app14.py              # Modelo 14 atualizado
│       ├── app15.py              # Modelo 15 atualizado
│       ├── app16.py              # Modelo 16 atualizado
│       ├── app17.py              # Modelo 17 atualizado
│       ├── app_plot_result_checkpoints.py # Visualização dos checkpoints
│       ├── app_plot_result_light.py       # Visualização simplificada dos resultados
│       ├── appv2-1.py             # Script principal para execução conjunta
│       └── consolidated_model_metrics.csv # Resumo consolidado das métricas
│       ├── myplot_app4.png        # Gráfico gerado para app4
│       ├── myplot_app5.png        # Gráfico gerado para app5
│       ├── myplot_app6.png        # Gráfico gerado para app6
│       ├── myplot_app7.png        # Gráfico gerado para app7
│       ├── myplot_app8.png        # Gráfico gerado para app8
│       ├── myplot_app9.png        # Gráfico gerado para app9
│       ├── myplot_app10.png       # Gráfico gerado para app10
│       ├── myplot_app11.png       # Gráfico gerado para app11
│       ├── myplot_app12.png       # Gráfico gerado para app12

## ⚙️ Funcionalidades
- Avaliação de 14 modelos de redes neurais aplicados à previsão de séries temporais meteorológicas:
  - **Modelos baseados em Transformer:**
    - AutoInformer
    - Autoformer
    - PatchTST
    - FEDformer
    - VanillaTransformer
    - iTransformer
  - **Modelos baseados em RNN:**
    - AutoRNN
    - LSTM
    - GRU
    - AutoTCN
    - AutoDeepAR
    - AutoDilatedRNN
    - AutoBiTCN

- Utilização de dados meteorológicos reais, coletados de estações locais, com medições horárias no período de um ano (2023).
- Comparação de desempenho dos modelos utilizando múltiplas métricas de erro:
  - RMSE (Root Mean Squared Error)
  - RMSPE (Root Mean Squared Percentage Error)
  - Max Abs Error (Erro Máximo Absoluto)
  - Mean Abs Error (Erro Médio Absoluto)
  - Median Abs Error (Erro Absoluto Mediano)

- Configuração e ajuste automático de hiperparâmetros:
  - Utilização de validação cruzada e técnica de *early stopping*.
  - Suporte a variáveis exógenas para melhorar a precisão preditiva.

- Implementação de previsões de curto, médio e longo prazo:
  - Previsão recursiva (utilizando previsões anteriores como entradas).
  - Previsão direta (geração de todos os passos do horizonte de uma única vez).

- Visualização e análise dos resultados:
  - Gráficos comparativos de métricas de desempenho entre os modelos.
  - Análise detalhada das previsões e padrões sazonais capturados pelos modelos.

- Armazenamento e organização dos resultados:
  - Modelos treinados e logs de validação armazenados nos diretórios `checkpoints` e `lightning_logs`.
  - Código e resultados disponibilizados no repositório GitHub para transparência e reprodutibilidade.


## 🛣️ Como executar o projeto

### Pré-requisitos

Antes de começar, você precisa ter instalado:
- [Python 3.13](https://www.python.org/downloads/)
- Bibliotecas específicas: Veja o arquivo `requirements.txt` no repositório.

### Execução

1. Clone este repositório:
   ```bash
   git clone https://github.com/rps-ifpr/Forecasting_Models_Irrigation.git

## 🛠 Tecnologias

O projeto foi desenvolvido utilizando as seguintes tecnologias e ferramentas:

- **Linguagem:** Python 3.13
- **Bibliotecas:**
  - **[NeuralForecast](https://github.com/Nixtla/neuralforecast):** Ferramenta avançada para previsão de séries temporais com suporte a modelos baseados em Redes Neurais (Transformers e RNNs).
  - **[PyTorch](https://pytorch.org/):** Framework de aprendizado de máquina, utilizado para implementar e treinar os modelos.
  - **[Pandas](https://pandas.pydata.org/):** Biblioteca para manipulação e análise de dados.
  - **[Matplotlib](https://matplotlib.org/):** Biblioteca para visualização de dados e geração de gráficos comparativos.
- **Recursos adicionais:**
  - **Configuração automática de hiperparâmetros:** utilizando técnicas como `early stopping` e validação cruzada.
  - **Suporte a variáveis exógenas:** integração de dados externos para enriquecer as previsões.

## 🧑‍💻 Autor

Este projeto foi desenvolvido por **Rogério Pereira dos Santos**, pesquisador e desenvolvedor com foco em redes neurais aplicadas à previsão de séries temporais meteorológicas.

- **Instituição:** Instituto Federal do Paraná (IFPR)
- **Contato:**
  - [LinkedIn](https://www.linkedin.com/in/rogerio-dosantos) — Conecte-se para discutir sobre redes neurais e projetos de previsão climática.
  - [Email](mailto:rogerio.dosantos@ifpr.edu.br) — Para dúvidas ou colaborações relacionadas ao projeto.
- **Publicações e Contribuições:**
  - Publicações acadêmicas em previsão climática e machine learning.
  - Experiência com tecnologias aplicadas à agricultura de precisão e sustentabilidade.

Sinta-se à vontade para entrar em contato ou explorar os demais projetos no [GitHub](https://github.com/rps-ifpr).

## 💪 Como contribuir para o projeto
1. Faça um **fork** do projeto.
2. Crie uma nova branch com as suas alterações: `git checkout -b my-feature`
3. Salve as alterações e crie uma mensagem de commit contando o que você fez: `git commit -m "feature: My new feature"`
4. Envie as suas alterações: `git push origin my-feature`
> Caso tenha alguma dúvida confira este [guia de como contribuir no GitHub](./CONTRIBUTING.md)


## 📝 Licença
<!-- Este projeto esta sobe a licença [MIT](./LICENSE). -->


