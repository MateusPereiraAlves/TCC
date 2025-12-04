# UniVAD para Controle de Qualidade Industrial: Detecção de Anomalias em Caixas de Papelão

Este repositório contém uma implementação adaptada do **UniVAD (Unified Model for Few-shot Visual Anomaly Detection)** aplicada a um cenário industrial real: o controle de qualidade em linhas de produção de caixas de papelão.

Este projeto foi desenvolvido como parte da disciplina de **Projetos Empreendedores B** do curso de Ciência de Dados e Inteligência Artificial na **PUC-Campinas**, com apoio da empresa **Ondupress Embalagens**. O trabalho valida o uso de Modelos de Fundação (Grounding DINO, SAM, DINOv2, CLIP) para detecção de anomalias *training-free* e *few-shot* em ambientes não controlados.

## 🏭 Visão Geral do Projeto

Os benchmarks padrões de detecção de anomalias (como MVTec-AD e VisA) geralmente apresentam objetos centralizados e fundos controlados, o que não reflete a complexidade do chão de fábrica. Este projeto ataca justamente esse desafio.

* **Objetivo:** Detectar defeitos de fabricação em caixas de papelão usando apenas imagens de referência (*Few-Shot*).
* **Arquitetura:** Baseada no UniVAD, integrando múltiplos modelos pré-treinados em larga escala.
* **Resultado Principal:** Acurácia média de **86,9%** e AUC média de **0,94** utilizando prompts otimizados e modelos destilados (mais leves).

## 📂 Dataset

Para validar o modelo em condições realistas, desenvolvemos e publicamos o conjunto de dados **Cardboard Box Anomaly Detection**. Ele contém 553 imagens capturadas em ambiente industrial real (chão de fábrica e esteiras), com variações de ângulo e iluminação.

O dataset está hospedado no Hugging Face:
[**Gabriel8/cardboard-box-anomaly-detection**](https://huggingface.co/datasets/Gabriel8/cardboard-box-anomaly-detection)

Para utilizá-lo neste projeto:
```python
from datasets import load_dataset
# Faça login usando `huggingface-cli login` para acessar o dataset
ds = load_dataset("Gabriel8/cardboard-box-anomaly-detection")
````

## 🛠️ Modificações e Arquivos do Projeto

Este repositório segue a organização do UniVAD original, mas inclui refatorações significativas para suportar a simulação de inferência em tempo real e execução em ambientes com recursos limitados (como o Google Colab).

Os seguintes arquivos foram modificados ou adicionados em relação ao projeto original:

  * **`UniVAD.ipynb` (Adicionado):** Notebook principal contendo todas as instruções de instalação, configuração de ambiente e execução do modelo passo a passo.
  * **`UniVAD.py` (Modificado):** Refatorado para integrar o pipeline de inferência.
  * **`test_univad.py` (Modificado):** Substitui os antigos scripts `test_univad.py` e `segment_components.py`, unificando a segmentação e a avaliação em um fluxo contínuo.
  * **`models/component_segmentation.py` (Modificado):** Implementação de gerenciamento de memória otimizado, carregando e descarregando modelos pesados da VRAM.
  * **`configs/class_histogram/cardboard_box.yaml` (Adicionado):** Arquivo de configuração específico para a classe de caixas de papelão.
  * **`datasets/cardboard_box.py` (Adicionado):** Script para carregamento e formatação do dataset personalizado.

## 🚀 Como Usar

### 1\. Pré-requisitos

O projeto depende de versões específicas de bibliotecas para garantir a compatibilidade entre os modelos de fundação:

  * Python 3.10+
  * `transformers==4.44.2`
  * `tokenizers==0.19.1`

### 2\. Executando o Modelo

O ponto de entrada recomendado é o Jupyter Notebook:

> **`UniVAD.ipynb`**

Este notebook guia todo o processo, desde a instalação das dependências até a geração dos mapas de anomalia.

## 📊 Resultados

O modelo foi avaliado no dataset industrial utilizando uma configuração **1-shot** (1 imagem de referência normal) e modelos otimizados (SAM-B + DINOv2-L).

| Métrica | Pontuação |
| :--- | :--- |
| **AUC** | **94,29%** |
| **Acurácia** | **86,98%** | 
| **Tempo por Imagem** | \~3,0s | 

## 👥 Autores

  * Gabriel de Antonio Mazetto
  * Felipe de Oliveira Santos
  * Gustavo Barbosa Silva
  * Lucas Mauad Sant' Anna
  * Mateus Pereira Alves

**Orientador:** Prof. Me. Fernando Soares de Aguiar Neto

**Instituição:** Pontifícia Universidade Católica de Campinas (PUC-Campinas)
