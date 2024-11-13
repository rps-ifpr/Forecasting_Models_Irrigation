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

   
#### 🧭 Rodando a aplicação web (Frontend)

```bash

# Clone este repositório
$ git clone git@github.com:cubos-academy/academy-template-readme-projects.git

# Acesse a pasta do projeto no seu terminal/cmd
$ cd academy-template-readme-projects

# Vá para a pasta da aplicação Front End
$ cd web

# Instale as dependências
$ npm install

# Execute a aplicação em modo de desenvolvimento
$ npm run start

# A aplicação será aberta na porta:3000 - acesse http://localhost:3000

```

---

## 🛠 Tecnologias

As seguintes ferramentas foram usadas na construção do projeto:

#### **Website**  ([React](https://reactjs.org/)  +  [TypeScript](https://www.typescriptlang.org/))

-   **[React Icons](https://react-icons.github.io/react-icons/)**
-   **[Axios](https://github.com/axios/axios)**

> Veja o arquivo  [package.json](https://github.com/cubos-academy/academy-template-readme-projects/blob/master/web/package.json)

#### [](https://github.com/cubos-academy/academy-template-readme-projects#server-nodejs--typescript)**Server**  ([NodeJS](https://nodejs.org/en/)  +  [TypeScript](https://www.typescriptlang.org/))

-   **[Express](https://expressjs.com/)**
-   **[CORS](https://expressjs.com/en/resources/middleware/cors.html)**
-   **[KnexJS](http://knexjs.org/)**
-   **[SQLite](https://github.com/mapbox/node-sqlite3)**
-   **[ts-node](https://github.com/TypeStrong/ts-node)**
-   **[dotENV](https://github.com/motdotla/dotenv)**

> Veja o arquivo  [package.json](https://github.com/cubos-academy/academy-template-readme-projects/blob/master/web/package.json)

#### [](https://github.com/cubos-academy/academy-template-readme-projects#mobile-flutter--typescript)**Mobile**  ([Flutter](https://flutter.dev/)  +  [TypeScript](https://www.typescriptlang.org/))

-   **[Flutter](https://flutter.dev/)**

> Veja o arquivo  [package.json](https://github.com/cubos-academy/academy-template-readme-projects/blob/master/mobile/package.json)

#### [](https://github.com/cubos-academy/academy-template-readme-projects#utilit%C3%A1rios)**Utilitários**

-   Protótipo:  **[Figma](https://www.figma.com/)**  →  **[Protótipo](https://www.figma.com/file/L4O2dxZzKKhEPspDgxzZ3a/Template-de-Capa-para-Projetos?type=design&node-id=0%3A1&t=zpQ8tnlNEfQtMeC8-1)**
-   API:  **[API](Link da API)**
-   Editor:  **[Visual Studio Code](https://code.visualstudio.com/)**  → Extensions:  **[SQLite](https://marketplace.visualstudio.com/items?itemName=alexcvzz.vscode-sqlite)**
-   Markdown:  **[StackEdit](https://stackedit.io/)**,  **[Markdown Emoji](https://gist.github.com/rxaviers/7360908)**
-   Commit Conventional:  **[Commitlint](https://github.com/conventional-changelog/commitlint)**
-   Teste de API:  **[Insomnia](https://insomnia.rest/)**
-   Ícones:  **[Feather Icons](https://feathericons.com/)**,  **[Font Awesome](https://fontawesome.com/)**
-   Fontes:  **[Ubuntu](https://fonts.google.com/specimen/Ubuntu)**,  **[Roboto](https://fonts.google.com/specimen/Roboto)**


---

## 👨‍💻 Contribuidores

Um praise para os cúbicos que contribuíram neste projeto 👏

<table>
  <tr>
    <td align="center"><a href="https://cubos.academy/"><img style="border-radius: 50%;" src="https://ca.slack-edge.com/T02BJRAJH6G-U02BMJ98N68-5e47f31c2a79-512" width="100px;" alt=""/><br /><sub><b>Guido Cerqueira</b></sub></a><br /><a href="https://cubos.academy/" title="Cubos Academy">👨‍💻</a></td>
    <td align="center"><a href="https://cubos.academy/"><img style="border-radius: 50%;" src="https://media.licdn.com/dms/image/D4E03AQG_0i4C04YeYg/profile-displayphoto-shrink_200_200/0/1677773908684?e=1688601600&v=beta&t=D1aefI0DMhoc7NZFvKUMn_LAIGEyEczyFaxUz0Auh6o" width="100px;" alt=""/><br /><sub><b>Daniel Lopes</b></sub></a><br /><a href="https://cubos.academy/" title="Cubos Academy">👨‍💻</a></td>
    <td align="center"><a href="https://cubos.academy/"><img style="border-radius: 50%;" src="https://media.licdn.com/dms/image/C4E03AQG1l_n_4-Bhsg/profile-displayphoto-shrink_200_200/0/1516624005627?e=1688601600&v=beta&t=5nA_EezcbJ068eNefrWeccM-FBUUBnmKWQ7frFCxG9U" width="100px;" alt=""/><br /><sub><b>Guilherme Bernal</b></sub></a><br /><a href="https://cubos.academy/" title="Cubos Academy">👨‍💻</a></td>
	 <td align="center"><a href="https://cubos.academy/"><img style="border-radius: 50%;" src="https://media.licdn.com/dms/image/C4E03AQGp3BsgWtthBg/profile-displayphoto-shrink_200_200/0/1643505110642?e=1688601600&v=beta&t=c_h3BkUr6POLelref_Nzc6AqzJpWTgENueNs9KqmvRM" width="100px;" alt=""/><br /><sub><b>Clara Battesini</b></sub></a><br /><a href="https://cubos.academy/" title="Cubos Academy">👩‍💻</a></td>
    
    
  </tr>
</table>

## 💪 Como contribuir para o projeto

1. Faça um **fork** do projeto.
2. Crie uma nova branch com as suas alterações: `git checkout -b my-feature`
3. Salve as alterações e crie uma mensagem de commit contando o que você fez: `git commit -m "feature: My new feature"`
4. Envie as suas alterações: `git push origin my-feature`
> Caso tenha alguma dúvida confira este [guia de como contribuir no GitHub](./CONTRIBUTING.md)

---

## 🧙‍♂️ Autor

<a href="https://www.figma.com/@caiux">
 <img style="border-radius: 50%;" src="https://media.licdn.com/dms/image/D4D03AQEDfulqSVXZqw/profile-displayphoto-shrink_200_200/0/1674667231041?e=1688601600&v=beta&t=C-f9fp3xJDwXm1u4c6eMwpWfVIyW0eCTDAKGIyNdRJA" width="100px;" alt=""/>
 <br />
 <sub><b>Caio Lopes</b></sub></a> <a href="https://www.figma.com/@caiux" title="Cubos Academy">✨</a>
 <br />

---

## 📝 Licença

<!-- Este projeto esta sobe a licença [MIT](./LICENSE). -->

Feito com ❤️ por Caio Lopes 👋🏽 [Entre em contato!](https://www.linkedin.com/in/caiovslopes/)

