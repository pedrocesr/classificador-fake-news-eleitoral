# Classificador de Fake News Eleitoral

Projeto desenvolvido para a disciplina de Machine Learning com o objetivo de criar um classificador de Fake News utilizando aprendizado supervisionado.

## Integrantes

- Erica Araújo
- Nalbert Santana
- Mickael Cedraz
- Monyc Luisa
- Pedro César
- Tiago Sângil

---

## Objetivo

Desenvolver um sistema capaz de classificar notícias e afirmações relacionadas ao contexto eleitoral brasileiro como:

- Falso
- Verdadeiro

---

## Coleta de Dados

A coleta de dados foi realizada utilizando a biblioteca:

[FactCheckExplorer no GitHub](https://github.com/GONZOsint/factcheckexplorer?utm_source=chatgpt.com)

Foram utilizadas palavras-chave relacionadas ao contexto eleitoral, como:

- Bolsonaro
- Lula
- urna eletrônica
- fraude eleitoral
- voto
- eleição

Os dados coletados foram armazenados em um arquivo CSV contendo textos e veredictos das notícias.

---

## Preparação do Dataset

O dataset foi organizado utilizando as colunas:

| Coluna | Descrição |
|---|---|
| Texto | Conteúdo da notícia |
| Label | Classificação da notícia |

Os veredictos foram padronizados para duas classes:

- Falso
- Verdadeiro

Também foram adicionadas manualmente algumas notícias verdadeiras para melhorar o balanceamento do modelo.

---

## Algoritmo Utilizado

O algoritmo utilizado foi:

- Naive Bayes


## Treinamento do Modelo

Etapas realizadas:

1. Separação dos dados em treino e teste
2. Vetorização dos textos utilizando TF-IDF
3. Treinamento do modelo
4. Avaliação das métricas

---

## Métricas Avaliadas

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de Confusão

---

## Estrutura do Projeto

```text
Machine Learning/
│
├── coleta_dados.py
├── classificador_fake_news.py
├── urna_eletrônica_fraude_Bolsonaro_Lula.csv
└── README.md
```

---

## Como Executar

### Instalar dependências

```bash
pip install pandas scikit-learn
pip install git+https://github.com/GONZOsint/factcheckexplorer.git
```

### Executar o projeto

```bash
python classificador_fake_news.py
```

---

## Demonstração

O sistema permite que o usuário digite uma frase relacionada às eleições para que o modelo classifique automaticamente como:

- Falso
- Verdadeiro

Exemplo:

```text
Digite uma frase sobre as eleições para testar:

"Urnas eletrônicas foram fraudadas"

Resultado:
Falso
```

---

## Conclusão

O projeto aplicou conceitos de aprendizado supervisionado e processamento de linguagem natural para classificação automática de Fake News relacionadas ao contexto eleitoral brasileiro.
