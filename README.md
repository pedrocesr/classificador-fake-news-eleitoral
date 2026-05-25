# Classificador de Fake News Eleitoral

Projeto desenvolvido para a disciplina de Machine Learning com o objetivo de criar um classificador de Fake News utilizando aprendizado supervisionado.

---

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

O projeto utiliza técnicas de Processamento de Linguagem Natural (PLN) e aprendizado supervisionado para identificar padrões em textos relacionados ao contexto eleitoral.

---

## Coleta de Dados

A coleta de dados foi realizada utilizando a biblioteca:

:contentReference[oaicite:0]{index=0}

Foram utilizadas palavras-chave relacionadas ao contexto eleitoral, como:

- Bolsonaro
- Lula
- urna eletrônica
- fraude eleitoral
- voto
- eleição
- TSE
- campanha

Os dados coletados foram armazenados em um arquivo CSV contendo:

- texto da notícia;
- fonte;
- veredito da checagem;
- data de publicação.

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

Exemplos de categorias convertidas para “Falso”:

- Enganoso
- Sem contexto
- Parcialmente falso
- Impreciso

Também foram adicionados manualmente exemplos de notícias verdadeiras e falsas para melhorar o balanceamento e o aprendizado do modelo.

---

## Algoritmo Utilizado

O algoritmo utilizado foi:

- Naive Bayes (MultinomialNB)

O Naive Bayes foi escolhido por apresentar bom desempenho em classificação de textos, além de possuir baixo custo computacional e boa eficiência em tarefas de Processamento de Linguagem Natural.

---

## Treinamento do Modelo

Etapas realizadas:

1. Coleta e preparação do dataset
2. Separação dos dados em treino e teste
3. Vetorização dos textos utilizando TF-IDF
4. Utilização de unigramas e bigramas (`ngram_range=(1,2)`)
5. Treinamento do modelo Naive Bayes
6. Balanceamento das classes para evitar enviesamento do modelo
7. Avaliação das métricas

---

## Métricas Avaliadas

As seguintes métricas foram utilizadas para avaliar o desempenho do modelo:

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de Confusão

Essas métricas permitiram analisar a capacidade do modelo em identificar corretamente notícias verdadeiras e falsas.

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
