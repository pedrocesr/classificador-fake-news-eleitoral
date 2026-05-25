import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.utils import resample

# 1: Preparar os dados

print("Preparando dataset...")

df = pd.read_csv(
    'urna_eletrônica_fraude_Bolsonaro_Lula.csv'
)[['Claim', 'Verdict']]

df.columns = ['Texto', 'Label']

df = df.dropna()

# 2. PADRONIZAÇÃO

mapa = {

    # FALSOS
    'Falso': 'Falso',
    'falso': 'Falso',
    'Enganoso': 'Falso',
    'Sem contexto': 'Falso',
    'não_é_bem_assim': 'Falso',
    'Parcialmente falso': 'Falso',
    'Impreciso': 'Falso',

    # VERDADEIROS
    'Verdadeiro': 'Verdadeiro',
    'Verdade': 'Verdadeiro',
    'Verdadeiro, mas': 'Verdadeiro'
}

df['Label'] = (
    df['Label']
    .astype(str)
    .str.strip()
    .map(mapa)
)

df = df.dropna()

# 3. FRASES MANUAIS

extras = pd.DataFrame([

    # VERDADEIROS
    ["TSE confirma início da votação em todo o Brasil às 8h", "Verdadeiro"],
    ["Apuração das urnas eletrônicas acontece em tempo real", "Verdadeiro"],
    ["Eleitor pode usar o e-Título para justificar ausência", "Verdadeiro"],
    ["Resultado oficial divulgado pelo TSE", "Verdadeiro"],
    ["Urnas eletrônicas passam por testes de segurança", "Verdadeiro"],
    ["O voto é obrigatório para maiores de 18 anos", "Verdadeiro"],
    ["A Justiça Eleitoral divulga os resultados oficiais", "Verdadeiro"],

    # FALSOS
    ["Urnas eletrônicas foram fraudadas", "Falso"],
    ["Hackers invadiram as urnas eletrônicas", "Falso"],
    ["O TSE manipulou o resultado das eleições", "Falso"],
    ["Bolsonaro venceu com 73 por cento dos votos", "Falso"],
    ["Lula roubou milhões de votos", "Falso"],
    ["Os votos foram alterados nas urnas eletrônicas", "Falso"]

], columns=['Texto', 'Label'])

df = pd.concat([df, extras], ignore_index=True)

# 4. BALANCEAMENTO

falsos = df[df['Label'] == 'Falso']
verdadeiros = df[df['Label'] == 'Verdadeiro']

if len(verdadeiros) < len(falsos):

    verdadeiros = resample(
        verdadeiros,
        replace=True,
        n_samples=len(falsos),
        random_state=42
    )

elif len(falsos) < len(verdadeiros):

    falsos = resample(
        falsos,
        replace=True,
        n_samples=len(verdadeiros),
        random_state=42
    )

df = pd.concat([falsos, verdadeiros])

print("\nDistribuição balanceada:")
print(df['Label'].value_counts())

# 5. TREINAMENTO

X_train, X_test, y_train, y_test = train_test_split(
    df['Texto'],
    df['Label'],
    test_size=0.2,
    random_state=42,
    stratify=df['Label']
)

# 6. TF-IDF

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. MODELO

modelo = MultinomialNB()

modelo.fit(X_train_vec, y_train)

# 8. MÉTRICAS

predicoes = modelo.predict(X_test_vec)

print("\n--- RESULTADOS ---")

print(f"\nAccuracy: {accuracy_score(y_test, predicoes):.2f}")

print("\nRelatório:")
print(classification_report(y_test, predicoes))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, predicoes))

# 9. TESTE

print("\n--- TESTE DE CLASSIFICAÇÃO ---")

while True:

    frase = input("\nDigite uma frase (ou 'sair'): ")

    if frase.lower() == 'sair':
        break

    frase_vec = vectorizer.transform([frase])

    resultado = modelo.predict(frase_vec)

    print(f"\nClassificação: {resultado[0]}")
