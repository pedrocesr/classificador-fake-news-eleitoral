import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1: Preparar os dados
print("Limpando e preparando o dataset...")

# Carregando o arquivo gerado pela coleta
df_bruto = pd.read_csv('urna_eletrônica_fraude_Bolsonaro_Lula.csv')

# Seleciona e renomeia para o padrão exigido Texto e Label
df = df_bruto[['Claim', 'Verdict']].copy()
df.columns = ['Texto', 'Label']

# Padroniza as Labels para Falso ou Verdadeiro
mapa = {
    'Falso': 'Falso',
    'falso': 'Falso',
    'Enganoso': 'Falso',
    'Sem contexto': 'Falso',
    'não_é_bem_assim': 'Falso'
}

df['Label'] = df['Label'].map(mapa).fillna('Verdadeiro')

# Lista de notícias verdadeiras
reais = pd.DataFrame([
    {"Texto": "TSE confirma início da votação em todo o Brasil às 8h", "Label": "Verdadeiro"},
    {"Texto": "Apuração das urnas eletrônicas acontece em tempo real", "Label": "Verdadeiro"},
    {"Texto": "Eleitor pode usar o e-Título para justificar ausência", "Label": "Verdadeiro"},
    {"Texto": "O Tribunal Superior Eleitoral confirma que o voto é obrigatório para brasileiros entre 18 e 70 anos.", "Label": "Verdadeiro"},
    {"Texto": "As seções eleitorais em todo o Brasil funcionam das 8h às 17h no dia da eleição.", "Label": "Verdadeiro"},
    {"Texto": "O transporte de armas e munições por colecionadores e caçadores é proibido no dia da votação.", "Label": "Verdadeiro"},
    {"Texto": "O teste de integridade das urnas é realizado em todas as eleições na presença de entidades fiscalizadoras.", "Label": "Verdadeiro"},
    {"Texto": "Para votar, o cidadão deve apresentar um documento oficial com foto, como RG ou CNH.", "Label": "Verdadeiro"},
    {"Texto": "O site do Tribunal Superior Eleitoral disponibiliza o local de votação para consulta antecipada.", "Label": "Verdadeiro"},
    {"Texto": "A Justiça Eleitoral oferece assistência especial para eleitores com deficiência ou mobilidade reduzida.", "Label": "Verdadeiro"},
    {"Texto": "O resultado oficial das eleições é proclamado apenas pela Justiça Eleitoral após o encerramento da apuração.", "Label": "Verdadeiro"}
])

df = pd.concat([df, reais], ignore_index=True).dropna()

# 2: Treinamento
X = df['Texto']
y = df['Label']

# Divide treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Transforma texto em números utilizando TF-IDF
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Treina o modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X_train_vec, y_train)

# 3: Métricas
predicoes = modelo.predict(X_test_vec)

print("\n--- RESULTADOS DO MODELO ---")
print(f"Accuracy: {accuracy_score(y_test, predicoes)}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, predicoes, zero_division=0))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, predicoes))

# 4: Classificação de novas frases
def classificar_noticia(frase):
    vec = vectorizer.transform([frase])
    resultado = modelo.predict(vec)
    return resultado[0]

print("\n--- TESTE DE CLASSIFICAÇÃO ---")

teste = input("Digite uma frase sobre as eleições para testar: ")

print(f"O modelo classificou como: {classificar_noticia(teste)}")