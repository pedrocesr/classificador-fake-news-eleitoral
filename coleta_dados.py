from factcheckexplorer.factcheckexplorer import FactCheckLib

# COLETA DE DADOS

fact_check = FactCheckLib(
    query="eleição Bolsonaro Lula urna fraude voto TSE campanha",
    language="pt",
    num_results=1000
)

print("Iniciando coleta de dados...")

fact_check.process()

print("Processo concluído!")
print("Arquivo CSV gerado com sucesso.")
