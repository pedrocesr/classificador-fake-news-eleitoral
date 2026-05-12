from factcheckexplorer.factcheckexplorer import FactCheckLib

fact_check = FactCheckLib(
    query="urna eletrônica fraude Bolsonaro Lula",
    language="pt", 
    num_results=500
)

print("Iniciando coleta de dados...")
fact_check.process()
print("Processo concluído! Verifique o arquivo CSV na sua pasta.")