#!/usr/bin/env python3
import os
from detailed_decision_logger import DetailedDecisionLogger

# Remover banco de teste se existir
test_db = "test_init.db"
if os.path.exists(test_db):
    os.remove(test_db)

print("Inicializando DetailedDecisionLogger...")
try:
    logger = DetailedDecisionLogger(test_db)
    print("Logger inicializado com sucesso")
except Exception as e:
    print(f"Erro ao inicializar logger: {e}")
    import traceback
    traceback.print_exc()

# Verificar se o banco foi criado
import sqlite3
if os.path.exists(test_db):
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    
    # Listar tabelas
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print('Tables:', tables)
    
    # Verificar colunas da tabela detailed_decisions
    cursor.execute('PRAGMA table_info(detailed_decisions)')
    columns = cursor.fetchall()
    print('Columns:', [col[1] for col in columns])
    
    conn.close()
else:
    print("Banco de dados não foi criado")