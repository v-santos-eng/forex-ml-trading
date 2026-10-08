import sqlite3

def check_database():
    try:
        conn = sqlite3.connect('production_monitoring.db')
        cursor = conn.cursor()
        
        # Verificar tabelas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print('Tabelas:', [table[0] for table in tables])
        
        # Contar registros
        cursor.execute('SELECT COUNT(*) FROM system_metrics;')
        count_sys = cursor.fetchone()[0]
        print(f'Registros system_metrics: {count_sys}')
        
        cursor.execute('SELECT COUNT(*) FROM process_metrics;')
        count_proc = cursor.fetchone()[0]
        print(f'Registros process_metrics: {count_proc}')
        
        cursor.execute('SELECT COUNT(*) FROM alerts;')
        count_alerts = cursor.fetchone()[0]
        print(f'Registros alerts: {count_alerts}')
        
        # Mostrar últimas métricas do sistema
        cursor.execute('SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT 3;')
        recent_sys = cursor.fetchall()
        print('\nÚltimas métricas do sistema:')
        for row in recent_sys:
            print(f'  {row}')
        
        # Mostrar últimas métricas de processo
        cursor.execute('SELECT * FROM process_metrics ORDER BY timestamp DESC LIMIT 3;')
        recent_proc = cursor.fetchall()
        print('\nÚltimas métricas de processo:')
        for row in recent_proc:
            print(f'  {row}')
        
        conn.close()
        
    except Exception as e:
        print(f'Erro ao verificar database: {e}')

if __name__ == '__main__':
    check_database()