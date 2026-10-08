import sqlite3
import json
from datetime import datetime

def analyze_decision_logs():
    """Analisa os logs de decisões para identificar problemas nas predições"""
    
    conn = sqlite3.connect('detailed_decisions.db')
    cursor = conn.cursor()
    
    print("=" * 80)
    print("ANÁLISE DETALHADA DOS LOGS DE DECISÕES")
    print("=" * 80)
    
    # 1. Análise geral das ações
    print("\n1. DISTRIBUIÇÃO DE AÇÕES:")
    cursor.execute("""
        SELECT action, COUNT(*) as count, 
               AVG(confidence) as avg_confidence
        FROM detailed_decisions 
        GROUP BY action 
        ORDER BY count DESC
    """)
    
    for row in cursor.fetchall():
        action, count, avg_conf = row
        print(f"   {action}: {count} ocorrências (confiança média: {avg_conf:.3f})")
    
    # 2. Análise de NO_ACTION com detalhes
    print("\n2. ANÁLISE DE NO_ACTION (últimas 10):")
    cursor.execute("""
        SELECT timestamp, symbol, reasoning_factors, technical_analysis, confidence
        FROM detailed_decisions 
        WHERE action = 'NO_ACTION' 
        ORDER BY timestamp DESC 
        LIMIT 10
    """)
    
    no_action_records = cursor.fetchall()
    
    for i, record in enumerate(no_action_records, 1):
        timestamp, symbol, reasoning, tech_analysis, confidence = record
        print(f"\n   [{i}] {timestamp} - {symbol} (Confiança: {confidence:.3f})")
        
        # Parse reasoning factors
        try:
            reasons = json.loads(reasoning) if reasoning else []
            print(f"       Motivos: {reasons}")
        except:
            print(f"       Motivos (raw): {reasoning}")
        
        # Parse technical analysis
        try:
            tech = json.loads(tech_analysis) if tech_analysis else {}
            if isinstance(tech, dict) and 'error' in str(tech).lower():
                print(f"       ⚠️ ERRO TÉCNICO: {tech}")
        except:
            if tech_analysis and 'error' in tech_analysis.lower():
                print(f"       ⚠️ ERRO TÉCNICO (raw): {tech_analysis}")
    
    # 3. Análise de erros específicos
    print("\n3. ANÁLISE DE ERROS NOS MODELOS:")
    cursor.execute("""
        SELECT technical_analysis, COUNT(*) as count
        FROM detailed_decisions 
        WHERE technical_analysis LIKE '%error%' 
        GROUP BY technical_analysis
    """)
    
    error_records = cursor.fetchall()
    
    if error_records:
        for tech_analysis, count in error_records:
            print(f"\n   Erro encontrado ({count} ocorrências):")
            try:
                error_data = json.loads(tech_analysis)
                for model, error_info in error_data.items():
                    if isinstance(error_info, dict) and 'error' in error_info:
                        print(f"     - {model}: {error_info['error']}")
            except:
                print(f"     Raw: {tech_analysis[:200]}...")
    else:
        print("   Nenhum erro explícito encontrado nos logs técnicos.")
    
    # 4. Análise de confiança baixa
    print("\n4. ANÁLISE DE CONFIANÇA BAIXA:")
    cursor.execute("""
        SELECT symbol, AVG(confidence) as avg_conf, COUNT(*) as count
        FROM detailed_decisions 
        WHERE confidence < 0.65
        GROUP BY symbol
        ORDER BY avg_conf ASC
    """)
    
    low_conf_records = cursor.fetchall()
    
    if low_conf_records:
        for symbol, avg_conf, count in low_conf_records:
            print(f"   {symbol}: {count} decisões com confiança média {avg_conf:.3f}")
    else:
        print("   Nenhuma decisão com confiança baixa encontrada.")
    
    # 5. Análise temporal
    print("\n5. ANÁLISE TEMPORAL (últimas 24h):")
    cursor.execute("""
        SELECT DATE(timestamp) as date, action, COUNT(*) as count
        FROM detailed_decisions 
        WHERE timestamp >= datetime('now', '-1 day')
        GROUP BY DATE(timestamp), action
        ORDER BY date DESC, count DESC
    """)
    
    temporal_records = cursor.fetchall()
    
    if temporal_records:
        for date, action, count in temporal_records:
            print(f"   {date}: {action} = {count} ocorrências")
    else:
        print("   Nenhuma atividade nas últimas 24h.")
    
    # 6. Análise de spreads altos
    print("\n6. ANÁLISE DE PROBLEMAS DE SPREAD:")
    cursor.execute("""
        SELECT reasoning_factors, COUNT(*) as count
        FROM detailed_decisions 
        WHERE reasoning_factors LIKE '%spread%' OR reasoning_factors LIKE '%Spread%'
        GROUP BY reasoning_factors
    """)
    
    spread_records = cursor.fetchall()
    
    if spread_records:
        for reasoning, count in spread_records:
            print(f"   Problema de spread ({count} ocorrências):")
            try:
                reasons = json.loads(reasoning)
                for reason in reasons:
                    if 'spread' in reason.lower():
                        print(f"     - {reason}")
            except:
                print(f"     Raw: {reasoning}")
    else:
        print("   Nenhum problema de spread identificado.")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("RESUMO DA ANÁLISE CONCLUÍDA")
    print("=" * 80)

if __name__ == "__main__":
    analyze_decision_logs()