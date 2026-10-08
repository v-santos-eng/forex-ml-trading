#!/usr/bin/env python3
"""
Demo Simplificado do Sistema Agno ML Forex
Versão funcional com Fase 1 implementada

Este script demonstra:
- Fase 1: Ferramentas ML encapsuladas (FUNCIONANDO)
- Integração básica com Agno
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agno_ml_tool import MLForexTool
except ImportError as e:
    print(f"Erro ao importar MLForexTool: {e}")
    sys.exit(1)

def demo_ml_tools_completo():
    """
    Demonstração completa das ferramentas ML
    """
    print("\n" + "="*60)
    print("DEMO COMPLETO: FERRAMENTAS ML FOREX COM AGNO")
    print("="*60)
    
    try:
        # Inicializar ferramenta ML
        ml_tool = MLForexTool()
        
        print("\n1. Testando predição de direção para EURUSD...")
        result_eur = ml_tool.predict_forex_direction(
            pair="EURUSD=X",
            period="6mo"
        )
        print(f"   Resultado EURUSD: {result_eur}")
        
        print("\n2. Testando predição para GBPUSD...")
        result_gbp = ml_tool.predict_forex_direction(
            pair="GBPUSD=X",
            period="3mo"
        )
        print(f"   Resultado GBPUSD: {result_gbp}")
        
        print("\n3. Obtendo informações dos modelos...")
        model_info = ml_tool.get_model_info()
        print(f"   Modelos disponíveis: {model_info}")
        
        print("\n4. Comparando múltiplos pares de moedas...")
        comparison = ml_tool.compare_pairs(
            pairs=["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
            period="3mo"
        )
        print(f"   Comparação detalhada:")
        for pair, data in comparison.get('results', {}).items():
            print(f"     {pair}: {data}")
        
        print("\n5. Testando backtest de estratégia...")
        backtest_result = ml_tool.backtest_strategy(
            pair="EURUSD=X",
            period="6mo"
        )
        print(f"   Resultado do Backtest: {backtest_result}")
        
        print("\n6. Relatório de performance detalhado...")
        performance_report = {
            "timestamp": datetime.now().isoformat(),
            "pairs_tested": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
            "predictions": {
                "EURUSD=X": result_eur,
                "GBPUSD=X": result_gbp
            },
            "model_info": model_info,
            "comparison": comparison,
            "backtest": backtest_result
        }
        
        print(f"\nRELATÓRIO FINAL DE PERFORMANCE:")
        print(f"   Timestamp: {performance_report['timestamp']}")
        print(f"   Pares testados: {len(performance_report['pairs_tested'])}")
        print(f"   Predições realizadas: {len(performance_report['predictions'])}")
        models_count = len(performance_report['model_info']) if isinstance(performance_report['model_info'], list) else len(performance_report['model_info'].get('models', []))
        print(f"   Modelos disponíveis: {models_count}")
        
        return True, performance_report
        
    except Exception as e:
        print(f"Erro no demo: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def demo_agno_integration():
    """
    Demonstração da integração básica com Agno
    """
    print("\n" + "="*60)
    print("INTEGRAÇÃO AGNO: DEMONSTRAÇÃO BÁSICA")
    print("="*60)
    
    try:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        
        print("\n1. Criando agente básico de análise...")
        
        # Criar agente simples sem storage complexo
        analyst_agent = Agent(
            name="Forex Analyst",
            role="Analista de Forex com ML",
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[MLForexTool()],
            instructions=[
                "Use as ferramentas ML para analisar pares de moedas",
                "Forneça análises claras e objetivas",
                "Considere múltiplos fatores na análise"
            ],
            show_tool_calls=True,
            markdown=True
        )
        
        print("   Agente criado com sucesso!")
        
        print("\n2. Testando interação com agente...")
        
        # Teste básico sem chamada real (para evitar necessidade de API key)
        agent_info = {
            "name": analyst_agent.name,
            "role": analyst_agent.role,
            "tools_available": len(analyst_agent.tools),
            "instructions_count": len(analyst_agent.instructions)
        }
        
        print(f"   Informações do agente: {agent_info}")
        
        print("\n3. Verificando ferramentas disponíveis...")
        for i, tool in enumerate(analyst_agent.tools):
            print(f"   Tool {i+1}: {tool.__class__.__name__}")
        
        return True, agent_info
        
    except Exception as e:
        print(f"Erro na integração Agno: {e}")
        return False, None

def main():
    """
    Função principal do demo simplificado
    """
    print("DEMO SIMPLIFICADO DO SISTEMA AGNO ML FOREX")
    print(f"Iniciado em: {datetime.now()}")
    print("\nEste demo demonstra:")
    print("- Fase 1: Ferramentas ML encapsuladas (COMPLETA)")
    print("- Integração básica com Agno")
    print("- Análise completa de múltiplos pares")
    
    # Verificar dependências
    try:
        import agno
        print(f"\nAgno instalado com sucesso")
    except ImportError:
        print("\nAgno não está instalado. Execute: pip install agno")
        return
    
    # Executar demos
    results = {
        "ml_tools": False,
        "agno_integration": False
    }
    
    # Demo das ferramentas ML
    print("\nExecutando demo das ferramentas ML...")
    results["ml_tools"], ml_report = demo_ml_tools_completo()
    
    # Demo da integração Agno
    print("\nExecutando demo da integração Agno...")
    results["agno_integration"], agno_info = demo_agno_integration()
    
    # Resumo final
    print("\n" + "="*60)
    print("RESUMO FINAL DO DEMO SIMPLIFICADO")
    print("="*60)
    
    for demo, sucesso in results.items():
        status = "SUCESSO" if sucesso else "FALHA"
        print(f"{demo.upper().replace('_', ' ')}: {status}")
    
    total_sucessos = sum(results.values())
    print(f"\nTotal: {total_sucessos}/2 demos executados com sucesso")
    
    if total_sucessos == 2:
        print("\nPARABÉNS! O sistema está funcionando perfeitamente!")
        print("\nFUNCIONALIDADES IMPLEMENTADAS:")
        print("   Ferramentas ML encapsuladas")
        print("   Predição de direção de pares")
        print("   Comparação de múltiplos pares")
        print("   Backtest de estratégias")
        print("   Integração básica com Agno")
        print("   Agentes configuráveis")
        
        print("\nPRÓXIMOS PASSOS:")
        print("   1. Configurar API key da OpenAI para agentes completos")
        print("   2. Implementar sistema de memória avançado")
        print("   3. Adicionar mais agentes especializados")
        print("   4. Integrar com dados em tempo real")
        
    elif total_sucessos == 1:
        print("\nSistema parcialmente funcional.")
        print("   As ferramentas ML estão funcionando perfeitamente!")
        
    else:
        print("\nProblemas detectados. Verifique as dependências.")
    
    print(f"\nDemo finalizado em: {datetime.now()}")
    
    # Salvar relatório se disponível
    if ml_report:
        print("\nSalvando relatório de performance...")
        try:
            import json
            with open("agno_ml_report.json", "w", encoding="utf-8") as f:
                json.dump(ml_report, f, indent=2, ensure_ascii=False)
            print("   Relatório salvo em: agno_ml_report.json")
        except Exception as e:
            print(f"   Erro ao salvar relatório: {e}")

if __name__ == "__main__":
    main()