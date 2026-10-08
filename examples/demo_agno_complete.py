#!/usr/bin/env python3
"""
Demo Completo do Sistema Agno ML Forex
Integração das Fases 1, 2 e 3

Este script demonstra:
- Fase 1: Ferramentas ML encapsuladas
- Fase 2: Sistema multi-agente
- Fase 3: Memória e raciocínio
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agno_ml_tool import MLForexTool
    from agno_forex_agents import ForexAgentTeam
    from agno_memory_system import ForexMemorySystem, create_memory_enhanced_analysis
except ImportError as e:
    print(f"Erro ao importar módulos Agno: {e}")
    print("Certifique-se de que o Agno está instalado: pip install agno")
    sys.exit(1)

def demo_fase_1_ml_tools():
    """
    Demonstração da Fase 1: Ferramentas ML
    """
    print("\n" + "="*60)
    print("FASE 1: DEMONSTRAÇÃO DAS FERRAMENTAS ML")
    print("="*60)
    
    try:
        # Inicializar ferramenta ML
        ml_tool = MLForexTool()
        
        print("\n1. Testando predição de direção...")
        result = ml_tool.predict_forex_direction(
            pair="EURUSD=X",
            period="6mo"
        )
        print(f"Resultado: {result}")
        
        print("\n2. Obtendo informações dos modelos...")
        model_info = ml_tool.get_model_info()
        print(f"Modelos disponíveis: {model_info}")
        
        print("\n3. Comparando múltiplos pares...")
        comparison = ml_tool.compare_pairs(
            pairs=["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
            period="3mo"
        )
        print(f"Comparação: {comparison}")
        
        return True
        
    except Exception as e:
        print(f"Erro na Fase 1: {e}")
        return False

def demo_fase_2_multi_agent():
    """
    Demonstração da Fase 2: Sistema Multi-Agente
    """
    print("\n" + "="*60)
    print("FASE 2: DEMONSTRAÇÃO DO SISTEMA MULTI-AGENTE")
    print("="*60)
    
    try:
        # Inicializar equipe de agentes
        agent_team = ForexAgentTeam()
        
        print("\n1. Análise completa com agentes especializados...")
        complete_analysis = agent_team.analyze_pair(
            pair="EURUSD=X",
            period="6mo"
        )
        print(f"Análise Completa: {complete_analysis}")
        
        return True
        
    except Exception as e:
        print(f"Erro na Fase 2: {e}")
        return False

def demo_fase_3_memory_reasoning():
    """
    Demonstração da Fase 3: Memória e Raciocínio
    """
    print("\n" + "="*60)
    print("FASE 3: DEMONSTRAÇÃO DE MEMÓRIA E RACIOCÍNIO")
    print("="*60)
    
    try:
        # Inicializar sistema de memória
        memory_system = ForexMemorySystem()
        
        print("\n1. Estatísticas do sistema de memória...")
        stats = memory_system.get_memory_stats()
        print(f"Estatísticas: {stats}")
        
        print("\n2. Armazenando predição de exemplo...")
        prediction_id = memory_system.store_prediction(
            pair="EURUSD=X",
            prediction="ALTA",
            confidence=0.75,
            model_used="Ensemble",
            notes="Predição de demonstração"
        )
        print(f"Predição armazenada com ID: {prediction_id}")
        
        print("\n3. Análise com memória histórica...")
        memory_analysis = memory_system.analyze_with_memory(
            pair="EURUSD=X",
            period="6mo"
        )
        print(f"Análise com Memória: {memory_analysis}")
        
        print("\n4. Sessão de aprendizado contínuo...")
        learning_session = memory_system.continuous_learning_session()
        print(f"Aprendizado Contínuo: {learning_session}")
        
        return True
        
    except Exception as e:
        print(f"Erro na Fase 3: {e}")
        return False

def demo_integracao_completa():
    """
    Demonstração da integração completa de todas as fases
    """
    print("\n" + "="*60)
    print("INTEGRAÇÃO COMPLETA: TODAS AS FASES JUNTAS")
    print("="*60)
    
    try:
        # Análise completa usando memória
        print("\n1. Análise completa com memória...")
        complete_analysis = create_memory_enhanced_analysis("EURUSD=X")
        print(f"Análise Completa com Memória: {complete_analysis}")
        
        # Simulação de workflow completo
        print("\n2. Workflow completo de trading...")
        
        # Fase 1: Predição ML
        ml_tool = MLForexTool()
        ml_prediction = ml_tool.predict_forex_direction("EURUSD=X", "6mo")
        
        # Fase 2: Análise multi-agente
        agent_team = ForexAgentTeam()
        agent_analysis = agent_team.complete_analysis("EURUSD=X", "6mo")
        
        # Fase 3: Armazenar na memória
        memory_system = ForexMemorySystem()
        prediction_id = memory_system.store_prediction(
            pair="EURUSD=X",
            prediction=ml_prediction.get('prediction', 'NEUTRO'),
            confidence=ml_prediction.get('confidence', 0.5),
            model_used="Agno Integrated System",
            notes=f"Análise integrada: {agent_analysis}"
        )
        
        workflow_result = {
            "timestamp": datetime.now().isoformat(),
            "pair": "EURUSD=X",
            "ml_prediction": ml_prediction,
            "agent_analysis": agent_analysis,
            "memory_id": prediction_id,
            "status": "success"
        }
        
        print(f"Workflow Completo: {workflow_result}")
        
        return True
        
    except Exception as e:
        print(f"Erro na Integração Completa: {e}")
        return False

def main():
    """
    Função principal do demo
    """
    print("DEMO COMPLETO DO SISTEMA AGNO ML FOREX")
    print(f"Iniciado em: {datetime.now()}")
    print("\nEste demo demonstra a integração completa das 3 fases:")
    print("- Fase 1: Ferramentas ML encapsuladas")
    print("- Fase 2: Sistema multi-agente especializado")
    print("- Fase 3: Memória e raciocínio avançado")
    
    # Verificar dependências
    try:
        import agno
        print(f"\nAgno instalado com sucesso")
    except ImportError:
        print("\nAgno não está instalado. Execute: pip install agno")
        return
    
    # Executar demos das fases
    results = {
        "fase_1": False,
        "fase_2": False,
        "fase_3": False,
        "integracao": False
    }
    
    # Fase 1
    results["fase_1"] = demo_fase_1_ml_tools()
    
    # Fase 2
    results["fase_2"] = demo_fase_2_multi_agent()
    
    # Fase 3
    results["fase_3"] = demo_fase_3_memory_reasoning()
    
    # Integração Completa
    results["integracao"] = demo_integracao_completa()
    
    # Resumo final
    print("\n" + "="*60)
    print("RESUMO FINAL DO DEMO")
    print("="*60)
    
    for fase, sucesso in results.items():
        status = "OK SUCESSO" if sucesso else "ERRO FALHA"
        print(f"{fase.upper()}: {status}")
    
    total_sucessos = sum(results.values())
    print(f"\nTotal: {total_sucessos}/4 fases executadas com sucesso")
    
    if total_sucessos == 4:
        print("\nPARABÉNS! Todas as fases foram implementadas e testadas com sucesso!")
        print("\nO sistema Agno ML Forex está pronto para uso com:")
        print("- Ferramentas ML avançadas")
        print("- Agentes especializados")
        print("- Sistema de memória e aprendizado")
        print("- Raciocínio estruturado")
    else:
        print("\nALERTA: Algumas fases apresentaram problemas. Verifique os logs acima.")
    
    print(f"\nDemo finalizado em: {datetime.now()}")

if __name__ == "__main__":
    main()