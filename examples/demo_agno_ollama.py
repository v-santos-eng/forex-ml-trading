#!/usr/bin/env python3
"""
Demo completo do sistema Agno ML Forex usando Ollama (modelos locais gratuitos)

Este demo demonstra:
1. Fase 1: Ferramentas ML Encapsuladas (mesmo que antes)
2. Fase 2: Sistema Multi-Agente usando Ollama
3. Fase 3: Memória e Raciocínio usando Ollama
4. Integração completa do sistema

Pré-requisitos:
- Ollama instalado e rodando (ollama serve)
- Modelo baixado (ollama pull llama3.2)
"""

import json
import os
from datetime import datetime
from typing import Dict, Any

# Imports do sistema ML (Fase 1)
from agno_ml_tool import MLForexTool
from config import ForexConfig

# Imports dos agentes Ollama (Fase 2)
from agno_ollama_agents import OllamaForexAgentTeam

# Imports do sistema de memória Ollama (Fase 3)
from agno_ollama_memory import OllamaForexMemorySystem

def check_ollama_requirements():
    """
    Verifica se o Ollama está disponível
    """
    print("Verificando requisitos do Ollama...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"Ollama está rodando com {len(models)} modelos")
            
            # Verificar se llama3.2 está disponível
            model_names = [model['name'] for model in models]
            if any('llama3.2' in name for name in model_names):
                print("Modelo llama3.2 encontrado")
                return True
            else:
                print("Modelo llama3.2 não encontrado")
                print("Execute: ollama pull llama3.2")
                return False
        else:
            print(f"Ollama retornou erro HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Ollama não está disponível: {e}")
        print("Execute: ollama serve")
        return False

def demo_fase_1_ml_tools():
    """
    Demonstra as ferramentas ML encapsuladas (Fase 1)
    """
    print("\n" + "="*60)
    print("FASE 1: FERRAMENTAS ML ENCAPSULADAS")
    print("="*60)
    
    # Inicializar ferramenta ML
    ml_tool = MLForexTool()
    
    # Testar com pares principais
    test_pairs = ["EURUSD=X", "GBPUSD=X"]
    
    results = {}
    
    for pair in test_pairs:
        print(f"\nTestando {pair}...")
        
        try:
            # Análise ML completa
            result = ml_tool.predict_forex_direction(
                pair=pair,
                period="6mo"
            )
            
            # Adicionar backtest se não houver erro
            if not result.get("error"):
                backtest_result = ml_tool.backtest_strategy(pair, "6mo")
                result["backtest"] = backtest_result
            
            results[pair] = result
            
            if result.get("error"):
                print(f"Erro: {result['error']}")
            else:
                print(f"Análise concluída")
                print(f"   - Amostras: {result.get('samples', 0)}")
                print(f"   - Modelos: {len(result.get('models', {}))}")
                print(f"   - Predição: {result.get('prediction', 'N/A')}")
                
        except Exception as e:
            print(f"Erro na análise: {e}")
            results[pair] = {"error": str(e)}
    
    # Salvar resultados
    with open("ollama_ml_report.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "phase": "ML Tools",
            "results": results
        }, f, indent=2)
    
    print(f"\nRelatório salvo em: ollama_ml_report.json")
    return results

def demo_fase_2_agentes_ollama(ml_results: Dict[str, Any]):
    """
    Demonstra o sistema multi-agente usando Ollama (Fase 2)
    """
    print("\n" + "="*60)
    print("FASE 2: SISTEMA MULTI-AGENTE OLLAMA")
    print("="*60)
    
    # Verificar se Ollama está disponível
    if not check_ollama_requirements():
        print("Ollama não está disponível. Pulando Fase 2.")
        return {"error": "Ollama não disponível"}
    
    # Inicializar equipe de agentes
    print("\nInicializando equipe de agentes Ollama...")
    agent_team = OllamaForexAgentTeam(model_id="llama3.2")
    
    # Verificar status
    status = agent_team.check_ollama_status()
    print(f"Status: {status}")
    
    if status["status"] != "online":
        print("Agentes não podem ser inicializados")
        return {"error": "Ollama offline"}
    
    # Análise com agentes
    test_pair = "EURUSD=X"
    print(f"\nAnálise completa de {test_pair} com agentes...")
    
    try:
        # Análise completa usando todos os agentes
        agent_analysis = agent_team.analyze_pair(test_pair)
        
        print("\nAnálise dos agentes concluída!")
        print(f"   - Análise Técnica: {len(agent_analysis.get('technical', ''))} caracteres")
        print(f"   - Análise Fundamental: {len(agent_analysis.get('fundamental', ''))} caracteres")
        print(f"   - Gestão de Risco: {len(agent_analysis.get('risk', ''))} caracteres")
        print(f"   - Decisão Final: {len(agent_analysis.get('decision', ''))} caracteres")
        
        # Análises rápidas adicionais
        print(f"\nAnálise rápida de GBPUSD=X...")
        quick_analysis = agent_team.quick_analysis("GBPUSD=X")
        
        print(f"\nAvaliação de risco para posição...")
        risk_assessment = agent_team.risk_assessment("EURUSD=X", 50000)
        
        results = {
            "main_analysis": agent_analysis,
            "quick_analysis": quick_analysis,
            "risk_assessment": risk_assessment,
            "ml_integration": ml_results.get(test_pair, {})
        }
        
        return results
        
    except Exception as e:
        print(f"Erro na análise dos agentes: {e}")
        return {"error": str(e)}

def demo_fase_3_memoria_ollama(agent_results: Dict[str, Any]):
    """
    Demonstra o sistema de memória e raciocínio usando Ollama (Fase 3)
    """
    print("\n" + "="*60)
    print("FASE 3: MEMÓRIA E RACIOCÍNIO OLLAMA")
    print("="*60)
    
    if agent_results.get("error"):
        print("Pulando Fase 3 devido a erros na Fase 2")
        return {"error": "Fase 2 falhou"}
    
    # Inicializar sistema de memória
    print("\nInicializando sistema de memória Ollama...")
    memory_system = OllamaForexMemorySystem(model_id="llama3.2")
    
    # Verificar status
    status = memory_system.check_ollama_status()
    stats = memory_system.get_memory_stats()
    
    print(f"Status Ollama: {status}")
    print(f"Estatísticas de Memória: {stats}")
    
    if status["status"] != "online":
        print("Sistema de memória não pode ser inicializado")
        return {"error": "Ollama offline"}
    
    try:
        # Armazenar decisão de trading
        print("\nArmazenando decisão de trading na memória...")
        session_id = memory_system.store_trading_decision(
            pair="EURUSD=X",
            decision=agent_results.get("main_analysis", {})
        )
        print(f"Decisão armazenada com ID: {session_id}")
        
        # Analisar histórico (simulado)
        print("\nAnalisando histórico de trading...")
        history_analysis = memory_system.analyze_trading_history(
            pair="EURUSD=X",
            days=30
        )
        
        # Obter recomendações
        print("\nObtendo recomendações baseadas em memória...")
        recommendations = memory_system.get_recommendations(
            pair="EURUSD=X",
            current_analysis=agent_results.get("main_analysis", {})
        )
        
        # Simular aprendizado com resultado
        print("\nSimulando aprendizado com resultado...")
        fake_outcome = {
            "profit_loss": 150.0,
            "success": True,
            "duration_hours": 24,
            "exit_reason": "take_profit"
        }
        
        learning_insights = memory_system.learn_from_outcomes(
            pair="EURUSD=X",
            actual_outcome=fake_outcome
        )
        
        results = {
            "session_id": session_id,
            "history_analysis": history_analysis,
            "recommendations": recommendations,
            "learning_insights": learning_insights,
            "memory_stats": stats
        }
        
        print("\nSistema de memória funcionando perfeitamente!")
        return results
        
    except Exception as e:
        print(f"Erro no sistema de memória: {e}")
        return {"error": str(e)}

def demo_integracao_completa(ml_results, agent_results, memory_results):
    """
    Demonstra a integração completa do sistema
    """
    print("\n" + "="*60)
    print("INTEGRAÇÃO COMPLETA DO SISTEMA OLLAMA")
    print("="*60)
    
    # Verificar se todas as fases funcionaram
    phases_status = {
        "Fase 1 (ML)": not bool(ml_results.get("error")),
        "Fase 2 (Agentes)": not bool(agent_results.get("error")),
        "Fase 3 (Memória)": not bool(memory_results.get("error"))
    }
    
    print("\nStatus das Fases:")
    for phase, status in phases_status.items():
        status_icon = "OK" if status else "ERRO"
        print(f"   {status_icon} {phase}: {'Funcionando' if status else 'Com problemas'}")
    
    # Relatório final
    final_report = {
        "timestamp": datetime.now().isoformat(),
        "system": "Agno ML Forex - Ollama Edition",
        "phases_status": phases_status,
        "ml_results": ml_results,
        "agent_results": agent_results,
        "memory_results": memory_results,
        "summary": {
            "total_phases": 3,
            "working_phases": sum(phases_status.values()),
            "success_rate": f"{(sum(phases_status.values()) / 3) * 100:.1f}%"
        }
    }
    
    # Salvar relatório final
    with open("ollama_complete_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\nRelatório completo salvo em: ollama_complete_report.json")
    
    # Resumo final
    working_phases = sum(phases_status.values())
    if working_phases == 3:
        print("\nSISTEMA COMPLETAMENTE FUNCIONAL!")
        print("   Todas as 3 fases estão operacionais com Ollama")
    elif working_phases >= 1:
        print(f"\nSISTEMA PARCIALMENTE FUNCIONAL")
        print(f"   {working_phases}/3 fases operacionais")
    else:
        print("\nSISTEMA COM PROBLEMAS")
        print("   Nenhuma fase está funcionando corretamente")
    
    return final_report

def main():
    """
    Função principal do demo
    """
    print("DEMO AGNO ML FOREX - OLLAMA EDITION")
    print("="*60)
    print("Sistema de trading Forex com IA usando modelos locais gratuitos")
    print("\nPré-requisitos:")
    print("- Ollama instalado e rodando (ollama serve)")
    print("- Modelo baixado (ollama pull llama3.2)")
    print("="*60)
    
    try:
        # Fase 1: Ferramentas ML
        ml_results = demo_fase_1_ml_tools()
        
        # Fase 2: Agentes Ollama
        agent_results = demo_fase_2_agentes_ollama(ml_results)
        
        # Fase 3: Memória Ollama
        memory_results = demo_fase_3_memoria_ollama(agent_results)
        
        # Integração completa
        final_report = demo_integracao_completa(ml_results, agent_results, memory_results)
        
        print("\nDemo concluído!")
        print("\nArquivos gerados:")
        print("   - ollama_ml_report.json")
        print("   - ollama_complete_report.json")
        print("   - forex_memory.db (banco de dados de memória)")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrompido pelo usuário")
    except Exception as e:
        print(f"\n\nErro inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()