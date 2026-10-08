#!/usr/bin/env python3
"""
Demo Agno Ollama - Versão Local
Demonstração completa do sistema de trading com Ollama usando dados locais

Esta versão usa dados históricos locais em vez do Yahoo Finance
"""

import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importações do sistema
from agno_ml_tool_local import MLForexToolLocal
from agno_ollama_agents import OllamaForexAgentTeam
from agno_ollama_memory import OllamaForexMemorySystem

def demo_fase_1_ml_tools():
    """
    Fase 1: Demonstração das ferramentas ML com dados locais
    """
    print("\n" + "=" * 80)
    print("FASE 1: FERRAMENTAS DE MACHINE LEARNING (DADOS LOCAIS)")
    print("=" * 80)
    
    # Inicializar ferramenta ML local
    ml_tool = MLForexToolLocal()
    
    # Listar pares disponíveis
    print("\nPares de moedas disponíveis:")
    available_pairs = ml_tool.predictor.get_available_pairs()
    
    # Usar pares locais sem sufixo =X
    test_pairs = ['EURUSD', 'GBPUSD'] if available_pairs else list(available_pairs.keys())[:2]
    
    results = {}
    
    for pair in test_pairs:
        print(f"\n{'-' * 60}")
        print(f"ANALISANDO {pair}")
        print("-" * 60)
        
        try:
            # Predição de direção
            direction_result = ml_tool.predict_forex_direction(pair)
            print(f"\nPredição de Direção:")
            print(f"  Par: {pair}")
            print(f"  Direção: {direction_result.get('direction', 'N/A')}")
            print(f"  Confiança: {direction_result.get('confidence', 0):.1f}%")
            print(f"  Modelo: {direction_result.get('model', 'N/A')}")
            
            # Backtest da estratégia
            print(f"\nRealizando backtest...")
            backtest_result = ml_tool.backtest_strategy(pair)
            
            if backtest_result and 'error' not in backtest_result:
                print(f"\nResultados do Backtest:")
                print(f"  Acurácia: {backtest_result.get('accuracy', 0):.3f}")
                print(f"  Taxa de acerto: {backtest_result.get('win_rate', 0):.3f}")
                print(f"  Total de trades: {backtest_result.get('total_trades', 0)}")
                print(f"  Retorno total: {backtest_result.get('total_return', 0)}")
            else:
                print(f"  Erro no backtest: {backtest_result.get('error', 'Desconhecido')}")
            
            # Armazenar resultados
            results[pair] = {
                'direction': direction_result,
                'backtest': backtest_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Erro ao analisar {pair}: {e}")
            results[pair] = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Salvar relatório
    report_file = "ollama_ml_report_local.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'pairs_analyzed': len(test_pairs),
            'results': results,
            'data_source': 'local_files'
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nRelatório salvo em: {report_file}")
    return results

def demo_fase_2_ollama_agents(ml_results):
    """
    Fase 2: Demonstração dos agentes Ollama
    """
    print("\n" + "=" * 80)
    print("FASE 2: AGENTES OLLAMA PARA ANÁLISE DE MERCADO")
    print("=" * 80)
    
    # Inicializar equipe de agentes Ollama
    agent_team = OllamaForexAgentTeam(model_id="llama3.2:1b")
    agent = agent_team.technical_agent  # Usar agente técnico
    
    agent_results = {}
    
    for pair, ml_data in ml_results.items():
        if 'error' in ml_data:
            continue
            
        print(f"\n{'-' * 60}")
        print(f"ANÁLISE OLLAMA PARA {pair}")
        print("-" * 60)
        
        try:
            # Preparar contexto para o agente
            market_context = {
                'pair': pair,
                'ml_prediction': ml_data.get('direction', {}),
                'backtest_results': ml_data.get('backtest', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # Análise técnica com ML
            print("\nAnalisando com agente técnico...")
            ml_prompt = f"Analise os dados de ML para {pair}: direção {ml_data.get('direction', {}).get('direction', 'N/A')}, confiança {ml_data.get('direction', {}).get('confidence', 0)}%, acurácia do backtest {ml_data.get('backtest', {}).get('accuracy', 0)}"
            
            technical_analysis = agent.run(ml_prompt)
            print(f"Análise Técnica: {str(technical_analysis)[:200]}...")
            
            # Usar agente fundamental para análise complementar
            fundamental_agent = agent_team.fundamental_agent
            fundamental_prompt = f"Forneça análise fundamental para o par {pair} considerando o contexto atual do mercado"
            
            fundamental_analysis = fundamental_agent.run(fundamental_prompt)
            print(f"Análise Fundamental: {str(fundamental_analysis)[:200]}...")
            
            # Decisão combinada
            trading_decision = f"Baseado na análise técnica (ML) e fundamental, recomendação para {pair}: {ml_data.get('direction', {}).get('direction', 'NEUTRO')}"
            
            # Armazenar resultados do agente
            agent_results[pair] = {
                'technical_analysis': str(technical_analysis),
                'fundamental_analysis': str(fundamental_analysis),
                'trading_decision': trading_decision,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Erro na análise Ollama para {pair}: {e}")
            agent_results[pair] = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    return agent_results

def demo_fase_3_memory_system(ml_results, agent_results):
    """
    Fase 3: Demonstração do sistema de memória Ollama
    """
    print("\n" + "=" * 80)
    print("FASE 3: SISTEMA DE MEMÓRIA OLLAMA")
    print("=" * 80)
    
    # Inicializar sistema de memória
    memory = OllamaForexMemorySystem()
    
    print("\nArmazenando decisões de trading na memória...")
    
    # Armazenar todas as decisões na memória
    for pair in ml_results.keys():
        if pair in agent_results and 'error' not in agent_results[pair]:
            try:
                decision_data = {
                    'pair': pair,
                    'ml_prediction': ml_results[pair].get('direction', {}),
                    'agent_decision': agent_results[pair].get('trading_decision', ''),
                    'timestamp': datetime.now().isoformat()
                }
                
                memory.store_trading_decision(decision_data)
                print(f"  ✓ Decisão para {pair} armazenada")
                
            except Exception as e:
                print(f"  ✗ Erro ao armazenar {pair}: {e}")
    
    # Analisar histórico de trading
    print("\nAnalisando histórico de trading...")
    try:
        history_analysis = memory.analyze_trading_history()
        print(f"Análise do Histórico: {history_analysis[:300]}...")
        
        # Buscar padrões similares
        print("\nBuscando padrões similares...")
        similar_patterns = memory.find_similar_patterns("EURUSD alta confiança")
        print(f"Padrões Similares: {len(similar_patterns)} encontrados")
        
        return {
            'decisions_stored': len([p for p in ml_results.keys() if p in agent_results]),
            'history_analysis': history_analysis,
            'similar_patterns_count': len(similar_patterns)
        }
        
    except Exception as e:
        print(f"Erro na análise de memória: {e}")
        return {'error': str(e)}

def main():
    """
    Demonstração completa do sistema Ollama com dados locais
    """
    print("DEMO AGNO OLLAMA - VERSÃO LOCAL")
    print("Sistema completo de trading com Ollama usando dados locais")
    print("=" * 80)
    
    try:
        # Fase 1: Ferramentas ML
        ml_results = demo_fase_1_ml_tools()
        
        if not ml_results:
            print("\n❌ Nenhum resultado ML obtido. Encerrando demo.")
            return
        
        # Fase 2: Agentes Ollama
        agent_results = demo_fase_2_ollama_agents(ml_results)
        
        # Fase 3: Sistema de Memória
        memory_results = demo_fase_3_memory_system(ml_results, agent_results)
        
        # Relatório final
        print("\n" + "=" * 80)
        print("RELATÓRIO FINAL")
        print("=" * 80)
        
        total_pairs = len(ml_results)
        successful_ml = len([r for r in ml_results.values() if 'error' not in r])
        successful_agents = len([r for r in agent_results.values() if 'error' not in r])
        
        print(f"\nEstatísticas da Execução:")
        print(f"  Pares analisados: {total_pairs}")
        print(f"  ML bem-sucedido: {successful_ml}/{total_pairs}")
        print(f"  Agentes bem-sucedidos: {successful_agents}/{total_pairs}")
        print(f"  Decisões na memória: {memory_results.get('decisions_stored', 0)}")
        
        # Salvar relatório completo
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'execution_stats': {
                'total_pairs': total_pairs,
                'successful_ml': successful_ml,
                'successful_agents': successful_agents,
                'memory_decisions': memory_results.get('decisions_stored', 0)
            },
            'ml_results': ml_results,
            'agent_results': agent_results,
            'memory_results': memory_results,
            'data_source': 'local_files'
        }
        
        report_file = f"demo_ollama_complete_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Demo concluída com sucesso!")
        print(f"📊 Relatório completo salvo em: {report_file}")
        
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()