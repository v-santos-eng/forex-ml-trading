# Demo do projeto Forex ML com Agno (sem chamadas de API)
# Este arquivo demonstra a estrutura sem fazer chamadas reais

import os
from config import config

def demo_project_structure():
    """Demonstra a estrutura do projeto sem fazer chamadas de API"""
    
    print("Demo: Projeto Forex ML com Agno")
    print("="*50)
    
    # Verificar configuração
    print("\nVerificando configuração...")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print(f"API Key configurada: {api_key[:20]}...")
    else:
        print("API Key não encontrada")
    
    # Mostrar configurações do projeto
    print("\nPares de moedas principais:")
    for i, pair in enumerate(config.MAJOR_PAIRS, 1):
        print(f"  {i}. {pair}")
    
    print("\nIndicadores técnicos disponíveis:")
    for i, indicator in enumerate(config.TECHNICAL_INDICATORS, 1):
        print(f"  {i}. {indicator}")
    
    print("\nBancos centrais monitorados:")
    for currency, bank in config.CENTRAL_BANKS.items():
        print(f"  {currency}: {bank}")
    
    print("\nConfigurações de risco:")
    for setting, value in config.RISK_SETTINGS.items():
        if isinstance(value, float):
            print(f"  {setting}: {value*100:.1f}%")
        else:
            print(f"  {setting}: {value}")
    
    # Demonstrar funcionalidades dos agentes
    print("\nAgentes disponíveis:")
    print("  1. Technical Analysis Agent - Análise técnica e indicadores")
    print("  2. Fundamental Analysis Agent - Dados econômicos e políticas")
    print("  3. Risk Management Agent - Gestão de risco e money management")
    print("  4. ML Prediction Agent - Machine learning e predições")
    
    # Exemplo de análise (simulada)
    print("\n" + "="*50)
    print("EXEMPLO DE ANÁLISE EUR/USD (SIMULADA)")
    print("="*50)
    
    pair_info = config.get_pair_info("EUR/USD")
    print(f"\nPar: {pair_info['pair']}")
    print(f"Moeda base: {pair_info['base_currency']} ({pair_info['base_bank']})")
    print(f"Moeda cotação: {pair_info['quote_currency']} ({pair_info['quote_bank']})")
    print(f"Tipo: {'Principal' if pair_info['is_major'] else 'Menor'}")
    
    # Simulação de análise técnica
    print("\nAnálise Técnica (Simulada):")
    print("  • Tendência: Bullish (Alta)")
    print("  • RSI: 65 (Neutro para sobrecomprado)")
    print("  • MACD: Sinal de compra")
    print("  • Suporte: 1.0850")
    print("  • Resistência: 1.0920")
    
    # Simulação de análise fundamental
    print("\nAnálise Fundamental (Simulada):")
    print("  • ECB: Política monetária restritiva")
    print("  • Fed: Possível pausa nos aumentos")
    print("  • Diferencial de juros: Favorável ao USD")
    print("  • Sentiment: Risk-on moderado")
    
    # Simulação de gestão de risco
    print("\nGestão de Risco (Simulada):")
    print(f"  • Risco máximo por trade: {config.RISK_SETTINGS['max_risk_per_trade']*100}%")
    print(f"  • Risk/Reward mínimo: {config.RISK_SETTINGS['min_risk_reward']}:1")
    print("  • Stop-loss sugerido: 1.0830")
    print("  • Take-profit: 1.0910")
    
    # Simulação de ML
    print("\nPredição ML (Simulada):")
    print("  • Probabilidade de alta (1h): 68%")
    print("  • Confiança do modelo: 72%")
    print("  • Features mais importantes: RSI, MACD, Volume")
    print("  • Sinal: COMPRA (moderado)")
    
    print("\n" + "="*50)
    print("DEMO CONCLUÍDA")
    print("="*50)
    
    print("\nPara usar com dados reais:")
    print("  1. Adicione créditos à sua conta Anthropic")
    print("  2. Execute: python simple_example.py")
    print("  3. Ou use: python forex_agent.py")
    
    print("\nLembre-se:")
    print("  • Esta é apenas uma demonstração")
    print("  • Não é aconselhamento financeiro")
    print("  • Sempre faça sua própria pesquisa")
    print("  • Use stop-loss em trades reais")

def show_file_structure():
    """Mostra a estrutura de arquivos do projeto"""
    print("\nEstrutura do Projeto:")
    print("forex-ml-trading/")
    print("├── simple_example.py       # COMECE AQUI")
    print("├── forex_agent.py          # Agente individual")
    print("├── forex_analysis.py       # Team de agentes")
    print("├── ml_forex_predictor.py   # ML avançado")
    print("├── config.py              # Configurações")
    print("├── demo_without_api.py    # Esta demo")
    print("├── requirements.txt       # Dependências")
    print("├── README.md              # Documentação")
    print("└── setup_guide.md         # Guia de setup")

def show_next_steps():
    """Mostra os próximos passos"""
    print("\nPróximos Passos:")
    print("\n1. Adicionar créditos Anthropic:")
    print("   • Acesse: https://console.anthropic.com/")
    print("   • Vá em 'Plans & Billing'")
    print("   • Adicione créditos ou upgrade o plano")
    
    print("\n2. Testar com dados reais:")
    print("   • python simple_example.py")
    print("   • python forex_agent.py")
    
    print("\n3. Personalizar:")
    print("   • Modifique config.py")
    print("   • Ajuste instruções dos agentes")
    print("   • Adicione novos pares de moedas")
    
    print("\n4. Expandir:")
    print("   • Implemente modelos ML reais")
    print("   • Adicione backtesting")
    print("   • Integre com brokers")

if __name__ == "__main__":
    demo_project_structure()
    show_file_structure()
    show_next_steps()