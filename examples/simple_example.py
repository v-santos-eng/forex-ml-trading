from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools
import os

def main():
    """Exemplo simples de uso do Agno para análise forex"""
    
    # Verificar se a API key está configurada
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERRO: ANTHROPIC_API_KEY não encontrada!")
        print("\nPara configurar:")
        print("1. Obtenha sua API key em: https://console.anthropic.com/")
        print("2. Configure como variável de ambiente:")
        print("   Windows: set ANTHROPIC_API_KEY=sk-ant-api03-xxxx")
        print("   Linux/Mac: export ANTHROPIC_API_KEY=sk-ant-api03-xxxx")
        return
    
    print("Iniciando Forex Agent com Agno...")
    
    # Criar agente simples para forex
    forex_agent = Agent(
        model=Claude(id="claude-3-5-sonnet-20241022"),
        tools=[
            ReasoningTools(add_instructions=True),
            YFinanceTools(
                stock_price=True,
                company_info=True,
                company_news=True
            )
        ],
        instructions=[
            "Você é um analista de forex experiente.",
            "Forneça análises claras e objetivas.",
            "Use dados atuais para suas análises.",
            "Sempre inclua disclaimers sobre riscos.",
            "Apresente dados em tabelas quando apropriado."
        ],
        markdown=True,
        show_tool_calls=True
    )
    
    print("Agent criado com sucesso!")
    print("\n" + "="*50)
    print("ANÁLISE FOREX - EUR/USD")
    print("="*50)
    
    # Fazer análise do EUR/USD
    response = forex_agent.print_response(
        """
        Faça uma análise rápida do par EUR/USD:
        
        1. Obtenha o preço atual
        2. Analise a tendência recente
        3. Identifique fatores que podem influenciar o par
        4. Forneça uma perspectiva de curto prazo
        5. Inclua disclaimers apropriados
        
        Mantenha a análise concisa mas informativa.
        """,
        stream=True,
        show_full_reasoning=True
    )
    
    print("\n" + "="*50)
    print("Análise concluída!")
    print("\nDicas:")
    print("- Modifique forex_agent.py para análises mais avançadas")
    print("- Use forex_analysis.py para análise com múltiplos agentes")
    print("- Sempre faça sua própria pesquisa antes de investir")

if __name__ == "__main__":
    main()