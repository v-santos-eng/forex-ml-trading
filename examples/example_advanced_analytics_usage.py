#!/usr/bin/env python3
"""
Exemplo Prático: Uso Integrado dos Sistemas de Análise Avançada

Este exemplo demonstra como usar os novos sistemas de:
1. Volume Profile Analysis
2. Correlation Analysis

Integrando-os com o sistema de risco existente para decisões de trading mais informadas.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Any

# Importar os novos sistemas
from volume_profile_analyzer import (
    VolumeProfileAnalyzer, VolumeProfileParameters, VolumeProfileType,
    VolumeNodeType
)
from correlation_analyzer import (
    CorrelationAnalyzer, CorrelationParameters, CorrelationType,
    CorrelationStrength
)

# Importar sistema de risco existente
from advanced_risk_system import AdvancedRiskSystem, RiskParameters


class AdvancedAnalyticsIntegrator:
    """Integrador dos sistemas de análise avançada com gestão de risco"""
    
    def __init__(self):
        """Inicializar o integrador com configurações otimizadas"""
        self.logger = self._setup_logging()
        
        # Configurar parâmetros dos analisadores
        self.volume_params = VolumeProfileParameters(
            price_bins=30,
            value_area_percentage=0.70,
            hvn_threshold=0.05,
            lvn_threshold=0.01,
            min_volume_threshold=100
        )
        
        self.correlation_params = CorrelationParameters(
            lookback_periods=50,
            min_correlation_threshold=0.3,
            rolling_window=20,
            update_frequency=15,
            p_value_threshold=0.05,
            min_sample_size=20
        )
        
        # Inicializar analisadores
        self.volume_analyzer = VolumeProfileAnalyzer(self.volume_params, self.logger)
        self.correlation_analyzer = CorrelationAnalyzer(self.correlation_params, self.logger)
        
        # Configurar sistema de risco
        self.risk_params = RiskParameters(
            max_risk_per_trade=0.02,
            max_daily_risk=0.06,
            max_monthly_risk=0.10,
            max_positions=5,
            min_risk_reward_ratio=1.5
        )
        
        self.logger.info("AdvancedAnalyticsIntegrator inicializado com sucesso")
    
    def _setup_logging(self) -> logging.Logger:
        """Configurar sistema de logging"""
        logger = logging.getLogger('AdvancedAnalyticsIntegrator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_sample_market_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Gerar dados de mercado simulados para demonstração"""
        self.logger.info(f"Gerando dados simulados para {symbol} ({days} dias)")
        
        # Configurações base por símbolo
        base_configs = {
            'EURUSD': {'base_price': 1.1000, 'volatility': 0.008, 'trend': 0.0001},
            'GBPUSD': {'base_price': 1.3000, 'volatility': 0.012, 'trend': 0.0002},
            'USDJPY': {'base_price': 110.00, 'volatility': 0.006, 'trend': -0.0001},
            'AUDUSD': {'base_price': 0.7500, 'volatility': 0.010, 'trend': 0.0001},
            'USDCHF': {'base_price': 0.9200, 'volatility': 0.007, 'trend': -0.0001}
        }
        
        config = base_configs.get(symbol, base_configs['EURUSD'])
        
        # Gerar timestamps (dados horários)
        periods = days * 24
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=periods,
            freq='1H'
        )
        
        # Gerar preços com tendência e volatilidade
        np.random.seed(42)  # Para resultados reproduzíveis
        
        prices = []
        volumes = []
        current_price = config['base_price']
        
        for i in range(periods):
            # Movimento de preço com tendência e ruído
            price_change = (
                config['trend'] +  # Tendência
                np.random.normal(0, config['volatility'])  # Volatilidade
            )
            
            current_price *= (1 + price_change)
            
            # Gerar OHLC realístico
            spread = current_price * 0.0002  # Spread típico
            
            high = current_price + np.random.uniform(0, spread * 3)
            low = current_price - np.random.uniform(0, spread * 3)
            open_price = np.random.uniform(low, high)
            close = current_price
            
            # Garantir ordem correta OHLC
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            prices.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close
            })
            
            # Volume correlacionado com volatilidade
            base_volume = 1000
            volume_multiplier = 1 + abs(price_change) * 50
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
            volumes.append(volume)
        
        # Criar DataFrame
        df = pd.DataFrame(prices)
        df['volume'] = volumes
        df['timestamp'] = timestamps
        df['symbol'] = symbol
        
        self.logger.info(f"Dados gerados: {len(df)} registros para {symbol}")
        return df
    
    def perform_volume_profile_analysis(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Realizar análise completa de volume profile"""
        self.logger.info(f"Iniciando análise de volume profile para {symbol}")
        
        try:
            # Calcular volume profile
            profile = self.volume_analyzer.calculate_volume_profile(data)
            
            # Extrair níveis de suporte e resistência
            levels = self.volume_analyzer.get_support_resistance_levels(profile)
            
            # Analisar ação do preço atual
            current_price = data['close'].iloc[-1]
            price_analysis = self.volume_analyzer.analyze_price_action_at_levels(
                current_price, profile
            )
            
            # Gerar sinais de trading
            market_bias = 'bullish' if data['close'].iloc[-1] > data['close'].iloc[-10] else 'bearish'
            trading_signals = self.volume_analyzer.get_trading_signals(
                current_price, profile, market_bias
            )
            
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'volume_profile': {
                    'total_volume': profile['total_volume'],
                    'poc': levels['poc'],
                    'value_area_high': levels.get('value_area_high', 0),
                    'value_area_low': levels.get('value_area_low', 0),
                    'hvn_levels': [node.price for node in profile['volume_nodes'] 
                                 if node.node_type == VolumeNodeType.HVN],
                    'lvn_levels': [node.price for node in profile['volume_nodes'] 
                                 if node.node_type == VolumeNodeType.LVN]
                },
                'support_resistance': {
                    'support_levels': levels['support_levels'],
                    'resistance_levels': levels['resistance_levels'],
                    'key_levels': levels['key_levels']
                },
                'price_analysis': price_analysis,
                'trading_signals': trading_signals
            }
            
            self.logger.info(f"Análise de volume profile concluída para {symbol}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Erro na análise de volume profile para {symbol}: {str(e)}")
            return {'error': str(e), 'symbol': symbol}
    
    def perform_correlation_analysis(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Realizar análise de correlação entre múltiplos ativos"""
        self.logger.info("Iniciando análise de correlação multi-ativo")
        
        try:
            # Adicionar dados de preço para todos os símbolos
            symbols = list(market_data.keys())
            
            for symbol, data in market_data.items():
                for _, row in data.iterrows():
                    self.correlation_analyzer.add_price_data(
                        symbol, row['close'], row['timestamp']
                    )
            
            # Calcular matriz de correlação
            correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(symbols)
            
            # Identificar clusters de correlação
            clusters = self.correlation_analyzer.get_correlation_clusters(
                correlation_matrix, min_cluster_size=2
            )
            
            # Detectar quebras de correlação
            breakdowns = self.correlation_analyzer.detect_correlation_breakdowns(
                threshold=0.15
            )
            
            # Gerar insights para cada símbolo principal
            insights_by_symbol = {}
            for symbol in symbols[:3]:  # Limitar a 3 símbolos principais
                insights_by_symbol[symbol] = self.correlation_analyzer.generate_trading_insights(
                    correlation_matrix, symbol
                )
            
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'analyzed_symbols': symbols,
                'correlation_matrix': {
                    'assets': correlation_matrix.assets,
                    'significant_pairs': [
                        {
                            'asset1': pair.asset1,
                            'asset2': pair.asset2,
                            'correlation': pair.correlation,
                            'strength': pair.strength.value
                        }
                        for pair in correlation_matrix.significant_pairs
                    ]
                },
                'correlation_clusters': clusters,
                'correlation_breakdowns': breakdowns,
                'trading_insights': insights_by_symbol
            }
            
            self.logger.info("Análise de correlação concluída")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Erro na análise de correlação: {str(e)}")
            return {'error': str(e)}
    
    def integrate_with_risk_management(self, 
                                     volume_analysis: Dict[str, Any],
                                     correlation_analysis: Dict[str, Any],
                                     symbol: str) -> Dict[str, Any]:
        """Integrar análises com sistema de gestão de risco"""
        self.logger.info(f"Integrando análises com gestão de risco para {symbol}")
        
        try:
            # Extrair informações relevantes
            current_price = volume_analysis.get('current_price', 0)
            poc = volume_analysis.get('volume_profile', {}).get('poc', current_price)
            
            # Calcular níveis de risco baseados em volume profile
            value_area_high = volume_analysis.get('volume_profile', {}).get('value_area_high', current_price * 1.01)
            value_area_low = volume_analysis.get('volume_profile', {}).get('value_area_low', current_price * 0.99)
            
            # Ajustar risco baseado em correlações
            correlation_adjustment = 1.0
            if symbol in correlation_analysis.get('trading_insights', {}):
                symbol_insights = correlation_analysis['trading_insights'][symbol]
                high_corr_pairs = len([c for c in symbol_insights.get('correlations', {}).values() 
                                     if abs(c) > 0.7])
                # Reduzir risco se há muitas correlações altas (concentração)
                correlation_adjustment = max(0.5, 1.0 - (high_corr_pairs * 0.1))
            
            # Calcular sugestões de posicionamento
            distance_to_poc = abs(current_price - poc) / current_price
            
            if distance_to_poc < 0.001:  # Muito próximo do POC
                position_suggestion = "NEUTRAL - Aguardar breakout da área de valor"
                risk_multiplier = 0.8
            elif current_price > value_area_high:
                position_suggestion = "BULLISH - Preço acima da área de valor"
                risk_multiplier = 1.2
            elif current_price < value_area_low:
                position_suggestion = "BEARISH - Preço abaixo da área de valor"
                risk_multiplier = 1.2
            else:
                position_suggestion = "RANGE - Preço dentro da área de valor"
                risk_multiplier = 1.0
            
            # Calcular tamanho de posição ajustado
            base_position_size = self.risk_params.max_risk_per_trade
            adjusted_position_size = base_position_size * correlation_adjustment * risk_multiplier
            
            # Definir níveis de stop loss e take profit
            if current_price > poc:
                stop_loss = max(value_area_low, poc * 0.995)
                take_profit = current_price + (current_price - stop_loss) * 2
            else:
                stop_loss = min(value_area_high, poc * 1.005)
                take_profit = current_price - (stop_loss - current_price) * 2
            
            risk_integration = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'risk_assessment': {
                    'position_suggestion': position_suggestion,
                    'risk_multiplier': risk_multiplier,
                    'correlation_adjustment': correlation_adjustment,
                    'adjusted_position_size': adjusted_position_size,
                    'max_position_size': base_position_size
                },
                'trading_levels': {
                    'poc': poc,
                    'value_area_high': value_area_high,
                    'value_area_low': value_area_low,
                    'suggested_stop_loss': stop_loss,
                    'suggested_take_profit': take_profit,
                    'risk_reward_ratio': abs(take_profit - current_price) / abs(current_price - stop_loss)
                },
                'volume_context': volume_analysis.get('trading_signals', {}),
                'correlation_context': correlation_analysis.get('trading_insights', {}).get(symbol, {})
            }
            
            self.logger.info(f"Integração com gestão de risco concluída para {symbol}")
            return risk_integration
            
        except Exception as e:
            self.logger.error(f"Erro na integração com gestão de risco: {str(e)}")
            return {'error': str(e), 'symbol': symbol}
    
    def generate_comprehensive_report(self, analyses: Dict[str, Any]) -> str:
        """Gerar relatório abrangente das análises"""
        self.logger.info("Gerando relatório abrangente")
        
        report_lines = [
            "=" * 80,
            "RELATÓRIO DE ANÁLISE AVANÇADA DE MERCADO",
            "=" * 80,
            f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Seção de Volume Profile
        if 'volume_analyses' in analyses:
            report_lines.extend([
                "📊 ANÁLISE DE VOLUME PROFILE",
                "-" * 40
            ])
            
            for symbol, analysis in analyses['volume_analyses'].items():
                if 'error' not in analysis:
                    vp = analysis.get('volume_profile', {})
                    signals = analysis.get('trading_signals', {})
                    
                    report_lines.extend([
                        f"\n{symbol}:",
                        f"  Preço Atual: {analysis.get('current_price', 0):.5f}",
                        f"  POC (Point of Control): {vp.get('poc', 0):.5f}",
                        f"  Área de Valor: {vp.get('value_area_low', 0):.5f} - {vp.get('value_area_high', 0):.5f}",
                        f"  Volume Total: {vp.get('total_volume', 0):,}",
                        f"  Força do Sinal: {signals.get('signal_strength', 0):.2f}",
                        f"  Confiança: {signals.get('confidence', 0):.2f}"
                    ])
        
        # Seção de Correlação
        if 'correlation_analysis' in analyses:
            corr_analysis = analyses['correlation_analysis']
            if 'error' not in corr_analysis:
                report_lines.extend([
                    "\n\n🔗 ANÁLISE DE CORRELAÇÃO",
                    "-" * 40
                ])
                
                significant_pairs = corr_analysis.get('correlation_matrix', {}).get('significant_pairs', [])
                
                report_lines.append(f"\nPares com Correlação Significativa ({len(significant_pairs)}):")
                for pair in significant_pairs[:5]:  # Top 5
                    report_lines.append(
                        f"  {pair['asset1']} x {pair['asset2']}: {pair['correlation']:.3f} ({pair['strength']})"
                    )
                
                clusters = corr_analysis.get('correlation_clusters', [])
                if clusters:
                    report_lines.append(f"\nClusters Identificados ({len(clusters)}):")
                    for i, cluster in enumerate(clusters[:3]):
                        assets = ', '.join(cluster['assets'])
                        report_lines.append(
                            f"  Cluster {i+1}: {assets} (Corr. Média: {cluster['avg_internal_correlation']:.3f})"
                        )
        
        # Seção de Integração de Risco
        if 'risk_integrations' in analyses:
            report_lines.extend([
                "\n\n⚠️ INTEGRAÇÃO COM GESTÃO DE RISCO",
                "-" * 40
            ])
            
            for symbol, integration in analyses['risk_integrations'].items():
                if 'error' not in integration:
                    risk_assess = integration.get('risk_assessment', {})
                    trading_levels = integration.get('trading_levels', {})
                    
                    report_lines.extend([
                        f"\n{symbol}:",
                        f"  Sugestão: {risk_assess.get('position_suggestion', 'N/A')}",
                        f"  Tamanho Ajustado: {risk_assess.get('adjusted_position_size', 0):.2%}",
                        f"  Stop Loss Sugerido: {trading_levels.get('suggested_stop_loss', 0):.5f}",
                        f"  Take Profit Sugerido: {trading_levels.get('suggested_take_profit', 0):.5f}",
                        f"  Risk/Reward: 1:{trading_levels.get('risk_reward_ratio', 0):.2f}"
                    ])
        
        # Rodapé
        report_lines.extend([
            "\n\n" + "=" * 80,
            "Relatório gerado pelo Sistema de Análise Avançada",
            "Para uso em decisões de trading informadas",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def run_complete_analysis(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Executar análise completa para múltiplos símbolos"""
        if symbols is None:
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        self.logger.info(f"Iniciando análise completa para símbolos: {symbols}")
        
        # Gerar dados de mercado
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = self.generate_sample_market_data(symbol, days=15)
        
        # Análises de volume profile
        volume_analyses = {}
        for symbol, data in market_data.items():
            volume_analyses[symbol] = self.perform_volume_profile_analysis(data, symbol)
        
        # Análise de correlação
        correlation_analysis = self.perform_correlation_analysis(market_data)
        
        # Integração com gestão de risco
        risk_integrations = {}
        for symbol in symbols:
            if symbol in volume_analyses and 'error' not in volume_analyses[symbol]:
                risk_integrations[symbol] = self.integrate_with_risk_management(
                    volume_analyses[symbol], correlation_analysis, symbol
                )
        
        # Compilar resultados
        complete_analysis = {
            'timestamp': datetime.now().isoformat(),
            'analyzed_symbols': symbols,
            'volume_analyses': volume_analyses,
            'correlation_analysis': correlation_analysis,
            'risk_integrations': risk_integrations
        }
        
        # Gerar relatório
        report = self.generate_comprehensive_report(complete_analysis)
        complete_analysis['report'] = report
        
        self.logger.info("Análise completa concluída com sucesso")
        return complete_analysis


def main():
    """Função principal para demonstração"""
    print("🚀 Iniciando Demonstração dos Sistemas de Análise Avançada")
    print("=" * 70)
    
    try:
        # Inicializar integrador
        integrator = AdvancedAnalyticsIntegrator()
        
        # Executar análise completa
        print("\n📈 Executando análise completa...")
        results = integrator.run_complete_analysis(['EURUSD', 'GBPUSD', 'USDJPY'])
        
        # Exibir relatório
        print("\n" + results['report'])
        
        # Salvar resultados detalhados
        output_file = 'advanced_analytics_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            # Remover o relatório do JSON (muito longo)
            json_results = {k: v for k, v in results.items() if k != 'report'}
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n💾 Resultados detalhados salvos em: {output_file}")
        
        # Estatísticas finais
        volume_count = len([a for a in results['volume_analyses'].values() if 'error' not in a])
        correlation_pairs = len(results['correlation_analysis'].get('correlation_matrix', {}).get('significant_pairs', []))
        risk_integrations = len([r for r in results['risk_integrations'].values() if 'error' not in r])
        
        print("\n📊 ESTATÍSTICAS DA ANÁLISE:")
        print(f"  ✅ Análises de Volume Profile: {volume_count}")
        print(f"  ✅ Pares de Correlação Significativos: {correlation_pairs}")
        print(f"  ✅ Integrações de Risco: {risk_integrations}")
        
        print("\n🎯 Demonstração concluída com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()