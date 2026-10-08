import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from unittest.mock import Mock, patch

# Importar os módulos que vamos testar
from volume_profile_analyzer import (
    VolumeProfileAnalyzer, VolumeProfileParameters, VolumeProfileType,
    VolumeNodeType
)
from correlation_analyzer import (
    CorrelationAnalyzer, CorrelationParameters, CorrelationType,
    CorrelationStrength
)


class TestVolumeProfileAnalyzer(unittest.TestCase):
    """Testes para o analisador de volume profile"""
    
    def setUp(self):
        """Configuração inicial para os testes"""
        self.params = VolumeProfileParameters(
            price_bins=20,
            value_area_percentage=0.70,
            hvn_threshold=0.05,
            lvn_threshold=0.01
        )
        
        self.logger = logging.getLogger('test')
        self.analyzer = VolumeProfileAnalyzer(self.params, self.logger)
        
        # Dados de teste
        self.sample_data = pd.DataFrame({
            'high': [1.1050, 1.1055, 1.1060, 1.1058, 1.1062, 1.1065, 1.1063, 1.1068, 1.1070, 1.1067],
            'low': [1.1045, 1.1048, 1.1052, 1.1054, 1.1058, 1.1060, 1.1059, 1.1063, 1.1065, 1.1062],
            'close': [1.1048, 1.1053, 1.1057, 1.1056, 1.1060, 1.1063, 1.1061, 1.1066, 1.1068, 1.1064],
            'volume': [1000, 1500, 2000, 1200, 800, 1800, 1300, 2200, 1600, 900],
            'timestamp': pd.date_range('2024-01-01 09:00', periods=10, freq='1h')
        })
    
    def test_volume_profile_calculation(self):
        """Testa o cálculo básico do volume profile"""
        profile = self.analyzer.calculate_volume_profile(self.sample_data)
        
        # Verificar estrutura básica
        self.assertIn('volume_profile', profile)
        self.assertIn('value_area', profile)
        self.assertIn('volume_nodes', profile)
        self.assertIn('total_volume', profile)
        
        # Verificar se o volume total está correto
        expected_total_volume = self.sample_data['volume'].sum()
        self.assertEqual(profile['total_volume'], expected_total_volume)
        
        # Verificar se temos nós de volume
        self.assertGreater(len(profile['volume_nodes']), 0)
        
        # Verificar se o POC foi identificado
        poc_nodes = [node for node in profile['volume_nodes'] if node.node_type == VolumeNodeType.POC]
        self.assertEqual(len(poc_nodes), 1)
    
    def test_value_area_calculation(self):
        """Testa o cálculo da área de valor"""
        profile = self.analyzer.calculate_volume_profile(self.sample_data)
        value_area = profile['value_area']
        
        # Verificar se a área de valor foi calculada
        self.assertGreater(value_area.high, 0)
        self.assertGreater(value_area.low, 0)
        self.assertGreater(value_area.poc, 0)
        
        # Verificar se high >= low
        self.assertGreaterEqual(value_area.high, value_area.low)
        
        # Verificar se POC está dentro da área de valor
        self.assertGreaterEqual(value_area.poc, value_area.low)
        self.assertLessEqual(value_area.poc, value_area.high)
        
        # Verificar se a percentagem de volume está próxima do target (70%)
        self.assertGreater(value_area.volume_percentage, 0.6)  # Pelo menos 60%
        self.assertLessEqual(value_area.volume_percentage, 1.0)  # Máximo 100%
    
    def test_support_resistance_levels(self):
        """Testa a extração de níveis de suporte e resistência"""
        profile = self.analyzer.calculate_volume_profile(self.sample_data)
        levels = self.analyzer.get_support_resistance_levels(profile)
        
        # Verificar estrutura
        self.assertIn('support_levels', levels)
        self.assertIn('resistance_levels', levels)
        self.assertIn('key_levels', levels)
        self.assertIn('poc', levels)
        
        # Verificar se temos pelo menos alguns níveis
        total_levels = len(levels['support_levels']) + len(levels['resistance_levels']) + len(levels['key_levels'])
        self.assertGreater(total_levels, 0)
        
        # Verificar se POC é válido
        self.assertGreater(levels['poc'], 0)
    
    def test_price_action_analysis(self):
        """Testa a análise da ação do preço"""
        profile = self.analyzer.calculate_volume_profile(self.sample_data)
        current_price = 1.1065
        
        analysis = self.analyzer.analyze_price_action_at_levels(current_price, profile)
        
        # Verificar estrutura
        self.assertIn('current_price', analysis)
        self.assertIn('price_position', analysis)
        self.assertIn('in_value_area', analysis)
        self.assertIn('volume_context', analysis)
        
        # Verificar se o preço atual está correto
        self.assertEqual(analysis['current_price'], current_price)
        
        # Verificar se a posição do preço foi determinada
        valid_positions = ['above_value_area', 'below_value_area', 'in_value_area', 'unknown']
        self.assertIn(analysis['price_position'], valid_positions)
    
    def test_trading_signals(self):
        """Testa a geração de sinais de trading"""
        profile = self.analyzer.calculate_volume_profile(self.sample_data)
        current_price = 1.1065
        
        signals = self.analyzer.get_trading_signals(current_price, profile, 'bullish')
        
        # Verificar estrutura
        self.assertIn('signal_strength', signals)
        self.assertIn('signal_type', signals)
        self.assertIn('confidence', signals)
        self.assertIn('recommendations', signals)
        
        # Verificar ranges válidos
        self.assertGreaterEqual(signals['signal_strength'], -1.0)
        self.assertLessEqual(signals['signal_strength'], 1.0)
        self.assertGreaterEqual(signals['confidence'], 0.0)
        self.assertLessEqual(signals['confidence'], 1.0)
        
        # Verificar se temos recomendações
        self.assertIsInstance(signals['recommendations'], list)
    
    def test_empty_data_handling(self):
        """Testa o tratamento de dados vazios"""
        empty_data = pd.DataFrame()
        profile = self.analyzer.calculate_volume_profile(empty_data)
        
        # Verificar se retorna estrutura válida mesmo com dados vazios
        self.assertIn('volume_profile', profile)
        self.assertIn('value_area', profile)
        self.assertEqual(profile['total_volume'], 0)
        self.assertEqual(len(profile['volume_nodes']), 0)
    
    def test_session_filtering(self):
        """Testa o filtro de dados por sessão"""
        # Criar dados com timestamps variados
        extended_data = self.sample_data.copy()
        extended_data['timestamp'] = pd.date_range('2024-01-01 06:00', periods=10, freq='2h')
        
        profile = self.analyzer.calculate_volume_profile(
            extended_data, 
            VolumeProfileType.SESSION
        )
        
        # Deve funcionar mesmo com filtro de sessão
        self.assertIn('volume_profile', profile)
        self.assertGreaterEqual(profile['total_volume'], 0)


class TestCorrelationAnalyzer(unittest.TestCase):
    """Testes para o analisador de correlação"""
    
    def setUp(self):
        """Configuração inicial para os testes"""
        self.params = CorrelationParameters(
            lookback_periods=30,
            min_correlation_threshold=0.3,
            rolling_window=10
        )
        
        self.logger = logging.getLogger('test')
        self.analyzer = CorrelationAnalyzer(self.params, self.logger)
        
        # Adicionar dados de teste simulados
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        self._add_simulated_data()
    
    def _add_simulated_data(self):
        """Adiciona dados simulados para teste"""
        np.random.seed(42)  # Para resultados reproduzíveis
        
        base_prices = {'EURUSD': 1.1000, 'GBPUSD': 1.3000, 'USDJPY': 110.00, 'AUDUSD': 0.7500}
        
        for i in range(50):
            timestamp = datetime.now() - timedelta(hours=50-i)
            
            # Gerar mudanças correlacionadas
            eur_change = np.random.normal(0, 0.001)
            
            changes = {
                'EURUSD': eur_change,
                'GBPUSD': eur_change * 0.7 + np.random.normal(0, 0.0008),  # Correlação positiva
                'USDJPY': -eur_change * 0.5 + np.random.normal(0, 0.0012),  # Correlação negativa
                'AUDUSD': eur_change * 0.4 + np.random.normal(0, 0.001)     # Correlação moderada
            }
            
            for asset in self.assets:
                new_price = base_prices[asset] * (1 + changes[asset])
                base_prices[asset] = new_price
                self.analyzer.add_price_data(asset, new_price, timestamp)
    
    def test_correlation_matrix_calculation(self):
        """Testa o cálculo da matriz de correlação"""
        correlation_matrix = self.analyzer.calculate_correlation_matrix(self.assets)
        
        # Verificar estrutura
        self.assertIsNotNone(correlation_matrix)
        self.assertEqual(len(correlation_matrix.assets), len(self.assets))
        self.assertGreaterEqual(len(correlation_matrix.significant_pairs), 0)
        
        # Verificar se a matriz não está vazia
        if not correlation_matrix.matrix.empty:
            # Verificar dimensões da matriz
            self.assertEqual(correlation_matrix.matrix.shape[0], len(correlation_matrix.assets))
            self.assertEqual(correlation_matrix.matrix.shape[1], len(correlation_matrix.assets))
            
            # Verificar diagonal (deve ser 1.0)
            for asset in correlation_matrix.assets:
                self.assertAlmostEqual(correlation_matrix.matrix.loc[asset, asset], 1.0, places=2)
    
    def test_significant_pairs_identification(self):
        """Testa a identificação de pares significativos"""
        correlation_matrix = self.analyzer.calculate_correlation_matrix(self.assets)
        
        # Verificar se temos pares significativos
        significant_pairs = correlation_matrix.significant_pairs
        
        for pair in significant_pairs:
            # Verificar estrutura do par
            self.assertIsInstance(pair.asset1, str)
            self.assertIsInstance(pair.asset2, str)
            self.assertIsInstance(pair.correlation, float)
            self.assertIsInstance(pair.strength, CorrelationStrength)
            
            # Verificar se a correlação está no range válido
            self.assertGreaterEqual(pair.correlation, -1.0)
            self.assertLessEqual(pair.correlation, 1.0)
            
            # Verificar se atende ao threshold mínimo
            self.assertGreaterEqual(abs(pair.correlation), self.params.min_correlation_threshold)
    
    def test_rolling_correlation(self):
        """Testa o cálculo de correlação rolling"""
        rolling_corr = self.analyzer.get_rolling_correlation('EURUSD', 'GBPUSD', window=10)
        
        # Verificar se temos resultados
        self.assertIsInstance(rolling_corr, list)
        
        if rolling_corr:  # Se temos dados suficientes
            for point in rolling_corr:
                self.assertIn('correlation', point)
                self.assertIn('strength', point)
                self.assertGreaterEqual(point['correlation'], -1.0)
                self.assertLessEqual(point['correlation'], 1.0)
    
    def test_correlation_breakdowns(self):
        """Testa a detecção de quebras de correlação"""
        # Primeiro calcular algumas correlações para ter histórico
        self.analyzer.calculate_correlation_matrix(self.assets)
        
        # Adicionar mais dados para criar mudanças
        for i in range(10):
            timestamp = datetime.now() + timedelta(hours=i)
            # Adicionar dados que quebram correlações anteriores
            self.analyzer.add_price_data('EURUSD', 1.1000 + i * 0.001, timestamp)
            self.analyzer.add_price_data('GBPUSD', 1.3000 - i * 0.001, timestamp)  # Inverter correlação
        
        # Recalcular matriz
        self.analyzer.calculate_correlation_matrix(self.assets)
        
        # Detectar quebras
        breakdowns = self.analyzer.detect_correlation_breakdowns(threshold=0.1)
        
        # Verificar estrutura
        self.assertIsInstance(breakdowns, list)
        
        for breakdown in breakdowns:
            self.assertIn('asset1', breakdown)
            self.assertIn('asset2', breakdown)
            self.assertIn('change', breakdown)
            self.assertGreaterEqual(breakdown['change'], 0)
    
    def test_correlation_clusters(self):
        """Testa a identificação de clusters de correlação"""
        correlation_matrix = self.analyzer.calculate_correlation_matrix(self.assets)
        
        if not correlation_matrix.matrix.empty:
            clusters = self.analyzer.get_correlation_clusters(correlation_matrix, min_cluster_size=2)
            
            # Verificar estrutura
            self.assertIsInstance(clusters, list)
            
            for cluster in clusters:
                self.assertIn('assets', cluster)
                self.assertIn('avg_internal_correlation', cluster)
                self.assertGreaterEqual(len(cluster['assets']), 2)
                self.assertGreaterEqual(cluster['avg_internal_correlation'], 0)
    
    def test_trading_insights(self):
        """Testa a geração de insights de trading"""
        correlation_matrix = self.analyzer.calculate_correlation_matrix(self.assets)
        
        if 'EURUSD' in correlation_matrix.assets:
            insights = self.analyzer.generate_trading_insights(correlation_matrix, 'EURUSD')
            
            # Verificar estrutura
            self.assertIn('target_asset', insights)
            self.assertIn('correlations', insights)
            self.assertIn('hedging_opportunities', insights)
            self.assertIn('diversification_suggestions', insights)
            self.assertIn('trading_pairs', insights)
            
            # Verificar se o asset alvo está correto
            self.assertEqual(insights['target_asset'], 'EURUSD')
            
            # Verificar se temos correlações
            self.assertIsInstance(insights['correlations'], dict)
    
    def test_data_export(self):
        """Testa a exportação de dados"""
        correlation_matrix = self.analyzer.calculate_correlation_matrix(self.assets)
        
        # Exportar em JSON
        json_data = self.analyzer.export_correlation_data(correlation_matrix, 'json')
        
        # Verificar se é uma string válida
        self.assertIsInstance(json_data, str)
        self.assertGreater(len(json_data), 10)  # Deve ter conteúdo
        
        # Verificar se é JSON válido
        import json
        try:
            parsed_data = json.loads(json_data)
            self.assertIn('timestamp', parsed_data)
            self.assertIn('assets', parsed_data)
        except json.JSONDecodeError:
            self.fail("Dados exportados não são JSON válido")
    
    def test_data_cleanup(self):
        """Testa a limpeza de dados antigos"""
        # Adicionar dados antigos
        old_timestamp = datetime.now() - timedelta(hours=48)
        self.analyzer.add_price_data('TEST_ASSET', 1.0, old_timestamp)
        
        # Executar limpeza
        self.analyzer.cleanup_old_data(max_age_hours=24)
        
        # Verificar se a limpeza foi executada (não deve gerar erro)
        self.assertTrue(True)  # Se chegou aqui, não houve erro
    
    def test_empty_assets_handling(self):
        """Testa o tratamento de lista vazia de ativos"""
        correlation_matrix = self.analyzer.calculate_correlation_matrix([])
        
        # Verificar se retorna estrutura válida
        self.assertIsNotNone(correlation_matrix)
        self.assertEqual(len(correlation_matrix.assets), 0)
        self.assertEqual(len(correlation_matrix.significant_pairs), 0)


class TestIntegrationAdvancedAnalytics(unittest.TestCase):
    """Testes de integração para os sistemas avançados"""
    
    def setUp(self):
        """Configuração para testes de integração"""
        self.volume_params = VolumeProfileParameters(price_bins=15)
        self.corr_params = CorrelationParameters(lookback_periods=20)
        
        self.logger = logging.getLogger('integration_test')
        
        self.volume_analyzer = VolumeProfileAnalyzer(self.volume_params, self.logger)
        self.correlation_analyzer = CorrelationAnalyzer(self.corr_params, self.logger)
    
    def test_combined_analysis_workflow(self):
        """Testa um fluxo completo de análise combinada"""
        # Dados de exemplo com mais registros para garantir nós de volume
        price_data = pd.DataFrame({
            'high': [1.1050, 1.1055, 1.1060, 1.1058, 1.1062, 1.1065, 1.1063, 1.1068, 1.1070, 1.1067],
            'low': [1.1045, 1.1048, 1.1052, 1.1054, 1.1058, 1.1060, 1.1059, 1.1063, 1.1065, 1.1062],
            'close': [1.1048, 1.1053, 1.1057, 1.1056, 1.1060, 1.1063, 1.1061, 1.1066, 1.1068, 1.1064],
            'volume': [1000, 1500, 2000, 1200, 800, 1800, 1300, 2200, 1600, 900],
            'timestamp': pd.date_range('2024-01-01 09:00', periods=10, freq='1h')
        })
        
        # Análise de volume profile
        volume_profile = self.volume_analyzer.calculate_volume_profile(price_data)
        volume_levels = self.volume_analyzer.get_support_resistance_levels(volume_profile)
        
        # Adicionar dados para análise de correlação
        assets = ['EURUSD', 'GBPUSD']
        for i, asset in enumerate(assets):
            for j, row in price_data.iterrows():
                price = row['close'] * (1 + i * 0.1)  # Diferenciação simples
                self.correlation_analyzer.add_price_data(asset, price, row['timestamp'])
        
        # Análise de correlação
        correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(assets)
        
        # Verificar se ambas as análises funcionaram
        self.assertGreater(len(volume_profile['volume_nodes']), 0)
        self.assertGreater(volume_levels['poc'], 0)
        self.assertIsNotNone(correlation_matrix)
        
        # Verificar se podemos combinar insights
        combined_insights = {
            'volume_analysis': {
                'poc': volume_levels['poc'],
                'value_area_high': volume_levels['value_area_high'],
                'value_area_low': volume_levels['value_area_low'],
                'key_levels': volume_levels['key_levels']
            },
            'correlation_analysis': {
                'assets': correlation_matrix.assets,
                'significant_pairs': len(correlation_matrix.significant_pairs)
            }
        }
        
        # Verificar estrutura combinada
        self.assertIn('volume_analysis', combined_insights)
        self.assertIn('correlation_analysis', combined_insights)
    
    def test_performance_with_large_dataset(self):
        """Testa performance com dataset maior"""
        # Criar dataset maior
        large_data = pd.DataFrame({
            'high': np.random.uniform(1.1000, 1.1100, 200),
            'low': np.random.uniform(1.0900, 1.1000, 200),
            'close': np.random.uniform(1.0950, 1.1050, 200),
            'volume': np.random.randint(500, 3000, 200),
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1H')
        })
        
        # Garantir que high >= low >= close é válido
        for i in range(len(large_data)):
            low = large_data.iloc[i]['low']
            high = max(large_data.iloc[i]['high'], low + 0.0001)
            close = np.random.uniform(low, high)
            
            large_data.iloc[i, large_data.columns.get_loc('high')] = high
            large_data.iloc[i, large_data.columns.get_loc('close')] = close
        
        # Testar análise de volume profile
        start_time = datetime.now()
        volume_profile = self.volume_analyzer.calculate_volume_profile(large_data)
        volume_time = (datetime.now() - start_time).total_seconds()
        
        # Verificar se completou em tempo razoável (< 5 segundos)
        self.assertLess(volume_time, 5.0)
        self.assertGreater(len(volume_profile['volume_nodes']), 0)
        
        # Testar análise de correlação com múltiplos ativos
        assets = ['ASSET1', 'ASSET2', 'ASSET3', 'ASSET4']
        
        start_time = datetime.now()
        for asset in assets:
            for i, row in large_data.iterrows():
                # Adicionar variação por asset
                price = row['close'] * (1 + hash(asset) % 100 / 10000)
                self.correlation_analyzer.add_price_data(asset, price, row['timestamp'])
        
        correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(assets)
        correlation_time = (datetime.now() - start_time).total_seconds()
        
        # Verificar performance
        self.assertLess(correlation_time, 10.0)
        self.assertIsNotNone(correlation_matrix)


if __name__ == '__main__':
    # Configurar logging para os testes
    logging.basicConfig(
        level=logging.WARNING,  # Reduzir verbosidade durante testes
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Executar testes
    unittest.main(verbosity=2)