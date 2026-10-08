#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Validação da Integração do Sistema Avançado de Risco
Testa todas as funcionalidades integradas no auto_trader
Autor: Sistema Forex ML
Data: 2024
"""

import json
import sqlite3
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_risk_system import (
        AdvancedRiskSystem, ATRParameters, VolatilityParameters, 
        TrailingStopParameters, GapProtectionParameters, TechnicalLevelsParameters,
        ScaledTakeProfitParameters, ATRMethod, VolatilityRegime, TrailingStopType
    )
    from detailed_decision_logger import DetailedDecisionLogger, AdvancedRiskData
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    sys.exit(1)

class IntegrationValidator:
    """Validador da integração do sistema avançado"""
    
    def __init__(self):
        self.results = []
        self.errors = []
        
    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Registra resultado de um teste"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = f"{status} {test_name}"
        if details:
            result += f" - {details}"
        
        self.results.append(result)
        print(result)
        
        if not success:
            self.errors.append(f"{test_name}: {details}")
    
    def test_config_loading(self) -> bool:
        """Testa carregamento da configuração avançada"""
        try:
            with open('auto_trader_config.json', 'r') as f:
                config = json.load(f)
            
            # Verificar se seção advanced_risk existe
            if 'advanced_risk' not in config:
                self.log_result("Config Loading", False, "Seção 'advanced_risk' não encontrada")
                return False
            
            advanced_config = config['advanced_risk']
            required_sections = ['atr_params', 'volatility_params', 'trailing_stop_params', 
                               'gap_protection', 'technical_levels', 'scaled_tp']
            
            for section in required_sections:
                if section not in advanced_config:
                    self.log_result("Config Loading", False, f"Seção '{section}' não encontrada")
                    return False
            
            self.log_result("Config Loading", True, "Todas as seções de configuração presentes")
            return True
            
        except Exception as e:
            self.log_result("Config Loading", False, str(e))
            return False
    
    def test_advanced_risk_system(self) -> bool:
        """Testa inicialização do sistema avançado de risco"""
        try:
            # Parâmetros de teste
            atr_params = ATRParameters(
                period=14,
                method=ATRMethod.WILDER,
                volatility_adjustment=True,
                session_adjustment=True
            )
            
            volatility_params = VolatilityParameters(
                lookback_period=20,
                regime_thresholds={'low': 0.5, 'normal': 1.0, 'high': 1.5, 'extreme': 2.0},
                adjustment_factors={'low': 0.8, 'normal': 1.0, 'high': 1.2, 'extreme': 1.5}
            )
            
            # Inicializar sistema
            trailing_params = TrailingStopParameters(
                type=TrailingStopType.ATR_BASED,
                activation_distance=1.5,
                trail_distance=1.0
            )
            
            gap_protection_params = GapProtectionParameters(
                enabled=True,
                max_gap_threshold=0.005
            )
            
            technical_params = TechnicalLevelsParameters(
                support_resistance_period=50,
                psychological_levels=True
            )
            
            scaled_tp_params = ScaledTakeProfitParameters(
                enabled=True,
                levels=[1.0, 2.0, 3.0],
                position_percentages=[0.3, 0.4, 0.3]
            )
            
            risk_system = AdvancedRiskSystem(
                mt5_client=None,  # Mock para teste
                atr_params=atr_params,
                volatility_params=volatility_params,
                trailing_params=trailing_params,
                gap_protection_params=gap_protection_params,
                technical_params=technical_params,
                scaled_tp_params=scaled_tp_params
            )
            
            self.log_result("Advanced Risk System Init", True, "Sistema inicializado com sucesso")
            return True
            
        except Exception as e:
            self.log_result("Advanced Risk System Init", False, str(e))
            return False
    
    def test_detailed_logger(self) -> bool:
        """Testa sistema de logging detalhado"""
        try:
            # Remover banco de teste se existir
            test_db = "test_decisions.db"
            if os.path.exists(test_db):
                os.remove(test_db)
            
            # Inicializar logger
            logger = DetailedDecisionLogger(test_db)
            
            # Testar log de decisão avançada
            atr_data = {
                'method': 'wilder',
                'value': 0.00125,
                'volatility_regime': 'normal',
                'gap_protection': True,
                'technical_levels': True,
                'trailing_stop': False,
                'metadata': {'test': True}
            }
            
            sl_tp_data = {
                'sl_multiplier': 2.0,
                'tp_multiplier': 3.0,
                'risk_reward_ratio': 1.5,
                'sl_price': 1.08500,
                'tp_price': 1.09000,
                'entry_price': 1.08750
            }
            
            market_data = {
                'symbol': 'EURUSD',
                'bid': 1.08745,
                'ask': 1.08755,
                'spread': 0.00010,
                'direction': 'BUY'
            }
            
            decision_id = logger.log_advanced_risk_decision(
                symbol="EURUSD",
                action="TEST_ADVANCED_CALCULATION",
                confidence=0.75,
                atr_data=atr_data,
                sl_tp_data=sl_tp_data,
                market_data=market_data
            )
            
            if decision_id:
                self.log_result("Detailed Logger", True, f"Decisão registrada: {decision_id[:8]}...")
                
                # Limpar arquivo de teste
                try:
                    os.remove("test_decisions.db")
                except:
                    pass
                    
                return True
            else:
                self.log_result("Detailed Logger", False, "Falha ao registrar decisão")
                return False
                
        except Exception as e:
            self.log_result("Detailed Logger", False, str(e))
            return False
    
    def test_database_schema(self) -> bool:
        """Testa schema do banco de dados"""
        try:
            # Criar um banco de teste para verificar o schema
            test_db = "schema_test.db"
            if os.path.exists(test_db):
                os.remove(test_db)
            
            # Inicializar logger para criar o schema
            from detailed_decision_logger import DetailedDecisionLogger
            logger = DetailedDecisionLogger(test_db)
            
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()
            
            # Verificar estrutura da tabela
            cursor.execute("PRAGMA table_info(detailed_decisions)")
            columns = [row[1] for row in cursor.fetchall()]
            
            required_columns = [
                'id', 'timestamp', 'symbol', 'action', 'confidence',
                'reasoning_factors', 'risk_assessment', 'market_data',
                'technical_analysis', 'ml_predictions', 'bot_action',
                'advanced_risk_data', 'created_at'
            ]
            
            missing_columns = [col for col in required_columns if col not in columns]
            
            conn.close()
            
            # Limpar banco de teste
            try:
                os.remove(test_db)
            except:
                pass
            
            if missing_columns:
                self.log_result("Database Schema", False, f"Colunas ausentes: {missing_columns}")
                return False
            else:
                self.log_result("Database Schema", True, "Schema válido")
                return True
                
        except Exception as e:
            self.log_result("Database Schema", False, str(e))
            return False
    
    def test_import_compatibility(self) -> bool:
        """Testa compatibilidade de imports"""
        try:
            # Testar imports principais
            imports_to_test = [
                "from advanced_risk_system import AdvancedRiskSystem",
                "from detailed_decision_logger import DetailedDecisionLogger, AdvancedRiskData",
                "import MetaTrader5 as mt5",
                "import pandas as pd",
                "import numpy as np"
            ]
            
            for import_stmt in imports_to_test:
                try:
                    exec(import_stmt)
                except ImportError as e:
                    self.log_result("Import Compatibility", False, f"Falha: {import_stmt} - {e}")
                    return False
            
            self.log_result("Import Compatibility", True, "Todos os imports funcionando")
            return True
            
        except Exception as e:
            self.log_result("Import Compatibility", False, str(e))
            return False
    
    def run_validation(self) -> Dict[str, Any]:
        """Executa todos os testes de validação"""
        print("🔍 Iniciando Validação da Integração do Sistema Avançado")
        print("=" * 60)
        
        tests = [
            self.test_import_compatibility,
            self.test_config_loading,
            self.test_advanced_risk_system,
            self.test_detailed_logger,
            self.test_database_schema
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
        
        print("\n" + "=" * 60)
        print(f"📊 Resultados da Validação: {passed}/{total} testes passaram")
        
        if self.errors:
            print("\n❌ Erros encontrados:")
            for error in self.errors:
                print(f"   • {error}")
        
        success_rate = (passed / total) * 100
        status = "✅ SUCESSO" if passed == total else "⚠️ PARCIAL" if passed > 0 else "❌ FALHA"
        
        print(f"\n{status} - Taxa de Sucesso: {success_rate:.1f}%")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': success_rate,
            'status': 'success' if passed == total else 'partial' if passed > 0 else 'failure',
            'errors': self.errors,
            'results': self.results
        }

def main():
    """Função principal"""
    validator = IntegrationValidator()
    results = validator.run_validation()
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"validation_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 Resultados salvos em: {results_file}")
    
    return results['status'] == 'success'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)