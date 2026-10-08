#!/usr/bin/env python3
"""
Script de Validação das Configurações do Sistema de Risco Avançado
Verifica se todas as configurações estão alinhadas com os requisitos de segurança e desempenho
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

class AdvancedRiskConfigValidator:
    """Validador de configurações do sistema de risco avançado"""
    
    def __init__(self, config_path: str = "auto_trader_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.validation_results = []
        self.warnings = []
        self.errors = []
    
    def _load_config(self) -> Dict:
        """Carrega o arquivo de configuração"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Erro ao carregar configuração: {e}")
            sys.exit(1)
    
    def validate_security_requirements(self) -> bool:
        """Valida requisitos de segurança"""
        print("\n🔒 VALIDAÇÃO DE SEGURANÇA")
        print("=" * 50)
        
        security_passed = True
        
        # 1. Verificar se SL/TP obrigatórios estão ativados
        enforce_sl_tp = self.config.get('enforce_sl_tp', False)
        if enforce_sl_tp:
            print("✅ SL/TP obrigatórios: ATIVADO")
        else:
            print("❌ SL/TP obrigatórios: DESATIVADO")
            self.errors.append("SL/TP obrigatórios devem estar ativados")
            security_passed = False
        
        # 2. Verificar limites de risco por trade
        max_risk = self.config.get('max_risk_per_trade', 0.02)
        if max_risk <= 0.02:  # Máximo 2% por trade
            print(f"✅ Risco máximo por trade: {max_risk*100:.1f}% (dentro do limite)")
        else:
            print(f"⚠️ Risco máximo por trade: {max_risk*100:.1f}% (acima do recomendado)")
            self.warnings.append(f"Risco por trade ({max_risk*100:.1f}%) acima do recomendado (2%)")
        
        # 3. Verificar limites de trades diários
        max_daily = self.config.get('max_daily_trades', 10)
        if max_daily <= 20:  # Máximo razoável
            print(f"✅ Trades diários máximos: {max_daily} (dentro do limite)")
        else:
            print(f"⚠️ Trades diários máximos: {max_daily} (pode ser excessivo)")
            self.warnings.append(f"Limite de trades diários ({max_daily}) pode ser excessivo")
        
        # 4. Verificar configurações de SL/TP mínimos
        min_sl = self.config.get('min_sl_pips', 20)
        min_tp = self.config.get('min_tp_pips', 30)
        
        if min_sl >= 15:  # SL mínimo razoável
            print(f"✅ SL mínimo: {min_sl} pips (adequado)")
        else:
            print(f"❌ SL mínimo: {min_sl} pips (muito baixo)")
            self.errors.append(f"SL mínimo ({min_sl} pips) muito baixo, recomendado >= 15 pips")
            security_passed = False
        
        if min_tp >= 20:  # TP mínimo razoável
            print(f"✅ TP mínimo: {min_tp} pips (adequado)")
        else:
            print(f"❌ TP mínimo: {min_tp} pips (muito baixo)")
            self.errors.append(f"TP mínimo ({min_tp} pips) muito baixo, recomendado >= 20 pips")
            security_passed = False
        
        # 5. Verificar proteção contra gaps
        advanced_risk = self.config.get('advanced_risk', {})
        gap_protection = advanced_risk.get('gap_protection', {})
        
        if gap_protection.get('enabled', False):
            print("✅ Proteção contra gaps: ATIVADA")
            weekend_protection = gap_protection.get('weekend_protection', False)
            if weekend_protection:
                print("✅ Proteção de fim de semana: ATIVADA")
            else:
                print("⚠️ Proteção de fim de semana: DESATIVADA")
                self.warnings.append("Proteção de fim de semana recomendada")
        else:
            print("⚠️ Proteção contra gaps: DESATIVADA")
            self.warnings.append("Proteção contra gaps recomendada")
        
        return security_passed
    
    def validate_performance_requirements(self) -> bool:
        """Valida requisitos de desempenho"""
        print("\n⚡ VALIDAÇÃO DE DESEMPENHO")
        print("=" * 50)
        
        performance_passed = True
        
        # 1. Verificar se sistema avançado está ativado
        advanced_risk = self.config.get('advanced_risk', {})
        advanced_enabled = advanced_risk.get('enabled', False)
        
        if advanced_enabled:
            print("✅ Sistema de Risco Avançado: ATIVADO")
            
            # 2. Verificar parâmetros ATR
            atr_params = advanced_risk.get('atr_params', {})
            atr_period = atr_params.get('period', 14)
            
            if 10 <= atr_period <= 21:
                print(f"✅ Período ATR: {atr_period} (otimizado)")
            else:
                print(f"⚠️ Período ATR: {atr_period} (fora do range otimizado 10-21)")
                self.warnings.append(f"Período ATR ({atr_period}) fora do range otimizado")
            
            # 3. Verificar multiplicadores SL/TP
            sl_mult = atr_params.get('multiplier_sl', 2.0)
            tp_mult = atr_params.get('multiplier_tp', 3.0)
            
            if 1.5 <= sl_mult <= 3.0:
                print(f"✅ Multiplicador SL: {sl_mult} (adequado)")
            else:
                print(f"⚠️ Multiplicador SL: {sl_mult} (fora do range recomendado 1.5-3.0)")
                self.warnings.append(f"Multiplicador SL ({sl_mult}) fora do range recomendado")
            
            if 2.0 <= tp_mult <= 4.0:
                print(f"✅ Multiplicador TP: {tp_mult} (adequado)")
            else:
                print(f"⚠️ Multiplicador TP: {tp_mult} (fora do range recomendado 2.0-4.0)")
                self.warnings.append(f"Multiplicador TP ({tp_mult}) fora do range recomendado")
            
            # 4. Verificar trailing stop
            trailing_params = advanced_risk.get('trailing_stop_params', {})
            trailing_enabled = trailing_params.get('enabled', False)
            
            if trailing_enabled:
                print("✅ Trailing Stop: ATIVADO")
                activation_dist = trailing_params.get('activation_distance', 1.5)
                trail_dist = trailing_params.get('trail_distance', 1.0)
                
                if 1.0 <= activation_dist <= 2.5:
                    print(f"✅ Distância de ativação: {activation_dist} (adequada)")
                else:
                    print(f"⚠️ Distância de ativação: {activation_dist} (fora do range 1.0-2.5)")
                    self.warnings.append(f"Distância de ativação trailing stop fora do range")
            else:
                print("⚠️ Trailing Stop: DESATIVADO")
                self.warnings.append("Trailing Stop recomendado para otimização")
            
            # 5. Verificar take profit escalonado
            scaled_tp = advanced_risk.get('scaled_tp', {})
            scaled_enabled = scaled_tp.get('enabled', False)
            
            if scaled_enabled:
                print("✅ Take Profit Escalonado: ATIVADO")
                levels = scaled_tp.get('levels', [])
                percentages = scaled_tp.get('percentages', [])
                
                if len(levels) == len(percentages) and len(levels) >= 2:
                    print(f"✅ Níveis TP: {levels} com distribuição {percentages}")
                else:
                    print(f"❌ Configuração TP escalonado inválida")
                    self.errors.append("Configuração de TP escalonado inválida")
                    performance_passed = False
            else:
                print("⚠️ Take Profit Escalonado: DESATIVADO")
                self.warnings.append("TP escalonado pode melhorar performance")
        
        else:
            print("❌ Sistema de Risco Avançado: DESATIVADO")
            self.errors.append("Sistema de risco avançado deve estar ativado")
            performance_passed = False
        
        return performance_passed
    
    def validate_adoption_optimization(self) -> bool:
        """Valida otimizações para taxa de adoção"""
        print("\n📈 VALIDAÇÃO DE OTIMIZAÇÃO DE ADOÇÃO")
        print("=" * 50)
        
        adoption_passed = True
        
        # 1. Verificar confiança mínima
        min_confidence = self.config.get('min_confidence', 0.65)
        
        if 0.4 <= min_confidence <= 0.6:
            print(f"✅ Confiança mínima: {min_confidence*100:.0f}% (otimizada para adoção)")
        elif min_confidence > 0.6:
            print(f"⚠️ Confiança mínima: {min_confidence*100:.0f}% (pode reduzir adoção)")
            self.warnings.append(f"Confiança mínima alta ({min_confidence*100:.0f}%) pode reduzir adoção")
        else:
            print(f"❌ Confiança mínima: {min_confidence*100:.0f}% (muito baixa, risco elevado)")
            self.errors.append(f"Confiança mínima ({min_confidence*100:.0f}%) muito baixa")
            adoption_passed = False
        
        # 2. Verificar ajustes de volatilidade
        advanced_risk = self.config.get('advanced_risk', {})
        vol_params = advanced_risk.get('volatility_params', {})
        
        if vol_params.get('regime_detection', False):
            print("✅ Detecção de regime de volatilidade: ATIVADA")
            
            # Verificar fatores de ajuste
            adj_factors = vol_params.get('adjustment_factors', {})
            if adj_factors:
                low_factor = adj_factors.get('low', 1.0)
                high_factor = adj_factors.get('high', 1.0)
                
                if 0.6 <= low_factor <= 0.8 and 1.2 <= high_factor <= 1.6:
                    print(f"✅ Fatores de ajuste: Baixa vol {low_factor}, Alta vol {high_factor}")
                else:
                    print(f"⚠️ Fatores de ajuste podem ser otimizados")
                    self.warnings.append("Fatores de ajuste de volatilidade podem ser otimizados")
        else:
            print("⚠️ Detecção de regime de volatilidade: DESATIVADA")
            self.warnings.append("Detecção de volatilidade recomendada para otimização")
        
        # 3. Verificar símbolos para trading
        symbols = self.config.get('symbols', [])
        recommended_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        
        if len(symbols) >= 3:
            print(f"✅ Símbolos configurados: {len(symbols)} ({', '.join(symbols)})")
            
            # Verificar se inclui pares principais
            major_pairs = [s for s in symbols if s in ['EURUSD', 'GBPUSD', 'USDJPY']]
            if len(major_pairs) >= 2:
                print(f"✅ Pares principais incluídos: {', '.join(major_pairs)}")
            else:
                print(f"⚠️ Poucos pares principais configurados")
                self.warnings.append("Recomendado incluir mais pares principais (EURUSD, GBPUSD, USDJPY)")
        else:
            print(f"❌ Poucos símbolos configurados: {len(symbols)}")
            self.errors.append("Recomendado configurar pelo menos 3 símbolos")
            adoption_passed = False
        
        return adoption_passed
    
    def generate_recommendations(self) -> List[str]:
        """Gera recomendações baseadas na validação"""
        recommendations = []
        
        if self.errors:
            recommendations.append("AÇÕES CRÍTICAS NECESSÁRIAS:")
            for error in self.errors:
                recommendations.append(f"  • {error}")
        
        if self.warnings:
            recommendations.append("\nOTIMIZAÇÕES RECOMENDADAS:")
            for warning in self.warnings:
                recommendations.append(f"  • {warning}")
        
        # Recomendações específicas para taxa de adoção
        recommendations.extend([
            "\nRECOMENDAÇÕES PARA MELHORAR TAXA DE ADOÇÃO:",
            "  • Manter confiança mínima entre 45-55% para equilibrar qualidade e quantidade",
            "  • Ativar trailing stop para maximizar lucros",
            "  • Usar TP escalonado para capturar movimentos maiores",
            "  • Configurar detecção de volatilidade para ajustes dinâmicos",
            "  • Monitorar performance regularmente e ajustar parâmetros"
        ])
        
        return recommendations
    
    def run_validation(self) -> bool:
        """Executa validação completa"""
        print("🔍 VALIDAÇÃO DO SISTEMA DE RISCO AVANÇADO")
        print("=" * 60)
        print(f"Arquivo de configuração: {self.config_path}")
        
        # Executar validações
        security_ok = self.validate_security_requirements()
        performance_ok = self.validate_performance_requirements()
        adoption_ok = self.validate_adoption_optimization()
        
        # Resumo final
        print("\n📊 RESUMO DA VALIDAÇÃO")
        print("=" * 50)
        
        overall_status = security_ok and performance_ok and adoption_ok
        
        print(f"Segurança: {'✅ APROVADO' if security_ok else '❌ REPROVADO'}")
        print(f"Desempenho: {'✅ APROVADO' if performance_ok else '❌ REPROVADO'}")
        print(f"Otimização: {'✅ APROVADO' if adoption_ok else '❌ REPROVADO'}")
        print(f"\nStatus Geral: {'✅ SISTEMA VALIDADO' if overall_status else '❌ CORREÇÕES NECESSÁRIAS'}")
        
        # Estatísticas
        print(f"\nEstatísticas:")
        print(f"  • Erros críticos: {len(self.errors)}")
        print(f"  • Avisos: {len(self.warnings)}")
        
        # Gerar recomendações
        recommendations = self.generate_recommendations()
        if recommendations:
            print("\n💡 RECOMENDAÇÕES")
            print("=" * 50)
            for rec in recommendations:
                print(rec)
        
        return overall_status

def main():
    """Função principal"""
    validator = AdvancedRiskConfigValidator()
    
    try:
        success = validator.run_validation()
        
        if success:
            print("\n🎉 Configuração validada com sucesso!")
            print("Sistema de risco avançado pronto para uso.")
            return 0
        else:
            print("\n⚠️ Configuração requer correções antes do uso.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Erro durante validação: {e}")
        return 1

if __name__ == "__main__":
    exit(main())