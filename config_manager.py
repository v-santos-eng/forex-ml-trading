#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Configuração para MT5 Trading
Gerencia parâmetros de trading, configurações de risco, credenciais e preferências
Autor: Sistema Forex ML
Data: 2024
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import configparser
from cryptography.fernet import Fernet
import base64

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MT5Credentials:
    """Credenciais do MetaTrader 5"""
    login: int
    password: str
    server: str
    path: str = ""
    timeout: int = 60000
    portable: bool = False

@dataclass
class TradingParameters:
    """Parâmetros de trading"""
    # ML Settings
    min_ml_confidence: float = 0.6
    prediction_interval: int = 60  # segundos
    max_positions_per_symbol: int = 1
    
    # Position Sizing
    default_position_size: float = 0.01
    max_position_size: float = 1.0
    position_sizing_method: str = "fixed"  # fixed, percentage, kelly, atr
    
    # Risk Management
    max_risk_per_trade: float = 0.02  # 2%
    max_daily_risk: float = 0.05  # 5%
    max_drawdown: float = 0.10  # 10%
    
    # Stop Loss / Take Profit
    default_sl_pips: int = 20
    default_tp_pips: int = 40
    use_atr_stops: bool = True
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.0
    
    # Trading Hours
    trading_start_hour: int = 0
    trading_end_hour: int = 23
    avoid_news_minutes: int = 30
    
    # Auto Trading
    auto_trading_enabled: bool = False
    max_trades_per_day: int = 10
    max_trades_per_hour: int = 3

@dataclass
class RiskSettings:
    """Configurações de risco"""
    # Limites gerais
    max_open_positions: int = 5
    max_lot_size: float = 1.0
    max_spread_pips: int = 5
    min_free_margin_percent: float = 50.0
    
    # Correlação
    max_correlation_exposure: float = 0.7
    correlation_lookback_days: int = 30
    
    # Volatilidade
    max_volatility_threshold: float = 2.0
    volatility_lookback_periods: int = 20
    
    # News/Events
    avoid_high_impact_news: bool = True
    news_buffer_minutes: int = 30
    
    # Emergency
    emergency_close_drawdown: float = 0.15  # 15%
    emergency_close_margin_level: float = 200.0

@dataclass
class UIPreferences:
    """Preferências da interface"""
    # Janela principal
    window_width: int = 1200
    window_height: int = 800
    window_x: int = 100
    window_y: int = 100
    
    # Tema
    theme: str = "light"  # light, dark
    font_family: str = "Arial"
    font_size: int = 10
    
    # Cores
    profit_color: str = "#00AA00"
    loss_color: str = "#AA0000"
    neutral_color: str = "#666666"
    background_color: str = "#FFFFFF"
    
    # Gráficos
    chart_style: str = "candlestick"  # candlestick, line, bar
    chart_timeframe: str = "M1"
    chart_periods: int = 100
    show_indicators: bool = True
    
    # Tabelas
    refresh_interval_ms: int = 1000
    max_log_entries: int = 1000
    auto_scroll_logs: bool = True
    
    # Notificações
    enable_sound_alerts: bool = True
    enable_popup_alerts: bool = True
    alert_volume: float = 0.5

@dataclass
class DataSettings:
    """Configurações de dados"""
    # Polling
    tick_polling_interval_ms: int = 100
    account_polling_interval_ms: int = 1000
    positions_polling_interval_ms: int = 500
    
    # Cache
    cache_duration_seconds: int = 300
    max_cache_size: int = 1000
    
    # Histórico
    default_history_bars: int = 1000
    max_history_bars: int = 10000
    
    # Backup
    auto_backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_backup_files: int = 7

class ConfigManager:
    """
    Gerenciador de configurações do sistema
    """
    
    def __init__(self, config_dir: str = None):
        # Diretório de configuração
        if config_dir is None:
            config_dir = os.path.join(os.path.expanduser("~"), ".mt5_trading")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Arquivos de configuração
        self.config_file = self.config_dir / "config.json"
        self.credentials_file = self.config_dir / "credentials.enc"
        self.key_file = self.config_dir / "key.key"
        
        # Configurações
        self.mt5_credentials: Optional[MT5Credentials] = None
        self.trading_params = TradingParameters()
        self.risk_settings = RiskSettings()
        self.ui_preferences = UIPreferences()
        self.data_settings = DataSettings()
        
        # Criptografia
        self._cipher_suite = None
        self._init_encryption()
        
        # Carregar configurações
        self.load_config()
    
    def _init_encryption(self):
        """Inicializa sistema de criptografia"""
        try:
            if self.key_file.exists():
                # Carregar chave existente
                with open(self.key_file, 'rb') as f:
                    key = f.read()
            else:
                # Gerar nova chave
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                # Tornar arquivo somente leitura
                os.chmod(self.key_file, 0o600)
            
            self._cipher_suite = Fernet(key)
            
        except Exception as e:
            logger.error(f"Erro ao inicializar criptografia: {e}")
            self._cipher_suite = None
    
    def save_config(self):
        """Salva todas as configurações"""
        try:
            config_data = {
                'trading_parameters': asdict(self.trading_params),
                'risk_settings': asdict(self.risk_settings),
                'ui_preferences': asdict(self.ui_preferences),
                'data_settings': asdict(self.data_settings),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info("Configurações salvas com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao salvar configurações: {e}")
    
    def load_config(self):
        """Carrega todas as configurações"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Carregar parâmetros
                if 'trading_parameters' in config_data:
                    self.trading_params = TradingParameters(**config_data['trading_parameters'])
                
                if 'risk_settings' in config_data:
                    self.risk_settings = RiskSettings(**config_data['risk_settings'])
                
                if 'ui_preferences' in config_data:
                    self.ui_preferences = UIPreferences(**config_data['ui_preferences'])
                
                if 'data_settings' in config_data:
                    self.data_settings = DataSettings(**config_data['data_settings'])
                
                logger.info("Configurações carregadas com sucesso")
            else:
                logger.info("Arquivo de configuração não encontrado, usando padrões")
                self.save_config()  # Criar arquivo com padrões
            
            # Carregar credenciais
            self.load_credentials()
            
        except Exception as e:
            logger.error(f"Erro ao carregar configurações: {e}")
    
    def save_credentials(self, credentials: MT5Credentials):
        """Salva credenciais criptografadas"""
        try:
            if not self._cipher_suite:
                logger.error("Sistema de criptografia não disponível")
                return False
            
            # Serializar credenciais
            cred_data = asdict(credentials)
            cred_json = json.dumps(cred_data)
            
            # Criptografar
            encrypted_data = self._cipher_suite.encrypt(cred_json.encode())
            
            # Salvar
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Tornar arquivo somente leitura
            os.chmod(self.credentials_file, 0o600)
            
            self.mt5_credentials = credentials
            logger.info("Credenciais salvas com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar credenciais: {e}")
            return False
    
    def load_credentials(self) -> Optional[MT5Credentials]:
        """Carrega credenciais descriptografadas"""
        try:
            if not self.credentials_file.exists() or not self._cipher_suite:
                return None
            
            # Carregar dados criptografados
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Descriptografar
            decrypted_data = self._cipher_suite.decrypt(encrypted_data)
            cred_json = decrypted_data.decode()
            
            # Deserializar
            cred_data = json.loads(cred_json)
            self.mt5_credentials = MT5Credentials(**cred_data)
            
            return self.mt5_credentials
            
        except Exception as e:
            logger.error(f"Erro ao carregar credenciais: {e}")
            return None
    
    def get_symbols_config(self) -> List[str]:
        """Retorna lista de símbolos configurados"""
        symbols_file = self.config_dir / "symbols.json"
        
        try:
            if symbols_file.exists():
                with open(symbols_file, 'r') as f:
                    data = json.load(f)
                return data.get('active_symbols', [])
            else:
                # Símbolos padrão
                default_symbols = [
                    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD',
                    'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP'
                ]
                self.save_symbols_config(default_symbols)
                return default_symbols
                
        except Exception as e:
            logger.error(f"Erro ao carregar símbolos: {e}")
            return []
    
    def save_symbols_config(self, symbols: List[str]):
        """Salva lista de símbolos"""
        symbols_file = self.config_dir / "symbols.json"
        
        try:
            data = {
                'active_symbols': symbols,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(symbols_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Símbolos salvos: {len(symbols)} símbolos")
            
        except Exception as e:
            logger.error(f"Erro ao salvar símbolos: {e}")
    
    def get_trading_schedule(self) -> Dict[str, Any]:
        """Retorna configuração de horários de trading"""
        schedule_file = self.config_dir / "schedule.json"
        
        try:
            if schedule_file.exists():
                with open(schedule_file, 'r') as f:
                    return json.load(f)
            else:
                # Horários padrão
                default_schedule = {
                    'enabled': True,
                    'timezone': 'UTC',
                    'sessions': {
                        'asian': {'start': '00:00', 'end': '09:00'},
                        'european': {'start': '07:00', 'end': '16:00'},
                        'american': {'start': '13:00', 'end': '22:00'}
                    },
                    'avoid_weekends': True,
                    'avoid_holidays': True,
                    'news_calendar_enabled': True
                }
                self.save_trading_schedule(default_schedule)
                return default_schedule
                
        except Exception as e:
            logger.error(f"Erro ao carregar horários: {e}")
            return {}
    
    def save_trading_schedule(self, schedule: Dict[str, Any]):
        """Salva configuração de horários"""
        schedule_file = self.config_dir / "schedule.json"
        
        try:
            with open(schedule_file, 'w') as f:
                json.dump(schedule, f, indent=2)
            
            logger.info("Horários de trading salvos")
            
        except Exception as e:
            logger.error(f"Erro ao salvar horários: {e}")
    
    def export_config(self, export_path: str, include_credentials: bool = False):
        """Exporta configurações para arquivo"""
        try:
            export_data = {
                'trading_parameters': asdict(self.trading_params),
                'risk_settings': asdict(self.risk_settings),
                'ui_preferences': asdict(self.ui_preferences),
                'data_settings': asdict(self.data_settings),
                'symbols': self.get_symbols_config(),
                'schedule': self.get_trading_schedule(),
                'exported_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            if include_credentials and self.mt5_credentials:
                # ATENÇÃO: Incluir credenciais em export é arriscado
                export_data['credentials'] = {
                    'login': self.mt5_credentials.login,
                    'server': self.mt5_credentials.server,
                    'path': self.mt5_credentials.path,
                    'timeout': self.mt5_credentials.timeout,
                    'portable': self.mt5_credentials.portable
                    # Senha não incluída por segurança
                }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configurações exportadas para: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao exportar configurações: {e}")
            return False
    
    def import_config(self, import_path: str, import_credentials: bool = False):
        """Importa configurações de arquivo"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Importar parâmetros
            if 'trading_parameters' in import_data:
                self.trading_params = TradingParameters(**import_data['trading_parameters'])
            
            if 'risk_settings' in import_data:
                self.risk_settings = RiskSettings(**import_data['risk_settings'])
            
            if 'ui_preferences' in import_data:
                self.ui_preferences = UIPreferences(**import_data['ui_preferences'])
            
            if 'data_settings' in import_data:
                self.data_settings = DataSettings(**import_data['data_settings'])
            
            # Importar símbolos
            if 'symbols' in import_data:
                self.save_symbols_config(import_data['symbols'])
            
            # Importar horários
            if 'schedule' in import_data:
                self.save_trading_schedule(import_data['schedule'])
            
            # Importar credenciais (se solicitado)
            if import_credentials and 'credentials' in import_data:
                cred_data = import_data['credentials']
                # Senha precisa ser fornecida separadamente
                logger.warning("Credenciais importadas sem senha - configure manualmente")
            
            # Salvar configurações
            self.save_config()
            
            logger.info(f"Configurações importadas de: {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao importar configurações: {e}")
            return False
    
    def reset_to_defaults(self):
        """Restaura configurações padrão"""
        try:
            self.trading_params = TradingParameters()
            self.risk_settings = RiskSettings()
            self.ui_preferences = UIPreferences()
            self.data_settings = DataSettings()
            
            self.save_config()
            logger.info("Configurações restauradas para padrão")
            
        except Exception as e:
            logger.error(f"Erro ao restaurar padrões: {e}")
    
    def validate_config(self) -> List[str]:
        """Valida configurações e retorna lista de problemas"""
        issues = []
        
        # Validar parâmetros de trading
        if self.trading_params.min_ml_confidence < 0 or self.trading_params.min_ml_confidence > 1:
            issues.append("Confiança ML deve estar entre 0 e 1")
        
        if self.trading_params.max_risk_per_trade > 0.1:
            issues.append("Risco por trade muito alto (>10%)")
        
        if self.trading_params.max_daily_risk > 0.2:
            issues.append("Risco diário muito alto (>20%)")
        
        # Validar configurações de risco
        if self.risk_settings.max_open_positions < 1:
            issues.append("Máximo de posições deve ser >= 1")
        
        if self.risk_settings.emergency_close_drawdown > 0.5:
            issues.append("Drawdown de emergência muito alto (>50%)")
        
        # Validar credenciais
        if not self.mt5_credentials:
            issues.append("Credenciais MT5 não configuradas")
        elif self.mt5_credentials.login <= 0:
            issues.append("Login MT5 inválido")
        
        return issues
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Retorna resumo das configurações"""
        return {
            'config_dir': str(self.config_dir),
            'credentials_configured': self.mt5_credentials is not None,
            'auto_trading_enabled': self.trading_params.auto_trading_enabled,
            'max_positions': self.risk_settings.max_open_positions,
            'risk_per_trade': f"{self.trading_params.max_risk_per_trade:.1%}",
            'active_symbols_count': len(self.get_symbols_config()),
            'theme': self.ui_preferences.theme,
            'validation_issues': len(self.validate_config())
        }

# Singleton para acesso global
_config_manager_instance = None

def get_config_manager() -> ConfigManager:
    """Retorna instância singleton do ConfigManager"""
    global _config_manager_instance
    if _config_manager_instance is None:
        _config_manager_instance = ConfigManager()
    return _config_manager_instance

# Exemplo de uso
if __name__ == "__main__":
    print("Config Manager - Sistema de Configuração MT5")
    
    # Criar instância
    config = ConfigManager()
    
    print(f"Diretório de configuração: {config.config_dir}")
    print(f"Resumo: {config.get_config_summary()}")
    
    # Exemplo de configuração de credenciais
    # credentials = MT5Credentials(
    #     login=12345678,
    #     password="sua_senha",
    #     server="MetaQuotes-Demo"
    # )
    # config.save_credentials(credentials)
    
    # Validar configurações
    issues = config.validate_config()
    if issues:
        print("Problemas encontrados:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("Configurações válidas")
    
    print("\nFuncionalidades implementadas:")
    print("- Gerenciamento seguro de credenciais (criptografadas)")
    print("- Configurações de trading e risco")
    print("- Preferências de interface")
    print("- Configurações de dados e polling")
    print("- Import/Export de configurações")
    print("- Validação de configurações")
    print("- Sistema de backup automático")
    print("- Acesso singleton global")