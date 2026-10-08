import MetaTrader5 as mt5
import hashlib
import base64
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import threading
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
from dataclasses import dataclass
from enum import Enum

# Importar sistema de logging de decisões
from decision_logging_system import (
    get_decision_logger, DecisionType, DecisionLevel,
    log_system_decision
)

class ConnectionStatus(Enum):
    """Status da conexão MT5"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class MT5Credentials:
    """Credenciais seguras para MT5"""
    login: str
    password: str
    server: str = "MetaQuotes-Demo"
    timeout: int = 60000
    portable: bool = False

@dataclass
class ConnectionResult:
    """Resultado da tentativa de conexão"""
    success: bool
    status: ConnectionStatus
    message: str
    account_info: Optional[Dict] = None
    error_code: Optional[int] = None
    connection_time: Optional[float] = None

class MT5SecureAuthenticator:
    """Sistema de autenticação segura para MetaTrader 5"""
    
    def __init__(self, config_file: str = "mt5_config.encrypted"):
        self.config_file = Path(config_file)
        self.logger = logging.getLogger(__name__)
        self.decision_logger = get_decision_logger()
        
        # Estado da conexão
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.last_connection_attempt = None
        self.connection_retries = 0
        self.max_retries = 3
        self.retry_delay = 5  # segundos
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Configurações de segurança
        self.session_timeout = 3600  # 1 hora
        self.last_activity = time.time()
        
        # Credenciais de autenticação fornecidas pelo usuário
        self.default_credentials = MT5Credentials(
            login=os.getenv("MT5_LOGIN", ""),  # Login fornecido pelo usuário
            password=os.getenv("MT5_PASSWORD", ""),  # Senha fornecida pelo usuário
            server="MetaQuotes-Demo"  # Servidor padrão (pode ser configurado)
        )
        
        # Inicializar sistema de criptografia
        self._init_encryption()
        
        # Salvar credenciais criptografadas se não existirem
        if not self.config_file.exists():
            self._save_encrypted_credentials(self.default_credentials)
    
    def _init_encryption(self):
        """Inicializa sistema de criptografia"""
        # Gerar chave baseada em informações do sistema
        system_info = f"{os.environ.get('COMPUTERNAME', 'default')}{os.environ.get('USERNAME', 'user')}"
        salt = b'mt5_secure_salt_2024'
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(system_info.encode()))
        self.cipher = Fernet(key)
    
    def _save_encrypted_credentials(self, credentials: MT5Credentials):
        """Salva credenciais criptografadas"""
        try:
            credentials_dict = {
                'login': credentials.login,
                'password': credentials.password,
                'server': credentials.server,
                'timeout': credentials.timeout,
                'portable': credentials.portable,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Criptografar dados
            json_data = json.dumps(credentials_dict)
            encrypted_data = self.cipher.encrypt(json_data.encode())
            
            # Salvar arquivo
            with open(self.config_file, 'wb') as f:
                f.write(encrypted_data)
            
            log_system_decision(
                "Credenciais MT5 salvas",
                "Credenciais criptografadas e armazenadas com segurança",
                "Arquivo de configuração criado com sucesso",
                DecisionLevel.MEDIUM
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar credenciais: {e}")
            log_system_decision(
                "Erro ao salvar credenciais MT5",
                f"Falha na criptografia ou escrita: {str(e)}",
                "Credenciais não foram salvas",
                DecisionLevel.HIGH
            )
    
    def _load_encrypted_credentials(self) -> Optional[MT5Credentials]:
        """Carrega credenciais criptografadas"""
        try:
            if not self.config_file.exists():
                return None
            
            # Ler arquivo criptografado
            with open(self.config_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Descriptografar
            decrypted_data = self.cipher.decrypt(encrypted_data)
            credentials_dict = json.loads(decrypted_data.decode())
            
            return MT5Credentials(
                login=credentials_dict['login'],
                password=credentials_dict['password'],
                server=credentials_dict.get('server', 'MetaQuotes-Demo'),
                timeout=credentials_dict.get('timeout', 60000),
                portable=credentials_dict.get('portable', False)
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar credenciais: {e}")
            log_system_decision(
                "Erro ao carregar credenciais MT5",
                f"Falha na descriptografia ou leitura: {str(e)}",
                "Usando credenciais padrão",
                DecisionLevel.MEDIUM
            )
            return None
    
    def connect_with_auth(self, custom_credentials: Optional[MT5Credentials] = None) -> ConnectionResult:
        """Conecta ao MT5 com autenticação segura"""
        with self._lock:
            start_time = time.time()
            
            # Log da tentativa de conexão
            with self.decision_logger.decision_context(
                DecisionType.SYSTEM_OPERATION,
                "Tentativa de conexão MT5 com autenticação",
                DecisionLevel.HIGH
            ) as ctx:
                
                try:
                    # Verificar se já está conectado
                    if self.connection_status == ConnectionStatus.AUTHENTICATED:
                        if self._verify_connection():
                            connection_time = (time.time() - start_time) * 1000
                            
                            ctx.add_criteria("connection_status", "already_connected", 1.0)
                            ctx.add_criteria("connection_time_ms", connection_time, 0.5)
                            ctx.set_justification("Conexão já estabelecida e verificada")
                            ctx.set_outcome("Reutilizada conexão existente")
                            ctx.set_confidence(0.9)
                            
                            return ConnectionResult(
                                success=True,
                                status=ConnectionStatus.AUTHENTICATED,
                                message="Já conectado e autenticado",
                                account_info=self._get_account_info(),
                                connection_time=connection_time
                            )
                    
                    # Atualizar status
                    self.connection_status = ConnectionStatus.CONNECTING
                    
                    # Obter credenciais
                    credentials = custom_credentials or self._load_encrypted_credentials() or self.default_credentials
                    
                    ctx.add_criteria("login", credentials.login, 1.0, description="Login do usuário")
                    ctx.add_criteria("server", credentials.server, 0.8, description="Servidor MT5")
                    ctx.add_criteria("timeout", credentials.timeout, 0.3, description="Timeout de conexão")
                    
                    # Tentar inicializar MT5
                    if not mt5.initialize(
                        login=int(credentials.login),
                        password=credentials.password,
                        server=credentials.server,
                        timeout=credentials.timeout,
                        portable=credentials.portable
                    ):
                        error_code = mt5.last_error()
                        error_msg = f"Falha na inicialização MT5: {error_code}"
                        
                        self.connection_status = ConnectionStatus.ERROR
                        self.connection_retries += 1
                        
                        ctx.add_criteria("error_code", error_code[0] if error_code else "unknown", 1.0)
                        ctx.add_criteria("retry_count", self.connection_retries, 0.5)
                        ctx.set_justification(f"Falha na inicialização: {error_msg}")
                        ctx.set_outcome("Conexão falhou")
                        ctx.set_confidence(0.1)
                        
                        return ConnectionResult(
                            success=False,
                            status=ConnectionStatus.ERROR,
                            message=error_msg,
                            error_code=error_code[0] if error_code else None
                        )
                    
                    # Verificar autenticação
                    self.connection_status = ConnectionStatus.AUTHENTICATING
                    account_info = mt5.account_info()
                    
                    if account_info is None:
                        error_msg = "Falha na autenticação - informações da conta não disponíveis"
                        self.connection_status = ConnectionStatus.ERROR
                        
                        ctx.add_criteria("account_info", "null", 1.0)
                        ctx.set_justification("Autenticação falhou - conta não acessível")
                        ctx.set_outcome("Autenticação rejeitada")
                        ctx.set_confidence(0.2)
                        
                        return ConnectionResult(
                            success=False,
                            status=ConnectionStatus.ERROR,
                            message=error_msg
                        )
                    
                    # Sucesso na conexão
                    self.connection_status = ConnectionStatus.AUTHENTICATED
                    self.last_activity = time.time()
                    self.connection_retries = 0
                    connection_time = (time.time() - start_time) * 1000
                    
                    # Converter account_info para dict
                    account_dict = {
                        'login': account_info.login,
                        'balance': account_info.balance,
                        'equity': account_info.equity,
                        'margin': account_info.margin,
                        'free_margin': account_info.margin_free,
                        'margin_level': account_info.margin_level,
                        'currency': account_info.currency,
                        'server': account_info.server,
                        'leverage': account_info.leverage,
                        'trade_allowed': account_info.trade_allowed,
                        'trade_expert': account_info.trade_expert
                    }
                    
                    ctx.add_criteria("account_login", account_dict['login'], 1.0)
                    ctx.add_criteria("account_balance", account_dict['balance'], 0.8)
                    ctx.add_criteria("trade_allowed", account_dict['trade_allowed'], 1.0)
                    ctx.add_criteria("connection_time_ms", connection_time, 0.5)
                    
                    ctx.set_justification(
                        f"Conexão estabelecida com sucesso. Login: {account_dict['login']}, "
                        f"Servidor: {account_dict['server']}, Trading permitido: {account_dict['trade_allowed']}"
                    )
                    ctx.set_outcome(f"Conectado e autenticado em {connection_time:.2f}ms")
                    ctx.set_confidence(0.95)
                    
                    return ConnectionResult(
                        success=True,
                        status=ConnectionStatus.AUTHENTICATED,
                        message="Conectado e autenticado com sucesso",
                        account_info=account_dict,
                        connection_time=connection_time
                    )
                    
                except Exception as e:
                    error_msg = f"Erro inesperado na conexão: {str(e)}"
                    self.connection_status = ConnectionStatus.ERROR
                    
                    ctx.add_criteria("exception", str(e), 1.0)
                    ctx.set_justification(f"Exceção durante conexão: {error_msg}")
                    ctx.set_outcome("Conexão falhou com exceção")
                    ctx.set_confidence(0.1)
                    
                    self.logger.error(error_msg)
                    
                    return ConnectionResult(
                        success=False,
                        status=ConnectionStatus.ERROR,
                        message=error_msg
                    )
    
    def _verify_connection(self) -> bool:
        """Verifica se a conexão ainda está ativa"""
        try:
            # Verificar timeout de sessão
            if time.time() - self.last_activity > self.session_timeout:
                log_system_decision(
                    "Timeout de sessão MT5",
                    f"Sessão inativa por {time.time() - self.last_activity:.0f} segundos",
                    "Conexão expirada",
                    DecisionLevel.MEDIUM
                )
                return False
            
            # Tentar obter informações da conta
            account_info = mt5.account_info()
            if account_info is None:
                return False
            
            # Atualizar última atividade
            self.last_activity = time.time()
            return True
            
        except Exception:
            return False
    
    def _get_account_info(self) -> Optional[Dict]:
        """Obtém informações da conta atual"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return None
            
            return {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'currency': account_info.currency,
                'server': account_info.server,
                'leverage': account_info.leverage,
                'trade_allowed': account_info.trade_allowed,
                'trade_expert': account_info.trade_expert
            }
        except Exception:
            return None
    
    def disconnect(self) -> bool:
        """Desconecta do MT5"""
        with self._lock:
            try:
                if self.connection_status != ConnectionStatus.DISCONNECTED:
                    mt5.shutdown()
                    self.connection_status = ConnectionStatus.DISCONNECTED
                    
                    log_system_decision(
                        "Desconexão MT5",
                        "Desconexão solicitada pelo usuário",
                        "MT5 desconectado com sucesso",
                        DecisionLevel.LOW
                    )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Erro ao desconectar: {e}")
                log_system_decision(
                    "Erro na desconexão MT5",
                    f"Exceção durante desconexão: {str(e)}",
                    "Desconexão pode ter falhado",
                    DecisionLevel.MEDIUM
                )
                return False
    
    def get_connection_status(self) -> ConnectionStatus:
        """Obtém status atual da conexão"""
        return self.connection_status
    
    def is_connected(self) -> bool:
        """Verifica se está conectado e autenticado"""
        return (self.connection_status == ConnectionStatus.AUTHENTICATED and 
                self._verify_connection())
    
    def reconnect_if_needed(self) -> ConnectionResult:
        """Reconecta se necessário"""
        if not self.is_connected():
            return self.connect_with_auth()
        
        return ConnectionResult(
            success=True,
            status=self.connection_status,
            message="Conexão já ativa",
            account_info=self._get_account_info()
        )
    
    def update_credentials(self, new_credentials: MT5Credentials) -> bool:
        """Atualiza credenciais armazenadas"""
        try:
            self._save_encrypted_credentials(new_credentials)
            
            log_system_decision(
                "Atualização de credenciais MT5",
                "Novas credenciais fornecidas pelo usuário",
                "Credenciais atualizadas e criptografadas",
                DecisionLevel.MEDIUM
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar credenciais: {e}")
            return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas da conexão"""
        return {
            'status': self.connection_status.value,
            'last_activity': self.last_activity,
            'session_timeout': self.session_timeout,
            'connection_retries': self.connection_retries,
            'max_retries': self.max_retries,
            'is_connected': self.is_connected(),
            'account_info': self._get_account_info()
        }

# Instância global do autenticador
_mt5_authenticator = None

def get_mt5_authenticator() -> MT5SecureAuthenticator:
    """Obtém instância global do autenticador MT5"""
    global _mt5_authenticator
    if _mt5_authenticator is None:
        _mt5_authenticator = MT5SecureAuthenticator()
    return _mt5_authenticator

# Funções de conveniência
def connect_mt5() -> ConnectionResult:
    """Conecta ao MT5 usando credenciais seguras"""
    authenticator = get_mt5_authenticator()
    return authenticator.connect_with_auth()

def disconnect_mt5() -> bool:
    """Desconecta do MT5"""
    authenticator = get_mt5_authenticator()
    return authenticator.disconnect()

def is_mt5_connected() -> bool:
    """Verifica se MT5 está conectado"""
    authenticator = get_mt5_authenticator()
    return authenticator.is_connected()

def get_mt5_account_info() -> Optional[Dict]:
    """Obtém informações da conta MT5"""
    authenticator = get_mt5_authenticator()
    return authenticator._get_account_info()

if __name__ == "__main__":
    # Teste do sistema de autenticação
    print("Testando sistema de autenticação MT5...")
    
    authenticator = get_mt5_authenticator()
    
    # Tentar conectar
    result = authenticator.connect_with_auth()
    
    if result.success:
        print(f"✓ Conexão bem-sucedida: {result.message}")
        print(f"  Tempo de conexão: {result.connection_time:.2f}ms")
        print(f"  Conta: {result.account_info['login']}")
        print(f"  Saldo: {result.account_info['balance']} {result.account_info['currency']}")
        
        # Obter estatísticas
        stats = authenticator.get_connection_stats()
        print(f"  Status: {stats['status']}")
        print(f"  Trading permitido: {stats['account_info']['trade_allowed']}")
        
    else:
        print(f"✗ Falha na conexão: {result.message}")
        if result.error_code:
            print(f"  Código de erro: {result.error_code}")
    
    # Desconectar
    if authenticator.disconnect():
        print("✓ Desconectado com sucesso")
    else:
        print("✗ Erro na desconexão")