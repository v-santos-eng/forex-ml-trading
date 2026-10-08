#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Manager - Gerenciamento seguro de credenciais e configurações
Autor: Sistema MT5 Trading
Data: 2024
"""

import os
import json
import base64
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


@dataclass
class SecurityConfig:
    """Configuração de segurança"""
    encryption_enabled: bool = True
    key_derivation_iterations: int = 100000
    salt_length: int = 32
    backup_enabled: bool = True
    max_backup_files: int = 5


class SecurityManager:
    """Gerenciador de segurança para credenciais e configurações"""
    
    def __init__(self, config_dir: str = "config", master_password: Optional[str] = None):
        self.config_dir = config_dir
        self.security_config = SecurityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Criar diretório de configuração se não existir
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Arquivos de segurança
        self.key_file = os.path.join(self.config_dir, ".security_key")
        self.salt_file = os.path.join(self.config_dir, ".security_salt")
        
        # Inicializar criptografia
        self._master_password = master_password or self._get_default_password()
        self._encryption_key = None
        self._initialize_encryption()
    
    def _get_default_password(self) -> str:
        """Gera senha padrão baseada no sistema"""
        import platform
        import getpass
        
        # Usar informações do sistema como base
        system_info = f"{platform.node()}-{getpass.getuser()}-MT5Trading"
        return hashlib.sha256(system_info.encode()).hexdigest()[:32]
    
    def _initialize_encryption(self):
        """Inicializa sistema de criptografia"""
        try:
            # Verificar se já existe chave
            if os.path.exists(self.key_file) and os.path.exists(self.salt_file):
                self._load_encryption_key()
            else:
                self._generate_encryption_key()
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar criptografia: {e}")
            raise
    
    def _generate_encryption_key(self):
        """Gera nova chave de criptografia"""
        try:
            # Gerar salt aleatório
            salt = os.urandom(self.security_config.salt_length)
            
            # Derivar chave da senha mestre
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.security_config.key_derivation_iterations,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(self._master_password.encode()))
            self._encryption_key = Fernet(key)
            
            # Salvar salt
            with open(self.salt_file, 'wb') as f:
                f.write(salt)
            
            # Salvar indicador de chave (não a chave real)
            key_hash = hashlib.sha256(key).hexdigest()
            with open(self.key_file, 'w') as f:
                f.write(key_hash)
            
            self.logger.info("Nova chave de criptografia gerada")
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar chave de criptografia: {e}")
            raise
    
    def _load_encryption_key(self):
        """Carrega chave de criptografia existente"""
        try:
            # Carregar salt
            with open(self.salt_file, 'rb') as f:
                salt = f.read()
            
            # Derivar chave da senha mestre
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.security_config.key_derivation_iterations,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(self._master_password.encode()))
            
            # Verificar se a chave está correta
            with open(self.key_file, 'r') as f:
                stored_hash = f.read().strip()
            
            key_hash = hashlib.sha256(key).hexdigest()
            if key_hash != stored_hash:
                raise ValueError("Senha mestre incorreta")
            
            self._encryption_key = Fernet(key)
            self.logger.info("Chave de criptografia carregada")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar chave de criptografia: {e}")
            raise
    
    def encrypt_data(self, data: str) -> str:
        """Criptografa dados"""
        if not self.security_config.encryption_enabled or not self._encryption_key:
            return data
        
        try:
            encrypted_data = self._encryption_key.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            self.logger.error(f"Erro ao criptografar dados: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Descriptografa dados"""
        if not self.security_config.encryption_enabled or not self._encryption_key:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._encryption_key.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Erro ao descriptografar dados: {e}")
            # Se falhar, pode ser dados não criptografados
            return encrypted_data
    
    def encrypt_credentials(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Criptografa credenciais sensíveis"""
        encrypted_creds = credentials.copy()
        
        # Campos sensíveis para criptografar
        sensitive_fields = ['password', 'api_key', 'secret_key', 'token']
        
        for field in sensitive_fields:
            if field in encrypted_creds and encrypted_creds[field]:
                encrypted_creds[field] = self.encrypt_data(str(encrypted_creds[field]))
                encrypted_creds[f'{field}_encrypted'] = True
        
        return encrypted_creds
    
    def decrypt_credentials(self, encrypted_credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Descriptografa credenciais"""
        decrypted_creds = encrypted_credentials.copy()
        
        # Campos sensíveis para descriptografar
        sensitive_fields = ['password', 'api_key', 'secret_key', 'token']
        
        for field in sensitive_fields:
            if f'{field}_encrypted' in decrypted_creds and decrypted_creds.get(f'{field}_encrypted'):
                if field in decrypted_creds:
                    decrypted_creds[field] = self.decrypt_data(decrypted_creds[field])
                    # Remover flag de criptografia
                    del decrypted_creds[f'{field}_encrypted']
        
        return decrypted_creds
    
    def save_secure_config(self, config: Dict[str, Any], config_file: str) -> bool:
        """Salva configuração com criptografia"""
        try:
            # Fazer backup se habilitado
            if self.security_config.backup_enabled:
                self._create_backup(config_file)
            
            # Criptografar credenciais MT5
            secure_config = config.copy()
            if 'mt5' in secure_config:
                secure_config['mt5'] = self.encrypt_credentials(secure_config['mt5'])
            
            # Salvar configuração
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(secure_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuração segura salva: {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar configuração segura: {e}")
            return False
    
    def load_secure_config(self, config_file: str) -> Optional[Dict[str, Any]]:
        """Carrega configuração com descriptografia"""
        try:
            if not os.path.exists(config_file):
                return None
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Descriptografar credenciais MT5
            if 'mt5' in config:
                config['mt5'] = self.decrypt_credentials(config['mt5'])
            
            self.logger.info(f"Configuração segura carregada: {config_file}")
            return config
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração segura: {e}")
            return None
    
    def _create_backup(self, config_file: str):
        """Cria backup da configuração"""
        try:
            if not os.path.exists(config_file):
                return
            
            backup_dir = os.path.join(self.config_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Nome do backup com timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{os.path.basename(config_file)}.{timestamp}.bak"
            backup_path = os.path.join(backup_dir, backup_name)
            
            # Copiar arquivo
            import shutil
            shutil.copy2(config_file, backup_path)
            
            # Limpar backups antigos
            self._cleanup_old_backups(backup_dir)
            
            self.logger.info(f"Backup criado: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Erro ao criar backup: {e}")
    
    def _cleanup_old_backups(self, backup_dir: str):
        """Remove backups antigos"""
        try:
            backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.bak')]
            backup_files.sort(key=lambda x: os.path.getctime(os.path.join(backup_dir, x)))
            
            # Remover arquivos excedentes
            while len(backup_files) > self.security_config.max_backup_files:
                old_backup = backup_files.pop(0)
                os.remove(os.path.join(backup_dir, old_backup))
                self.logger.info(f"Backup antigo removido: {old_backup}")
                
        except Exception as e:
            self.logger.error(f"Erro ao limpar backups: {e}")
    
    def validate_credentials(self, credentials: Dict[str, Any]) -> Tuple[bool, str]:
        """Valida credenciais MT5"""
        try:
            # Verificar campos obrigatórios
            required_fields = ['login', 'password', 'server']
            for field in required_fields:
                if not credentials.get(field):
                    return False, f"Campo obrigatório ausente: {field}"
            
            # Validar formato do login
            login = credentials.get('login')
            if not str(login).isdigit():
                return False, "Login deve ser numérico"
            
            # Validar senha
            password = credentials.get('password')
            if len(password) < 4:
                return False, "Senha muito curta (mínimo 4 caracteres)"
            
            # Validar servidor
            server = credentials.get('server')
            if not server or len(server) < 3:
                return False, "Servidor inválido"
            
            return True, "Credenciais válidas"
            
        except Exception as e:
            return False, f"Erro na validação: {e}"
    
    def change_master_password(self, new_password: str) -> bool:
        """Altera senha mestre"""
        try:
            old_password = self._master_password
            self._master_password = new_password
            
            # Regenerar chave com nova senha
            self._generate_encryption_key()
            
            self.logger.info("Senha mestre alterada com sucesso")
            return True
            
        except Exception as e:
            # Restaurar senha antiga em caso de erro
            self._master_password = old_password
            self.logger.error(f"Erro ao alterar senha mestre: {e}")
            return False
    
    def get_security_status(self) -> Dict[str, Any]:
        """Retorna status de segurança"""
        return {
            'encryption_enabled': self.security_config.encryption_enabled,
            'key_file_exists': os.path.exists(self.key_file),
            'salt_file_exists': os.path.exists(self.salt_file),
            'backup_enabled': self.security_config.backup_enabled,
            'config_dir': self.config_dir
        }