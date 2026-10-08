#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Validação de Credenciais MT5
Implementa validação avançada de credenciais antes do salvamento
Autor: Sistema Forex ML
Data: 2024
"""

import re
import socket
import threading
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Resultado da validação"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class ValidationIssue:
    """Representa um problema de validação"""
    field: str
    result: ValidationResult
    message: str
    suggestion: Optional[str] = None

@dataclass
class CredentialValidationReport:
    """Relatório completo de validação"""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings: List[ValidationIssue]
    errors: List[ValidationIssue]
    
    def __post_init__(self):
        self.warnings = [issue for issue in self.issues if issue.result == ValidationResult.WARNING]
        self.errors = [issue for issue in self.issues if issue.result == ValidationResult.ERROR]
        self.is_valid = len(self.errors) == 0

class CredentialValidator:
    """
    Validador avançado de credenciais MT5
    Implementa verificações de formato, conectividade e segurança
    """
    
    def __init__(self):
        self.known_servers = {
            # Servidores MetaQuotes
            'MetaQuotes-Demo': {'host': 'mt5-demo.metaquotes.net', 'port': 443},
            'MetaQuotes-Live': {'host': 'mt5.metaquotes.net', 'port': 443},
            
            # Servidores comuns de brokers
            'Alpari-Demo': {'host': 'mt5-demo.alpari.com', 'port': 443},
            'Alpari-Live': {'host': 'mt5.alpari.com', 'port': 443},
            'XM-Demo': {'host': 'mt5-demo.xm.com', 'port': 443},
            'XM-Live': {'host': 'mt5.xm.com', 'port': 443},
            'FXCM-Demo': {'host': 'mt5-demo.fxcm.com', 'port': 443},
            'FXCM-Live': {'host': 'mt5.fxcm.com', 'port': 443}
        }
        
        self.validation_timeout = 5.0  # segundos
    
    def validate_credentials(self, credentials: Dict[str, Any]) -> CredentialValidationReport:
        """
        Valida credenciais MT5 completas
        
        Args:
            credentials: Dicionário com login, password, server, path
        
        Returns:
            CredentialValidationReport: Relatório completo de validação
        """
        issues = []
        
        # Validar cada campo
        issues.extend(self._validate_login(credentials.get('login')))
        issues.extend(self._validate_password(credentials.get('password')))
        issues.extend(self._validate_server(credentials.get('server')))
        issues.extend(self._validate_path(credentials.get('path')))
        
        # Validações de conectividade (se servidor válido)
        server_issues = [issue for issue in issues if issue.field == 'server' and issue.result == ValidationResult.ERROR]
        if not server_issues:
            issues.extend(self._validate_server_connectivity(credentials.get('server')))
        
        return CredentialValidationReport(is_valid=True, issues=issues)
    
    def _validate_login(self, login: Any) -> List[ValidationIssue]:
        """
        Valida o campo login
        
        Args:
            login: Valor do login
        
        Returns:
            List[ValidationIssue]: Lista de problemas encontrados
        """
        issues = []
        
        if not login:
            issues.append(ValidationIssue(
                field='login',
                result=ValidationResult.ERROR,
                message='Login é obrigatório',
                suggestion='Insira o número da conta fornecido pelo broker'
            ))
            return issues
        
        # Converter para string para validação
        login_str = str(login)
        
        # Verificar se é numérico
        if not login_str.isdigit():
            issues.append(ValidationIssue(
                field='login',
                result=ValidationResult.ERROR,
                message='Login deve conter apenas números',
                suggestion='Verifique se o número da conta está correto'
            ))
        
        # Verificar comprimento típico
        if len(login_str) < 4:
            issues.append(ValidationIssue(
                field='login',
                result=ValidationResult.WARNING,
                message='Login muito curto (menos de 4 dígitos)',
                suggestion='Verifique se o número da conta está completo'
            ))
        elif len(login_str) > 12:
            issues.append(ValidationIssue(
                field='login',
                result=ValidationResult.WARNING,
                message='Login muito longo (mais de 12 dígitos)',
                suggestion='Verifique se o número da conta está correto'
            ))
        
        return issues
    
    def _validate_password(self, password: Any) -> List[ValidationIssue]:
        """
        Valida o campo password
        
        Args:
            password: Valor da senha
        
        Returns:
            List[ValidationIssue]: Lista de problemas encontrados
        """
        issues = []
        
        if not password:
            issues.append(ValidationIssue(
                field='password',
                result=ValidationResult.ERROR,
                message='Senha é obrigatória',
                suggestion='Insira a senha da conta MT5'
            ))
            return issues
        
        password_str = str(password)
        
        # Verificar comprimento mínimo
        if len(password_str) < 4:
            issues.append(ValidationIssue(
                field='password',
                result=ValidationResult.WARNING,
                message='Senha muito curta (menos de 4 caracteres)',
                suggestion='Senhas MT5 geralmente têm pelo menos 4 caracteres'
            ))
        
        # Verificar se não é uma senha óbvia
        weak_passwords = ['1234', 'password', 'admin', '0000', '1111']
        if password_str.lower() in weak_passwords:
            issues.append(ValidationIssue(
                field='password',
                result=ValidationResult.WARNING,
                message='Senha pode ser fraca',
                suggestion='Considere usar uma senha mais segura'
            ))
        
        return issues
    
    def _validate_server(self, server: Any) -> List[ValidationIssue]:
        """
        Valida o campo server
        
        Args:
            server: Nome do servidor
        
        Returns:
            List[ValidationIssue]: Lista de problemas encontrados
        """
        issues = []
        
        if not server:
            issues.append(ValidationIssue(
                field='server',
                result=ValidationResult.ERROR,
                message='Servidor é obrigatório',
                suggestion='Insira o nome do servidor fornecido pelo broker'
            ))
            return issues
        
        server_str = str(server).strip()
        
        # Verificar formato básico
        if len(server_str) < 3:
            issues.append(ValidationIssue(
                field='server',
                result=ValidationResult.ERROR,
                message='Nome do servidor muito curto',
                suggestion='Verifique o nome do servidor com seu broker'
            ))
        
        # Verificar caracteres válidos
        if not re.match(r'^[a-zA-Z0-9._-]+$', server_str):
            issues.append(ValidationIssue(
                field='server',
                result=ValidationResult.WARNING,
                message='Nome do servidor contém caracteres inválidos',
                suggestion='Servidores MT5 geralmente usam apenas letras, números, pontos e hífens'
            ))
        
        # Verificar se é um servidor conhecido
        if server_str in self.known_servers:
            issues.append(ValidationIssue(
                field='server',
                result=ValidationResult.VALID,
                message=f'Servidor reconhecido: {server_str}'
            ))
        
        return issues
    
    def _validate_path(self, path: Any) -> List[ValidationIssue]:
        """
        Valida o caminho do MT5
        
        Args:
            path: Caminho para o executável MT5
        
        Returns:
            List[ValidationIssue]: Lista de problemas encontrados
        """
        issues = []
        
        if not path:
            issues.append(ValidationIssue(
                field='path',
                result=ValidationResult.WARNING,
                message='Caminho do MT5 não especificado',
                suggestion='O MT5 tentará usar a instalação padrão'
            ))
            return issues
        
        path_str = str(path).strip()
        path_obj = Path(path_str)
        
        # Verificar se o arquivo existe
        if not path_obj.exists():
            issues.append(ValidationIssue(
                field='path',
                result=ValidationResult.ERROR,
                message='Arquivo MT5 não encontrado',
                suggestion=f'Verifique se o caminho está correto: {path_str}'
            ))
            return issues
        
        # Verificar se é um arquivo executável
        if not path_obj.is_file():
            issues.append(ValidationIssue(
                field='path',
                result=ValidationResult.ERROR,
                message='Caminho não aponta para um arquivo',
                suggestion='Especifique o caminho completo para terminal64.exe'
            ))
        elif not path_str.lower().endswith('.exe'):
            issues.append(ValidationIssue(
                field='path',
                result=ValidationResult.WARNING,
                message='Arquivo pode não ser um executável',
                suggestion='MT5 geralmente usa terminal64.exe'
            ))
        
        return issues
    
    def _validate_server_connectivity(self, server: str) -> List[ValidationIssue]:
        """
        Valida conectividade com o servidor
        
        Args:
            server: Nome do servidor
        
        Returns:
            List[ValidationIssue]: Lista de problemas encontrados
        """
        issues = []
        
        if not server:
            return issues
        
        # Verificar se é um servidor conhecido
        server_info = self.known_servers.get(server)
        if server_info:
            host = server_info['host']
            port = server_info['port']
        else:
            # Tentar resolver o servidor como hostname
            host = server
            port = 443  # Porta padrão MT5
        
        try:
            # Teste de conectividade básica
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.validation_timeout)
            
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                issues.append(ValidationIssue(
                    field='server',
                    result=ValidationResult.VALID,
                    message=f'Servidor {server} está acessível'
                ))
            else:
                issues.append(ValidationIssue(
                    field='server',
                    result=ValidationResult.WARNING,
                    message=f'Não foi possível conectar ao servidor {server}',
                    suggestion='Verifique sua conexão de internet e o nome do servidor'
                ))
        
        except socket.gaierror:
            issues.append(ValidationIssue(
                field='server',
                result=ValidationResult.WARNING,
                message=f'Não foi possível resolver o servidor {server}',
                suggestion='Verifique se o nome do servidor está correto'
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                field='server',
                result=ValidationResult.WARNING,
                message=f'Erro ao testar conectividade: {str(e)}',
                suggestion='Verifique sua conexão de internet'
            ))
        
        return issues
    
    def validate_credentials_async(self, credentials: Dict[str, Any], 
                                 callback: callable) -> threading.Thread:
        """
        Valida credenciais de forma assíncrona
        
        Args:
            credentials: Credenciais para validar
            callback: Função chamada com o resultado
        
        Returns:
            threading.Thread: Thread de validação
        """
        def validate_worker():
            try:
                result = self.validate_credentials(credentials)
                callback(result, None)
            except Exception as e:
                callback(None, e)
        
        thread = threading.Thread(target=validate_worker, daemon=True)
        thread.start()
        return thread
    
    def get_server_suggestions(self, partial_name: str) -> List[str]:
        """
        Retorna sugestões de servidores baseadas em nome parcial
        
        Args:
            partial_name: Nome parcial do servidor
        
        Returns:
            List[str]: Lista de sugestões
        """
        if not partial_name:
            return list(self.known_servers.keys())
        
        partial_lower = partial_name.lower()
        suggestions = []
        
        for server_name in self.known_servers.keys():
            if partial_lower in server_name.lower():
                suggestions.append(server_name)
        
        return suggestions