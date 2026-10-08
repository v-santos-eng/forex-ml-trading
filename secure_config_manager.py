#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerenciador Seguro de Configurações
Implementa backup automático, validação de integridade e recuperação
Autor: Sistema Forex ML
Data: 2024
"""

import os
import json
import hashlib
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class BackupStatus(Enum):
    """Status do backup"""
    SUCCESS = "success"
    FAILED = "failed"
    CORRUPTED = "corrupted"
    MISSING = "missing"

class ValidationResult(Enum):
    """Resultado da validação"""
    VALID = "valid"
    INVALID = "invalid"
    CORRUPTED = "corrupted"
    MISSING_FIELDS = "missing_fields"

@dataclass
class BackupInfo:
    """Informações do backup"""
    filename: str
    timestamp: datetime
    size: int
    checksum: str
    status: BackupStatus
    config_version: Optional[str] = None

@dataclass
class ValidationReport:
    """Relatório de validação"""
    is_valid: bool
    result: ValidationResult
    issues: List[str]
    missing_fields: List[str]
    checksum_valid: bool
    file_exists: bool

class SecureConfigManager:
    """
    Gerenciador seguro de configurações com backup e validação
    """
    
    def __init__(self, config_file: str = "config.json",
                 backup_dir: str = "config_backups",
                 max_backups: int = 10,
                 backup_interval_hours: int = 24,
                 auto_backup: bool = True):
        
        self.config_file = Path(config_file)
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self.backup_interval_hours = backup_interval_hours
        self.auto_backup = auto_backup
        
        # Criar diretório de backup se não existir
        self.backup_dir.mkdir(exist_ok=True)
        
        # Campos obrigatórios para validação
        self.required_fields = {
            'mt5': ['login', 'password', 'server', 'path'],
            'system': ['auto_start_components'],
            'dashboard': ['auto_start'],
            'ml': ['prediction_interval_seconds'],
            'risk': ['max_risk_per_trade'],
            'logging': ['level']
        }
        
        # Thread de backup automático
        self._backup_thread: Optional[threading.Thread] = None
        self._stop_backup = threading.Event()
        
        # Cache de checksums
        self._checksum_cache: Dict[str, str] = {}
        
        if self.auto_backup:
            self.start_auto_backup()
    
    def load_config(self, validate: bool = True) -> Dict[str, Any]:
        """
        Carrega configuração com validação opcional
        
        Args:
            validate: Se deve validar a configuração
        
        Returns:
            Dict[str, Any]: Configuração carregada
        
        Raises:
            FileNotFoundError: Se arquivo não existir
            ValueError: Se configuração for inválida
        """
        if not self.config_file.exists():
            logger.warning(f"Arquivo de configuração não encontrado: {self.config_file}")
            return self._get_default_config()
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if validate:
                validation_report = self.validate_config(config)
                if not validation_report.is_valid:
                    logger.error(f"Configuração inválida: {validation_report.issues}")
                    
                    # Tentar recuperar de backup
                    backup_config = self._try_recover_from_backup()
                    if backup_config:
                        logger.info("Configuração recuperada de backup")
                        return backup_config
                    
                    raise ValueError(f"Configuração inválida: {validation_report.issues}")
            
            logger.info("Configuração carregada com sucesso")
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar JSON: {e}")
            
            # Tentar recuperar de backup
            backup_config = self._try_recover_from_backup()
            if backup_config:
                logger.info("Configuração recuperada de backup após erro de JSON")
                return backup_config
            
            raise ValueError(f"Arquivo de configuração corrompido: {e}")
        
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            raise
    
    def save_config(self, config: Dict[str, Any], 
                   create_backup: bool = True,
                   validate: bool = True) -> bool:
        """
        Salva configuração com backup e validação
        
        Args:
            config: Configuração para salvar
            create_backup: Se deve criar backup antes de salvar
            validate: Se deve validar antes de salvar
        
        Returns:
            bool: True se salvou com sucesso
        """
        try:
            if validate:
                validation_report = self.validate_config(config)
                if not validation_report.is_valid:
                    logger.error(f"Configuração inválida: {validation_report.issues}")
                    return False
            
            # Criar backup se solicitado e arquivo existir
            if create_backup and self.config_file.exists():
                backup_path = self.create_backup()
                if backup_path:
                    logger.info(f"Backup criado: {backup_path}")
            
            # Salvar configuração
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Atualizar cache de checksum
            self._checksum_cache[str(self.config_file)] = self._calculate_checksum(self.config_file)
            
            logger.info("Configuração salva com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar configuração: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationReport:
        """
        Valida configuração completa
        
        Args:
            config: Configuração para validar
        
        Returns:
            ValidationReport: Relatório de validação
        """
        issues = []
        missing_fields = []
        
        # Verificar campos obrigatórios
        for section, fields in self.required_fields.items():
            if section not in config:
                issues.append(f"Seção obrigatória ausente: {section}")
                missing_fields.extend([f"{section}.{field}" for field in fields])
                continue
            
            section_config = config[section]
            for field in fields:
                if field not in section_config:
                    issues.append(f"Campo obrigatório ausente: {section}.{field}")
                    missing_fields.append(f"{section}.{field}")
        
        # Validações específicas
        self._validate_mt5_config(config.get('mt5', {}), issues)
        self._validate_risk_config(config.get('risk', {}), issues)
        self._validate_ml_config(config.get('ml', {}), issues)
        
        # Verificar integridade do arquivo
        checksum_valid = True
        file_exists = self.config_file.exists()
        
        if file_exists:
            current_checksum = self._calculate_checksum(self.config_file)
            cached_checksum = self._checksum_cache.get(str(self.config_file))
            
            if cached_checksum and current_checksum != cached_checksum:
                issues.append("Checksum do arquivo não confere - possível corrupção")
                checksum_valid = False
        
        # Determinar resultado
        if not file_exists:
            result = ValidationResult.MISSING_FIELDS
        elif not checksum_valid:
            result = ValidationResult.CORRUPTED
        elif missing_fields:
            result = ValidationResult.MISSING_FIELDS
        elif issues:
            result = ValidationResult.INVALID
        else:
            result = ValidationResult.VALID
        
        return ValidationReport(
            is_valid=len(issues) == 0 and checksum_valid,
            result=result,
            issues=issues,
            missing_fields=missing_fields,
            checksum_valid=checksum_valid,
            file_exists=file_exists
        )
    
    def _validate_mt5_config(self, mt5_config: Dict[str, Any], issues: List[str]):
        """Valida configuração MT5"""
        if 'login' in mt5_config:
            login = mt5_config['login']
            if not isinstance(login, int) or login <= 0:
                issues.append("Login MT5 deve ser um número inteiro positivo")
        
        if 'server' in mt5_config:
            server = mt5_config['server']
            if not isinstance(server, str) or len(server.strip()) < 3:
                issues.append("Servidor MT5 deve ser uma string válida")
        
        if 'path' in mt5_config:
            path = mt5_config['path']
            if path and not Path(path).exists():
                issues.append(f"Caminho MT5 não existe: {path}")
    
    def _validate_risk_config(self, risk_config: Dict[str, Any], issues: List[str]):
        """Valida configuração de risco"""
        if 'max_risk_per_trade' in risk_config:
            max_risk = risk_config['max_risk_per_trade']
            if not isinstance(max_risk, (int, float)) or max_risk <= 0 or max_risk > 100:
                issues.append("Risco máximo por trade deve estar entre 0 e 100")
    
    def _validate_ml_config(self, ml_config: Dict[str, Any], issues: List[str]):
        """Valida configuração ML"""
        if 'prediction_interval_seconds' in ml_config:
            interval = ml_config['prediction_interval_seconds']
            if not isinstance(interval, int) or interval < 0:
                issues.append("Intervalo de predição deve ser um número inteiro não negativo")
    
    def create_backup(self, suffix: str = None) -> Optional[Path]:
        """
        Cria backup da configuração atual
        
        Args:
            suffix: Sufixo opcional para o nome do backup
        
        Returns:
            Optional[Path]: Caminho do backup criado ou None se falhou
        """
        if not self.config_file.exists():
            logger.warning("Arquivo de configuração não existe para backup")
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if suffix:
                backup_filename = f"config_backup_{timestamp}_{suffix}.json"
            else:
                backup_filename = f"config_backup_{timestamp}.json"
            
            backup_path = self.backup_dir / backup_filename
            
            # Copiar arquivo
            shutil.copy2(self.config_file, backup_path)
            
            # Calcular checksum
            checksum = self._calculate_checksum(backup_path)
            
            # Salvar informações do backup
            backup_info = BackupInfo(
                filename=backup_filename,
                timestamp=datetime.now(),
                size=backup_path.stat().st_size,
                checksum=checksum,
                status=BackupStatus.SUCCESS
            )
            
            self._save_backup_info(backup_info)
            
            # Limpar backups antigos
            self._cleanup_old_backups()
            
            logger.info(f"Backup criado com sucesso: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Erro ao criar backup: {e}")
            return None
    
    def list_backups(self) -> List[BackupInfo]:
        """
        Lista todos os backups disponíveis
        
        Returns:
            List[BackupInfo]: Lista de informações dos backups
        """
        backups = []
        
        try:
            for backup_file in self.backup_dir.glob("config_backup_*.json"):
                try:
                    stat = backup_file.stat()
                    checksum = self._calculate_checksum(backup_file)
                    
                    # Extrair timestamp do nome do arquivo
                    timestamp_str = backup_file.stem.split('_')[2:4]  # ['YYYYMMDD', 'HHMMSS']
                    if len(timestamp_str) >= 2:
                        timestamp = datetime.strptime(f"{timestamp_str[0]}_{timestamp_str[1]}", "%Y%m%d_%H%M%S")
                    else:
                        timestamp = datetime.fromtimestamp(stat.st_mtime)
                    
                    backup_info = BackupInfo(
                        filename=backup_file.name,
                        timestamp=timestamp,
                        size=stat.st_size,
                        checksum=checksum,
                        status=BackupStatus.SUCCESS
                    )
                    
                    backups.append(backup_info)
                    
                except Exception as e:
                    logger.error(f"Erro ao processar backup {backup_file}: {e}")
        
        except Exception as e:
            logger.error(f"Erro ao listar backups: {e}")
        
        # Ordenar por timestamp (mais recente primeiro)
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        return backups
    
    def restore_from_backup(self, backup_filename: str) -> bool:
        """
        Restaura configuração de um backup
        
        Args:
            backup_filename: Nome do arquivo de backup
        
        Returns:
            bool: True se restaurou com sucesso
        """
        backup_path = self.backup_dir / backup_filename
        
        if not backup_path.exists():
            logger.error(f"Backup não encontrado: {backup_path}")
            return False
        
        try:
            # Validar backup antes de restaurar
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_config = json.load(f)
            
            validation_report = self.validate_config(backup_config)
            if not validation_report.is_valid:
                logger.error(f"Backup inválido: {validation_report.issues}")
                return False
            
            # Criar backup da configuração atual antes de restaurar
            if self.config_file.exists():
                self.create_backup("before_restore")
            
            # Restaurar configuração
            shutil.copy2(backup_path, self.config_file)
            
            # Atualizar cache de checksum
            self._checksum_cache[str(self.config_file)] = self._calculate_checksum(self.config_file)
            
            logger.info(f"Configuração restaurada de: {backup_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao restaurar backup: {e}")
            return False
    
    def _try_recover_from_backup(self) -> Optional[Dict[str, Any]]:
        """
        Tenta recuperar configuração do backup mais recente válido
        
        Returns:
            Optional[Dict[str, Any]]: Configuração recuperada ou None
        """
        backups = self.list_backups()
        
        for backup in backups:
            try:
                backup_path = self.backup_dir / backup.filename
                
                with open(backup_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                validation_report = self.validate_config(config)
                if validation_report.is_valid:
                    logger.info(f"Configuração recuperada do backup: {backup.filename}")
                    return config
                    
            except Exception as e:
                logger.error(f"Erro ao tentar backup {backup.filename}: {e}")
                continue
        
        logger.error("Nenhum backup válido encontrado")
        return None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Retorna configuração padrão
        
        Returns:
            Dict[str, Any]: Configuração padrão
        """
        return {
            "mt5": {
                "login": 0,
                "password": "",
                "server": "",
                "path": "",
                "timeout": 10000
            },
            "system": {
                "auto_start_components": True,
                "enable_auto_trading": False
            },
            "dashboard": {
                "auto_start": True,
                "port": 8050
            },
            "ml": {
                "prediction_interval_seconds": 30,
                "model_retrain_hours": 24
            },
            "risk": {
                "max_risk_per_trade": 2.0,
                "max_daily_loss": 5.0
            },
            "logging": {
                "level": "INFO",
                "file_rotation": True
            }
        }
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """
        Calcula checksum MD5 do arquivo
        
        Args:
            file_path: Caminho do arquivo
        
        Returns:
            str: Checksum MD5
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _save_backup_info(self, backup_info: BackupInfo):
        """Salva informações do backup"""
        info_file = self.backup_dir / f"{backup_info.filename}.info"
        
        try:
            info_data = {
                'filename': backup_info.filename,
                'timestamp': backup_info.timestamp.isoformat(),
                'size': backup_info.size,
                'checksum': backup_info.checksum,
                'status': backup_info.status.value,
                'config_version': backup_info.config_version
            }
            
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erro ao salvar informações do backup: {e}")
    
    def _cleanup_old_backups(self):
        """Remove backups antigos mantendo apenas os mais recentes"""
        try:
            backups = self.list_backups()
            
            if len(backups) > self.max_backups:
                # Remover backups mais antigos
                backups_to_remove = backups[self.max_backups:]
                
                for backup in backups_to_remove:
                    backup_path = self.backup_dir / backup.filename
                    info_path = self.backup_dir / f"{backup.filename}.info"
                    
                    try:
                        if backup_path.exists():
                            backup_path.unlink()
                        if info_path.exists():
                            info_path.unlink()
                        
                        logger.info(f"Backup antigo removido: {backup.filename}")
                        
                    except Exception as e:
                        logger.error(f"Erro ao remover backup {backup.filename}: {e}")
        
        except Exception as e:
            logger.error(f"Erro na limpeza de backups: {e}")
    
    def start_auto_backup(self):
        """Inicia backup automático em thread separada"""
        if self._backup_thread and self._backup_thread.is_alive():
            return
        
        self._stop_backup.clear()
        self._backup_thread = threading.Thread(
            target=self._auto_backup_loop,
            name="AutoBackup",
            daemon=True
        )
        self._backup_thread.start()
        logger.info(f"Backup automático iniciado (intervalo: {self.backup_interval_hours}h)")
    
    def stop_auto_backup(self):
        """Para backup automático"""
        self._stop_backup.set()
        if self._backup_thread and self._backup_thread.is_alive():
            self._backup_thread.join(timeout=5.0)
        logger.info("Backup automático parado")
    
    def _auto_backup_loop(self):
        """Loop de backup automático"""
        while not self._stop_backup.is_set():
            try:
                # Aguardar intervalo ou sinal de parada
                if self._stop_backup.wait(timeout=self.backup_interval_hours * 3600):
                    break
                
                # Criar backup se arquivo existir e foi modificado
                if self.config_file.exists():
                    current_checksum = self._calculate_checksum(self.config_file)
                    cached_checksum = self._checksum_cache.get(str(self.config_file))
                    
                    if current_checksum != cached_checksum:
                        self.create_backup("auto")
                        self._checksum_cache[str(self.config_file)] = current_checksum
                
            except Exception as e:
                logger.error(f"Erro no backup automático: {e}")
                time.sleep(300)  # Aguardar 5 minutos antes de tentar novamente
    
    def get_integrity_report(self) -> Dict[str, Any]:
        """
        Gera relatório de integridade completo
        
        Returns:
            Dict[str, Any]: Relatório de integridade
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'config_file': {
                'exists': self.config_file.exists(),
                'size': self.config_file.stat().st_size if self.config_file.exists() else 0,
                'checksum': self._calculate_checksum(self.config_file) if self.config_file.exists() else None
            },
            'backups': [],
            'validation': None
        }
        
        # Informações dos backups
        backups = self.list_backups()
        for backup in backups:
            report['backups'].append({
                'filename': backup.filename,
                'timestamp': backup.timestamp.isoformat(),
                'size': backup.size,
                'status': backup.status.value
            })
        
        # Validação da configuração atual
        if self.config_file.exists():
            try:
                config = self.load_config(validate=False)
                validation_report = self.validate_config(config)
                
                report['validation'] = {
                    'is_valid': validation_report.is_valid,
                    'result': validation_report.result.value,
                    'issues': validation_report.issues,
                    'missing_fields': validation_report.missing_fields
                }
            except Exception as e:
                report['validation'] = {
                    'error': str(e)
                }
        
        return report