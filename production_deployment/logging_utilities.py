#!/usr/bin/env python3
"""
Utilitários para Gerenciamento do Sistema de Logging Detalhado

Este módulo fornece ferramentas para manutenção, análise e gerenciamento
dos componentes do sistema de logging detalhado.

Autor: Sistema de Trading ML
Versão: 1.0.0
Data: 2024
"""

import os
import json
import gzip
import shutil
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml
import pandas as pd
from dataclasses import dataclass


@dataclass
class LogFileInfo:
    """Informações sobre um arquivo de log."""
    path: str
    size_mb: float
    created_date: datetime
    modified_date: datetime
    record_count: int
    file_type: str  # 'comprehensive', 'detailed', 'trading'


class LoggingUtilities:
    """Utilitários para gerenciamento do sistema de logging."""
    
    def __init__(self, config_path: str = "logging_config.yaml"):
        """Inicializar utilitários com configuração."""
        self.config_path = config_path
        self.config = self._load_config()
        self.log_base_dir = Path(self.config.get('general', {}).get('log_base_directory', './logs'))
        self.reports_dir = Path(self.config.get('general', {}).get('reports_directory', './reports'))
    
    def _load_config(self) -> Dict[str, Any]:
        """Carregar configuração do arquivo YAML."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Arquivo de configuração não encontrado: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Erro ao carregar configuração: {e}")
            return {}
    
    def get_log_files_info(self) -> List[LogFileInfo]:
        """Obter informações sobre todos os arquivos de log."""
        log_files = []
        
        if not self.log_base_dir.exists():
            return log_files
        
        for file_path in self.log_base_dir.rglob('*.jsonl'):
            try:
                stat = file_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                created_date = datetime.fromtimestamp(stat.st_ctime)
                modified_date = datetime.fromtimestamp(stat.st_mtime)
                
                # Contar registros no arquivo
                record_count = self._count_records(file_path)
                
                # Determinar tipo do arquivo
                file_type = self._determine_file_type(file_path.name)
                
                log_files.append(LogFileInfo(
                    path=str(file_path),
                    size_mb=size_mb,
                    created_date=created_date,
                    modified_date=modified_date,
                    record_count=record_count,
                    file_type=file_type
                ))
            except Exception as e:
                print(f"Erro ao processar arquivo {file_path}: {e}")
        
        return sorted(log_files, key=lambda x: x.modified_date, reverse=True)
    
    def _count_records(self, file_path: Path) -> int:
        """Contar número de registros em um arquivo JSONL."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0
    
    def _determine_file_type(self, filename: str) -> str:
        """Determinar tipo do arquivo baseado no nome."""
        if 'comprehensive' in filename.lower():
            return 'comprehensive'
        elif 'detailed' in filename.lower():
            return 'detailed'
        elif 'trading' in filename.lower():
            return 'trading'
        else:
            return 'unknown'
    
    def cleanup_old_logs(self, days_to_keep: Optional[int] = None) -> Dict[str, Any]:
        """Limpar logs antigos baseado na configuração."""
        if days_to_keep is None:
            days_to_keep = self.config.get('cleanup', {}).get('log_retention_days', 90)
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        deleted_files = []
        total_space_freed = 0
        errors = []
        
        log_files = self.get_log_files_info()
        
        for log_file in log_files:
            if log_file.modified_date < cutoff_date:
                try:
                    file_path = Path(log_file.path)
                    total_space_freed += log_file.size_mb
                    file_path.unlink()
                    deleted_files.append(log_file.path)
                except Exception as e:
                    errors.append(f"Erro ao deletar {log_file.path}: {e}")
        
        return {
            'deleted_files': deleted_files,
            'total_files_deleted': len(deleted_files),
            'total_space_freed_mb': round(total_space_freed, 2),
            'errors': errors,
            'cutoff_date': cutoff_date.isoformat()
        }
    
    def compress_old_logs(self, days_old: Optional[int] = None) -> Dict[str, Any]:
        """Comprimir logs antigos para economizar espaço."""
        if days_old is None:
            days_old = self.config.get('performance', {}).get('compression_age_days', 7)
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        compressed_files = []
        total_space_saved = 0
        errors = []
        
        log_files = self.get_log_files_info()
        
        for log_file in log_files:
            if (log_file.modified_date < cutoff_date and 
                not log_file.path.endswith('.gz')):
                
                try:
                    original_path = Path(log_file.path)
                    compressed_path = Path(f"{log_file.path}.gz")
                    
                    # Comprimir arquivo
                    with open(original_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Calcular espaço economizado
                    original_size = original_path.stat().st_size
                    compressed_size = compressed_path.stat().st_size
                    space_saved = (original_size - compressed_size) / (1024 * 1024)
                    
                    # Remover arquivo original
                    original_path.unlink()
                    
                    compressed_files.append({
                        'original_path': str(original_path),
                        'compressed_path': str(compressed_path),
                        'space_saved_mb': round(space_saved, 2)
                    })
                    
                    total_space_saved += space_saved
                    
                except Exception as e:
                    errors.append(f"Erro ao comprimir {log_file.path}: {e}")
        
        return {
            'compressed_files': compressed_files,
            'total_files_compressed': len(compressed_files),
            'total_space_saved_mb': round(total_space_saved, 2),
            'errors': errors
        }
    
    def analyze_log_patterns(self, file_type: str = 'all', days_back: int = 7) -> Dict[str, Any]:
        """Analisar padrões nos logs para insights."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        analysis = {
            'period': {
                'start_date': cutoff_date.isoformat(),
                'end_date': datetime.now().isoformat(),
                'days_analyzed': days_back
            },
            'file_statistics': {},
            'record_patterns': {},
            'error_analysis': {},
            'performance_metrics': {}
        }
        
        log_files = self.get_log_files_info()
        relevant_files = [
            f for f in log_files 
            if (file_type == 'all' or f.file_type == file_type) and 
               f.modified_date >= cutoff_date
        ]
        
        # Estatísticas de arquivos
        analysis['file_statistics'] = {
            'total_files': len(relevant_files),
            'total_size_mb': sum(f.size_mb for f in relevant_files),
            'total_records': sum(f.record_count for f in relevant_files),
            'average_file_size_mb': sum(f.size_mb for f in relevant_files) / len(relevant_files) if relevant_files else 0,
            'files_by_type': {}
        }
        
        # Agrupar por tipo
        for file_info in relevant_files:
            file_type_key = file_info.file_type
            if file_type_key not in analysis['file_statistics']['files_by_type']:
                analysis['file_statistics']['files_by_type'][file_type_key] = {
                    'count': 0,
                    'total_size_mb': 0,
                    'total_records': 0
                }
            
            analysis['file_statistics']['files_by_type'][file_type_key]['count'] += 1
            analysis['file_statistics']['files_by_type'][file_type_key]['total_size_mb'] += file_info.size_mb
            analysis['file_statistics']['files_by_type'][file_type_key]['total_records'] += file_info.record_count
        
        # Analisar padrões de registros
        analysis['record_patterns'] = self._analyze_record_patterns(relevant_files)
        
        return analysis
    
    def _analyze_record_patterns(self, log_files: List[LogFileInfo]) -> Dict[str, Any]:
        """Analisar padrões nos registros dos logs."""
        patterns = {
            'hourly_distribution': {},
            'daily_distribution': {},
            'record_types': {},
            'confidence_scores': [],
            'processing_times': [],
            'symbols_analyzed': set()
        }
        
        for log_file in log_files:
            try:
                with open(log_file.path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        
                        try:
                            record = json.loads(line)
                            
                            # Distribuição por hora
                            timestamp = record.get('timestamp', '')
                            if timestamp:
                                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                hour = dt.hour
                                patterns['hourly_distribution'][hour] = patterns['hourly_distribution'].get(hour, 0) + 1
                                
                                day = dt.strftime('%Y-%m-%d')
                                patterns['daily_distribution'][day] = patterns['daily_distribution'].get(day, 0) + 1
                            
                            # Tipos de registro
                            record_type = record.get('analysis_type') or record.get('decision_type', 'unknown')
                            patterns['record_types'][record_type] = patterns['record_types'].get(record_type, 0) + 1
                            
                            # Scores de confiança
                            confidence = record.get('confidence_score')
                            if confidence is not None:
                                patterns['confidence_scores'].append(confidence)
                            
                            # Tempos de processamento
                            processing_time = record.get('processing_time_ms')
                            if processing_time is not None:
                                patterns['processing_times'].append(processing_time)
                            
                            # Símbolos analisados
                            symbol = record.get('symbol')
                            if symbol:
                                patterns['symbols_analyzed'].add(symbol)
                        
                        except json.JSONDecodeError:
                            continue
            
            except Exception as e:
                print(f"Erro ao analisar arquivo {log_file.path}: {e}")
        
        # Converter set para list para serialização JSON
        patterns['symbols_analyzed'] = list(patterns['symbols_analyzed'])
        
        # Calcular estatísticas
        if patterns['confidence_scores']:
            patterns['confidence_stats'] = {
                'mean': sum(patterns['confidence_scores']) / len(patterns['confidence_scores']),
                'min': min(patterns['confidence_scores']),
                'max': max(patterns['confidence_scores']),
                'count': len(patterns['confidence_scores'])
            }
        
        if patterns['processing_times']:
            patterns['processing_time_stats'] = {
                'mean_ms': sum(patterns['processing_times']) / len(patterns['processing_times']),
                'min_ms': min(patterns['processing_times']),
                'max_ms': max(patterns['processing_times']),
                'count': len(patterns['processing_times'])
            }
        
        return patterns
    
    def export_logs_to_csv(self, output_dir: str, file_type: str = 'all', days_back: int = 7) -> Dict[str, str]:
        """Exportar logs para formato CSV para análise externa."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        log_files = self.get_log_files_info()
        relevant_files = [
            f for f in log_files 
            if (file_type == 'all' or f.file_type == file_type) and 
               f.modified_date >= cutoff_date
        ]
        
        # Agrupar por tipo de arquivo
        files_by_type = {}
        for log_file in relevant_files:
            if log_file.file_type not in files_by_type:
                files_by_type[log_file.file_type] = []
            files_by_type[log_file.file_type].append(log_file)
        
        # Exportar cada tipo
        for file_type_key, files in files_by_type.items():
            records = []
            
            for log_file in files:
                try:
                    with open(log_file.path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            
                            try:
                                record = json.loads(line)
                                records.append(record)
                            except json.JSONDecodeError:
                                continue
                
                except Exception as e:
                    print(f"Erro ao processar arquivo {log_file.path}: {e}")
            
            if records:
                # Converter para DataFrame e exportar
                df = pd.json_normalize(records)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_filename = f"{file_type_key}_logs_{timestamp}.csv"
                csv_path = output_path / csv_filename
                
                df.to_csv(csv_path, index=False, encoding='utf-8')
                exported_files[file_type_key] = str(csv_path)
        
        return exported_files
    
    def verify_log_integrity(self) -> Dict[str, Any]:
        """Verificar integridade dos arquivos de log."""
        integrity_report = {
            'total_files_checked': 0,
            'valid_files': 0,
            'corrupted_files': [],
            'empty_files': [],
            'parsing_errors': [],
            'summary': {}
        }
        
        log_files = self.get_log_files_info()
        integrity_report['total_files_checked'] = len(log_files)
        
        for log_file in log_files:
            try:
                if log_file.record_count == 0:
                    integrity_report['empty_files'].append(log_file.path)
                    continue
                
                # Verificar se o arquivo pode ser lido e parsed
                valid_records = 0
                invalid_records = 0
                
                with open(log_file.path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        
                        try:
                            json.loads(line)
                            valid_records += 1
                        except json.JSONDecodeError as e:
                            invalid_records += 1
                            integrity_report['parsing_errors'].append({
                                'file': log_file.path,
                                'line': line_num,
                                'error': str(e)
                            })
                
                if invalid_records == 0:
                    integrity_report['valid_files'] += 1
                else:
                    integrity_report['corrupted_files'].append({
                        'file': log_file.path,
                        'valid_records': valid_records,
                        'invalid_records': invalid_records
                    })
            
            except Exception as e:
                integrity_report['corrupted_files'].append({
                    'file': log_file.path,
                    'error': str(e)
                })
        
        # Resumo
        integrity_report['summary'] = {
            'integrity_percentage': (integrity_report['valid_files'] / integrity_report['total_files_checked'] * 100) if integrity_report['total_files_checked'] > 0 else 0,
            'total_corrupted': len(integrity_report['corrupted_files']),
            'total_empty': len(integrity_report['empty_files']),
            'total_parsing_errors': len(integrity_report['parsing_errors'])
        }
        
        return integrity_report
    
    def get_system_health(self) -> Dict[str, Any]:
        """Obter status de saúde do sistema de logging."""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {},
            'disk_usage': {},
            'recent_activity': {},
            'alerts': []
        }
        
        # Verificar espaço em disco
        try:
            disk_usage = shutil.disk_usage(self.log_base_dir)
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_percentage = ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100
            
            health_status['disk_usage'] = {
                'free_gb': round(free_gb, 2),
                'total_gb': round(total_gb, 2),
                'used_percentage': round(used_percentage, 2)
            }
            
            # Alerta de espaço baixo
            min_free_space = self.config.get('cleanup', {}).get('min_free_space_gb', 5)
            if free_gb < min_free_space:
                health_status['alerts'].append({
                    'type': 'disk_space',
                    'severity': 'warning',
                    'message': f"Espaço em disco baixo: {free_gb:.2f}GB disponível"
                })
                health_status['overall_status'] = 'warning'
        
        except Exception as e:
            health_status['alerts'].append({
                'type': 'disk_check',
                'severity': 'error',
                'message': f"Erro ao verificar espaço em disco: {e}"
            })
            health_status['overall_status'] = 'error'
        
        # Verificar atividade recente
        try:
            log_files = self.get_log_files_info()
            recent_files = [f for f in log_files if f.modified_date > datetime.now() - timedelta(hours=1)]
            
            health_status['recent_activity'] = {
                'files_modified_last_hour': len(recent_files),
                'total_log_files': len(log_files),
                'newest_file_age_minutes': 0 if not log_files else int((datetime.now() - log_files[0].modified_date).total_seconds() / 60)
            }
            
            # Alerta se não há atividade recente
            if not recent_files and log_files:
                silence_threshold = self.config.get('alerting', {}).get('alert_conditions', {}).get('logging_silence_minutes', 30)
                newest_age_minutes = health_status['recent_activity']['newest_file_age_minutes']
                
                if newest_age_minutes > silence_threshold:
                    health_status['alerts'].append({
                        'type': 'logging_silence',
                        'severity': 'warning',
                        'message': f"Nenhuma atividade de logging nos últimos {newest_age_minutes} minutos"
                    })
                    if health_status['overall_status'] == 'healthy':
                        health_status['overall_status'] = 'warning'
        
        except Exception as e:
            health_status['alerts'].append({
                'type': 'activity_check',
                'severity': 'error',
                'message': f"Erro ao verificar atividade recente: {e}"
            })
            health_status['overall_status'] = 'error'
        
        # Verificar integridade dos componentes
        try:
            integrity = self.verify_log_integrity()
            health_status['components']['log_integrity'] = {
                'status': 'healthy' if integrity['summary']['integrity_percentage'] > 95 else 'warning',
                'integrity_percentage': integrity['summary']['integrity_percentage'],
                'corrupted_files': integrity['summary']['total_corrupted']
            }
            
            if integrity['summary']['integrity_percentage'] <= 95:
                health_status['alerts'].append({
                    'type': 'log_integrity',
                    'severity': 'warning',
                    'message': f"Integridade dos logs abaixo de 95%: {integrity['summary']['integrity_percentage']:.1f}%"
                })
                if health_status['overall_status'] == 'healthy':
                    health_status['overall_status'] = 'warning'
        
        except Exception as e:
            health_status['components']['log_integrity'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['alerts'].append({
                'type': 'integrity_check',
                'severity': 'error',
                'message': f"Erro ao verificar integridade: {e}"
            })
            health_status['overall_status'] = 'error'
        
        return health_status
    
    def generate_maintenance_report(self) -> Dict[str, Any]:
        """Gerar relatório completo de manutenção."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'system_health': self.get_system_health(),
            'log_analysis': self.analyze_log_patterns(),
            'file_statistics': {},
            'recommendations': []
        }
        
        # Estatísticas detalhadas de arquivos
        log_files = self.get_log_files_info()
        report['file_statistics'] = {
            'total_files': len(log_files),
            'total_size_mb': sum(f.size_mb for f in log_files),
            'oldest_file': min(log_files, key=lambda x: x.created_date).created_date.isoformat() if log_files else None,
            'newest_file': max(log_files, key=lambda x: x.modified_date).modified_date.isoformat() if log_files else None,
            'files_by_type': {}
        }
        
        # Agrupar estatísticas por tipo
        for log_file in log_files:
            file_type = log_file.file_type
            if file_type not in report['file_statistics']['files_by_type']:
                report['file_statistics']['files_by_type'][file_type] = {
                    'count': 0,
                    'total_size_mb': 0,
                    'average_size_mb': 0
                }
            
            report['file_statistics']['files_by_type'][file_type]['count'] += 1
            report['file_statistics']['files_by_type'][file_type]['total_size_mb'] += log_file.size_mb
        
        # Calcular médias
        for file_type, stats in report['file_statistics']['files_by_type'].items():
            if stats['count'] > 0:
                stats['average_size_mb'] = stats['total_size_mb'] / stats['count']
        
        # Gerar recomendações
        recommendations = []
        
        # Recomendação de limpeza
        old_files = [f for f in log_files if f.modified_date < datetime.now() - timedelta(days=30)]
        if old_files:
            total_old_size = sum(f.size_mb for f in old_files)
            recommendations.append({
                'type': 'cleanup',
                'priority': 'medium',
                'description': f"Considere limpar {len(old_files)} arquivos antigos (>{total_old_size:.1f}MB)",
                'action': 'Execute cleanup_old_logs() para liberar espaço'
            })
        
        # Recomendação de compressão
        large_files = [f for f in log_files if f.size_mb > 50 and not f.path.endswith('.gz')]
        if large_files:
            recommendations.append({
                'type': 'compression',
                'priority': 'low',
                'description': f"Considere comprimir {len(large_files)} arquivos grandes",
                'action': 'Execute compress_old_logs() para economizar espaço'
            })
        
        # Recomendação de integridade
        if report['system_health']['components'].get('log_integrity', {}).get('corrupted_files', 0) > 0:
            recommendations.append({
                'type': 'integrity',
                'priority': 'high',
                'description': "Arquivos corrompidos detectados",
                'action': 'Verifique e repare arquivos corrompidos'
            })
        
        report['recommendations'] = recommendations
        
        return report


def main():
    """Função principal para execução via linha de comando."""
    parser = argparse.ArgumentParser(description='Utilitários do Sistema de Logging')
    parser.add_argument('--config', default='logging_config.yaml', help='Caminho para arquivo de configuração')
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis')
    
    # Comando de limpeza
    cleanup_parser = subparsers.add_parser('cleanup', help='Limpar logs antigos')
    cleanup_parser.add_argument('--days', type=int, help='Dias para manter (padrão: configuração)')
    
    # Comando de compressão
    compress_parser = subparsers.add_parser('compress', help='Comprimir logs antigos')
    compress_parser.add_argument('--days', type=int, help='Idade mínima para compressão (padrão: configuração)')
    
    # Comando de análise
    analyze_parser = subparsers.add_parser('analyze', help='Analisar padrões nos logs')
    analyze_parser.add_argument('--type', choices=['all', 'comprehensive', 'detailed', 'trading'], default='all')
    analyze_parser.add_argument('--days', type=int, default=7, help='Dias para analisar')
    
    # Comando de exportação
    export_parser = subparsers.add_parser('export', help='Exportar logs para CSV')
    export_parser.add_argument('--output', required=True, help='Diretório de saída')
    export_parser.add_parser('--type', choices=['all', 'comprehensive', 'detailed', 'trading'], default='all')
    export_parser.add_argument('--days', type=int, default=7, help='Dias para exportar')
    
    # Comando de verificação
    verify_parser = subparsers.add_parser('verify', help='Verificar integridade dos logs')
    
    # Comando de saúde
    health_parser = subparsers.add_parser('health', help='Verificar saúde do sistema')
    
    # Comando de relatório
    report_parser = subparsers.add_parser('report', help='Gerar relatório de manutenção')
    report_parser.add_argument('--output', help='Arquivo de saída para o relatório')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Inicializar utilitários
    utils = LoggingUtilities(args.config)
    
    try:
        if args.command == 'cleanup':
            result = utils.cleanup_old_logs(args.days)
            print(f"Limpeza concluída: {result['total_files_deleted']} arquivos deletados, {result['total_space_freed_mb']:.2f}MB liberados")
            if result['errors']:
                print(f"Erros: {result['errors']}")
        
        elif args.command == 'compress':
            result = utils.compress_old_logs(args.days)
            print(f"Compressão concluída: {result['total_files_compressed']} arquivos comprimidos, {result['total_space_saved_mb']:.2f}MB economizados")
            if result['errors']:
                print(f"Erros: {result['errors']}")
        
        elif args.command == 'analyze':
            result = utils.analyze_log_patterns(args.type, args.days)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        elif args.command == 'export':
            result = utils.export_logs_to_csv(args.output, getattr(args, 'type', 'all'), args.days)
            print(f"Exportação concluída: {len(result)} arquivos criados")
            for file_type, path in result.items():
                print(f"  {file_type}: {path}")
        
        elif args.command == 'verify':
            result = utils.verify_log_integrity()
            print(f"Verificação concluída: {result['summary']['integrity_percentage']:.1f}% de integridade")
            if result['corrupted_files']:
                print(f"Arquivos corrompidos: {len(result['corrupted_files'])}")
            if result['parsing_errors']:
                print(f"Erros de parsing: {len(result['parsing_errors'])}")
        
        elif args.command == 'health':
            result = utils.get_system_health()
            print(f"Status do sistema: {result['overall_status']}")
            if result['alerts']:
                print("Alertas:")
                for alert in result['alerts']:
                    print(f"  [{alert['severity']}] {alert['message']}")
        
        elif args.command == 'report':
            result = utils.generate_maintenance_report()
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Relatório salvo em: {args.output}")
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
    
    except Exception as e:
        print(f"Erro ao executar comando: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())