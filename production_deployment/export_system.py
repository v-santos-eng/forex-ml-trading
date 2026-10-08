import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import json
import csv
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import threading

class ExportFormat(Enum):
    """Formatos de exportação disponíveis"""
    CSV = "csv"
    EXCEL = "xlsx"
    JSON = "json"
    PDF = "pdf"
    TXT = "txt"
    HTML = "html"

@dataclass
class ExportConfig:
    """Configuração de exportação"""
    format: ExportFormat
    include_headers: bool = True
    date_range: Optional[tuple] = None
    filters: Dict[str, Any] = None
    custom_fields: List[str] = None
    
class DataExporter:
    """Classe para exportação de dados"""
    
    def __init__(self):
        self.supported_formats = {
            ExportFormat.CSV: self._export_csv,
            ExportFormat.EXCEL: self._export_excel,
            ExportFormat.JSON: self._export_json,
            ExportFormat.TXT: self._export_txt,
            ExportFormat.HTML: self._export_html
        }
    
    def export_data(self, data: List[Dict], filepath: str, config: ExportConfig) -> bool:
        """Exporta dados no formato especificado"""
        try:
            if not data:
                raise ValueError("Nenhum dado para exportar")
            
            # Aplicar filtros se especificados
            filtered_data = self._apply_filters(data, config)
            
            # Aplicar range de datas se especificado
            if config.date_range:
                filtered_data = self._apply_date_filter(filtered_data, config.date_range)
            
            # Selecionar campos customizados se especificados
            if config.custom_fields:
                filtered_data = self._select_fields(filtered_data, config.custom_fields)
            
            # Exportar no formato especificado
            export_func = self.supported_formats.get(config.format)
            if not export_func:
                raise ValueError(f"Formato {config.format.value} não suportado")
            
            return export_func(filtered_data, filepath, config)
            
        except Exception as e:
            print(f"Erro na exportação: {e}")
            return False
    
    def _apply_filters(self, data: List[Dict], config: ExportConfig) -> List[Dict]:
        """Aplica filtros aos dados"""
        if not config.filters:
            return data
        
        filtered = []
        for item in data:
            include = True
            for key, value in config.filters.items():
                if key in item and item[key] != value:
                    include = False
                    break
            if include:
                filtered.append(item)
        
        return filtered
    
    def _apply_date_filter(self, data: List[Dict], date_range: tuple) -> List[Dict]:
        """Aplica filtro de data"""
        start_date, end_date = date_range
        filtered = []
        
        for item in data:
            # Procurar campo de data
            date_field = None
            for key in ['timestamp', 'date', 'created_at', 'time']:
                if key in item:
                    date_field = key
                    break
            
            if date_field:
                try:
                    item_date = pd.to_datetime(item[date_field])
                    if start_date <= item_date <= end_date:
                        filtered.append(item)
                except:
                    filtered.append(item)  # Incluir se não conseguir parsear
            else:
                filtered.append(item)  # Incluir se não tiver campo de data
        
        return filtered
    
    def _select_fields(self, data: List[Dict], fields: List[str]) -> List[Dict]:
        """Seleciona apenas campos especificados"""
        return [{k: item.get(k, '') for k in fields} for item in data]
    
    def _export_csv(self, data: List[Dict], filepath: str, config: ExportConfig) -> bool:
        """Exporta para CSV"""
        try:
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            return True
        except Exception as e:
            print(f"Erro ao exportar CSV: {e}")
            return False
    
    def _export_excel(self, data: List[Dict], filepath: str, config: ExportConfig) -> bool:
        """Exporta para Excel"""
        try:
            df = pd.DataFrame(data)
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Dados', index=False)
            return True
        except Exception as e:
            print(f"Erro ao exportar Excel: {e}")
            return False
    
    def _export_json(self, data: List[Dict], filepath: str, config: ExportConfig) -> bool:
        """Exporta para JSON"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            print(f"Erro ao exportar JSON: {e}")
            return False
    
    def _export_txt(self, data: List[Dict], filepath: str, config: ExportConfig) -> bool:
        """Exporta para TXT"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if config.include_headers and data:
                    headers = list(data[0].keys())
                    f.write('\t'.join(headers) + '\n')
                
                for item in data:
                    values = [str(item.get(k, '')) for k in item.keys()]
                    f.write('\t'.join(values) + '\n')
            return True
        except Exception as e:
            print(f"Erro ao exportar TXT: {e}")
            return False
    
    def _export_html(self, data: List[Dict], filepath: str, config: ExportConfig) -> bool:
        """Exporta para HTML"""
        try:
            df = pd.DataFrame(data)
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Relatório de Dados</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .header {{ margin-bottom: 20px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Relatório de Dados</h1>
                    <p>Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                    <p>Total de registros: {len(data)}</p>
                </div>
                {df.to_html(index=False, classes='data-table')}
            </body>
            </html>
            """
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return True
        except Exception as e:
            print(f"Erro ao exportar HTML: {e}")
            return False

class ReportGenerator:
    """Gerador de relatórios personalizados"""
    
    def __init__(self):
        self.templates = {
            'trading_summary': self._generate_trading_summary,
            'performance_report': self._generate_performance_report,
            'risk_analysis': self._generate_risk_analysis,
            'daily_summary': self._generate_daily_summary
        }
    
    def generate_report(self, report_type: str, data: Dict[str, Any], filepath: str) -> bool:
        """Gera relatório específico"""
        try:
            generator = self.templates.get(report_type)
            if not generator:
                raise ValueError(f"Tipo de relatório '{report_type}' não encontrado")
            
            content = generator(data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        except Exception as e:
            print(f"Erro ao gerar relatório: {e}")
            return False
    
    def _generate_trading_summary(self, data: Dict[str, Any]) -> str:
        """Gera resumo de trading"""
        positions = data.get('positions', [])
        trades = data.get('trades', [])
        
        total_positions = len(positions)
        total_trades = len(trades)
        
        profit_trades = [t for t in trades if t.get('profit', 0) > 0]
        loss_trades = [t for t in trades if t.get('profit', 0) < 0]
        
        win_rate = (len(profit_trades) / total_trades * 100) if total_trades > 0 else 0
        total_profit = sum(t.get('profit', 0) for t in trades)
        
        return f"""
        RELATÓRIO DE TRADING
        ==================
        
        Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        
        RESUMO GERAL:
        - Total de Posições Abertas: {total_positions}
        - Total de Trades Executados: {total_trades}
        - Taxa de Acerto: {win_rate:.2f}%
        - Lucro Total: ${total_profit:.2f}
        
        DETALHAMENTO:
        - Trades Lucrativos: {len(profit_trades)}
        - Trades com Prejuízo: {len(loss_trades)}
        - Maior Lucro: ${max([t.get('profit', 0) for t in profit_trades], default=0):.2f}
        - Maior Prejuízo: ${min([t.get('profit', 0) for t in loss_trades], default=0):.2f}
        
        POSIÇÕES ABERTAS:
        {self._format_positions(positions)}
        """
    
    def _generate_performance_report(self, data: Dict[str, Any]) -> str:
        """Gera relatório de performance"""
        metrics = data.get('metrics', {})
        
        return f"""
        RELATÓRIO DE PERFORMANCE
        =======================
        
        Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        
        MÉTRICAS PRINCIPAIS:
        - Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}
        - Drawdown Máximo: {metrics.get('max_drawdown', 'N/A')}%
        - Retorno Total: {metrics.get('total_return', 'N/A')}%
        - Volatilidade: {metrics.get('volatility', 'N/A')}%
        
        ANÁLISE DE RISCO:
        - VaR (95%): {metrics.get('var_95', 'N/A')}
        - Expected Shortfall: {metrics.get('expected_shortfall', 'N/A')}
        - Beta: {metrics.get('beta', 'N/A')}
        """
    
    def _generate_risk_analysis(self, data: Dict[str, Any]) -> str:
        """Gera análise de risco"""
        risk_data = data.get('risk', {})
        
        return f"""
        ANÁLISE DE RISCO
        ===============
        
        Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        
        EXPOSIÇÃO ATUAL:
        - Exposição Total: {risk_data.get('total_exposure', 'N/A')}
        - Margem Utilizada: {risk_data.get('margin_used', 'N/A')}%
        - Margem Livre: {risk_data.get('free_margin', 'N/A')}
        
        LIMITES DE RISCO:
        - Limite por Trade: {risk_data.get('trade_limit', 'N/A')}
        - Limite Diário: {risk_data.get('daily_limit', 'N/A')}
        - Stop Loss Médio: {risk_data.get('avg_stop_loss', 'N/A')}
        """
    
    def _generate_daily_summary(self, data: Dict[str, Any]) -> str:
        """Gera resumo diário"""
        daily_data = data.get('daily', {})
        
        return f"""
        RESUMO DIÁRIO
        ============
        
        Data: {datetime.now().strftime('%d/%m/%Y')}
        
        ATIVIDADE DO DIA:
        - Trades Executados: {daily_data.get('trades_count', 0)}
        - Volume Negociado: {daily_data.get('volume', 'N/A')}
        - P&L do Dia: ${daily_data.get('daily_pnl', 0):.2f}
        
        SINAIS GERADOS:
        - Total de Sinais: {daily_data.get('signals_count', 0)}
        - Sinais de Compra: {daily_data.get('buy_signals', 0)}
        - Sinais de Venda: {daily_data.get('sell_signals', 0)}
        """
    
    def _format_positions(self, positions: List[Dict]) -> str:
        """Formata lista de posições"""
        if not positions:
            return "Nenhuma posição aberta"
        
        formatted = []
        for pos in positions[:10]:  # Limitar a 10 posições
            symbol = pos.get('symbol', 'N/A')
            type_pos = pos.get('type', 'N/A')
            volume = pos.get('volume', 0)
            profit = pos.get('profit', 0)
            formatted.append(f"  {symbol} - {type_pos} - Vol: {volume} - P&L: ${profit:.2f}")
        
        if len(positions) > 10:
            formatted.append(f"  ... e mais {len(positions) - 10} posições")
        
        return "\n".join(formatted)

class ExportDialog:
    """Dialog para configuração de exportação"""
    
    def __init__(self, parent, data_sources: Dict[str, callable]):
        self.parent = parent
        self.data_sources = data_sources
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Exportar Dados")
        self.dialog.geometry("500x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self._create_widgets()
        self._center_dialog()
    
    def _create_widgets(self):
        """Cria widgets do dialog"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Fonte de dados
        ttk.Label(main_frame, text="Fonte de Dados:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.source_var = tk.StringVar()
        source_combo = ttk.Combobox(main_frame, textvariable=self.source_var, 
                                   values=list(self.data_sources.keys()), state="readonly")
        source_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Formato de exportação
        ttk.Label(main_frame, text="Formato:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.format_var = tk.StringVar()
        format_combo = ttk.Combobox(main_frame, textvariable=self.format_var,
                                   values=[f.value for f in ExportFormat], state="readonly")
        format_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Range de datas
        date_frame = ttk.LabelFrame(main_frame, text="Filtro de Data", padding="5")
        date_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.use_date_filter = tk.BooleanVar()
        ttk.Checkbutton(date_frame, text="Usar filtro de data", 
                       variable=self.use_date_filter).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Label(date_frame, text="De:").grid(row=1, column=0, sticky=tk.W)
        self.start_date = tk.StringVar(value=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
        ttk.Entry(date_frame, textvariable=self.start_date).grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        ttk.Label(date_frame, text="Até:").grid(row=2, column=0, sticky=tk.W)
        self.end_date = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        ttk.Entry(date_frame, textvariable=self.end_date).grid(row=2, column=1, sticky=(tk.W, tk.E))
        
        # Opções adicionais
        options_frame = ttk.LabelFrame(main_frame, text="Opções", padding="5")
        options_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.include_headers = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Incluir cabeçalhos", 
                       variable=self.include_headers).grid(row=0, column=0, sticky=tk.W)
        
        # Botões
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Exportar", 
                  command=self._export).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                  command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Configurar grid weights
        main_frame.columnconfigure(1, weight=1)
        date_frame.columnconfigure(1, weight=1)
    
    def _center_dialog(self):
        """Centraliza o dialog"""
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
    
    def _export(self):
        """Executa exportação"""
        try:
            # Validar seleções
            if not self.source_var.get():
                messagebox.showerror("Erro", "Selecione uma fonte de dados")
                return
            
            if not self.format_var.get():
                messagebox.showerror("Erro", "Selecione um formato")
                return
            
            # Obter dados
            data_func = self.data_sources[self.source_var.get()]
            data = data_func()
            
            if not data:
                messagebox.showwarning("Aviso", "Nenhum dado disponível para exportação")
                return
            
            # Configurar exportação
            config = ExportConfig(
                format=ExportFormat(self.format_var.get()),
                include_headers=self.include_headers.get()
            )
            
            # Aplicar filtro de data se selecionado
            if self.use_date_filter.get():
                try:
                    start = pd.to_datetime(self.start_date.get())
                    end = pd.to_datetime(self.end_date.get())
                    config.date_range = (start, end)
                except:
                    messagebox.showerror("Erro", "Formato de data inválido")
                    return
            
            # Selecionar arquivo
            format_ext = config.format.value
            filetypes = {
                'csv': [('CSV files', '*.csv')],
                'xlsx': [('Excel files', '*.xlsx')],
                'json': [('JSON files', '*.json')],
                'txt': [('Text files', '*.txt')],
                'html': [('HTML files', '*.html')]
            }
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=f".{format_ext}",
                filetypes=filetypes.get(format_ext, [('All files', '*.*')])
            )
            
            if not filepath:
                return
            
            # Executar exportação
            exporter = DataExporter()
            
            def export_thread():
                success = exporter.export_data(data, filepath, config)
                
                # Atualizar UI na thread principal
                self.dialog.after(0, lambda: self._export_complete(success, filepath))
            
            # Executar em thread separada
            threading.Thread(target=export_thread, daemon=True).start()
            
            # Mostrar progresso
            messagebox.showinfo("Exportação", "Exportação iniciada...")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na exportação: {str(e)}")
    
    def _export_complete(self, success: bool, filepath: str):
        """Callback para conclusão da exportação"""
        if success:
            messagebox.showinfo("Sucesso", f"Dados exportados para:\n{filepath}")
            self.dialog.destroy()
        else:
            messagebox.showerror("Erro", "Falha na exportação")

class ExportManager:
    """Gerenciador principal de exportações"""
    
    def __init__(self, dashboard_ref):
        self.dashboard = dashboard_ref
        self.exporter = DataExporter()
        self.report_generator = ReportGenerator()
        
        # Definir fontes de dados disponíveis
        self.data_sources = {
            "Posições Abertas": self._get_positions_data,
            "Histórico de Trades": self._get_trades_data,
            "Logs do Sistema": self._get_logs_data,
            "Sinais de Trading": self._get_signals_data,
            "Métricas de Performance": self._get_metrics_data
        }
    
    def show_export_dialog(self):
        """Mostra dialog de exportação"""
        dialog = ExportDialog(self.dashboard.root, self.data_sources)
        return dialog.result
    
    def quick_export_positions(self):
        """Exportação rápida de posições"""
        try:
            data = self._get_positions_data()
            if not data:
                messagebox.showwarning("Aviso", "Nenhuma posição para exportar")
                return
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[('CSV files', '*.csv')],
                initialvalue=f"posicoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if filepath:
                config = ExportConfig(format=ExportFormat.CSV)
                success = self.exporter.export_data(data, filepath, config)
                
                if success:
                    messagebox.showinfo("Sucesso", f"Posições exportadas para:\n{filepath}")
                else:
                    messagebox.showerror("Erro", "Falha na exportação")
                    
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na exportação: {str(e)}")
    
    def generate_trading_report(self):
        """Gera relatório de trading"""
        try:
            data = {
                'positions': self._get_positions_data(),
                'trades': self._get_trades_data(),
                'metrics': self._get_metrics_data()
            }
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[('Text files', '*.txt')],
                initialvalue=f"relatorio_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            
            if filepath:
                success = self.report_generator.generate_report('trading_summary', data, filepath)
                
                if success:
                    messagebox.showinfo("Sucesso", f"Relatório gerado:\n{filepath}")
                else:
                    messagebox.showerror("Erro", "Falha na geração do relatório")
                    
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na geração: {str(e)}")
    
    def _get_positions_data(self) -> List[Dict]:
        """Obtém dados de posições"""
        try:
            # Simular dados de posições (integrar com MT5 real)
            return [
                {
                    'symbol': 'EURUSD',
                    'type': 'BUY',
                    'volume': 0.1,
                    'open_price': 1.0850,
                    'current_price': 1.0865,
                    'profit': 15.0,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'symbol': 'GBPUSD',
                    'type': 'SELL',
                    'volume': 0.05,
                    'open_price': 1.2650,
                    'current_price': 1.2635,
                    'profit': 7.5,
                    'timestamp': datetime.now().isoformat()
                }
            ]
        except Exception as e:
            print(f"Erro ao obter posições: {e}")
            return []
    
    def _get_trades_data(self) -> List[Dict]:
        """Obtém dados de trades"""
        try:
            # Simular dados de trades (integrar com MT5 real)
            return [
                {
                    'symbol': 'EURUSD',
                    'type': 'BUY',
                    'volume': 0.1,
                    'open_price': 1.0820,
                    'close_price': 1.0850,
                    'profit': 30.0,
                    'open_time': (datetime.now() - timedelta(hours=2)).isoformat(),
                    'close_time': datetime.now().isoformat()
                }
            ]
        except Exception as e:
            print(f"Erro ao obter trades: {e}")
            return []
    
    def _get_logs_data(self) -> List[Dict]:
        """Obtém dados de logs"""
        try:
            # Simular dados de logs
            return [
                {
                    'timestamp': datetime.now().isoformat(),
                    'level': 'INFO',
                    'message': 'Sistema iniciado com sucesso',
                    'module': 'dashboard'
                },
                {
                    'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
                    'level': 'INFO',
                    'message': 'Dados MT5 atualizados com sucesso',
                    'module': 'mt5_connector'
                }
            ]
        except Exception as e:
            print(f"Erro ao obter logs: {e}")
            return []
    
    def _get_signals_data(self) -> List[Dict]:
        """Obtém dados de sinais"""
        try:
            # Simular dados de sinais
            return [
                {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': 'EURUSD',
                    'signal': 'BUY',
                    'confidence': 0.85,
                    'price': 1.0865,
                    'source': 'ML_Model'
                }
            ]
        except Exception as e:
            print(f"Erro ao obter sinais: {e}")
            return []
    
    def _get_metrics_data(self) -> List[Dict]:
        """Obtém dados de métricas"""
        try:
            # Simular dados de métricas
            return [
                {
                    'metric': 'sharpe_ratio',
                    'value': 1.25,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'metric': 'max_drawdown',
                    'value': -5.2,
                    'timestamp': datetime.now().isoformat()
                }
            ]
        except Exception as e:
            print(f"Erro ao obter métricas: {e}")
            return []