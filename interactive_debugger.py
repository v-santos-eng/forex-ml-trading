#!/usr/bin/env python3
"""
Sistema de Debugging Interativo para Aplicação Forex ML
Permite inspeção de variáveis em tempo real, breakpoints condicionais e análise de stack traces.
"""

import os
import sys
import json
import time
import threading
import traceback
import inspect
import pdb
import code
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import ast
import dis
from contextlib import contextmanager
import signal
import psutil
import gc
from advanced_logging_system import get_logging_system

@dataclass
class StackFrame:
    """Informações de um frame da pilha"""
    filename: str
    function_name: str
    line_number: int
    code_context: List[str]
    local_variables: Dict[str, Any]
    global_variables: Dict[str, Any]
    frame_id: str
    
@dataclass
class DebugSession:
    """Sessão de debugging ativa"""
    session_id: str
    start_time: datetime
    current_frame: Optional[StackFrame] = None
    stack_frames: List[StackFrame] = field(default_factory=list)
    breakpoint_hits: int = 0
    commands_executed: List[str] = field(default_factory=list)
    variables_watched: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

class VariableInspector:
    """Inspetor de variáveis com análise detalhada"""
    
    def __init__(self):
        self.watch_list = set()
        self.variable_history = defaultdict(deque)
        self.max_history = 100
    
    def add_watch(self, variable_name: str):
        """Adiciona variável à lista de observação"""
        self.watch_list.add(variable_name)
    
    def remove_watch(self, variable_name: str):
        """Remove variável da lista de observação"""
        self.watch_list.discard(variable_name)
    
    def inspect_variable(self, var_name: str, var_value: Any, frame_locals: Dict[str, Any]) -> Dict[str, Any]:
        """Inspeciona uma variável detalhadamente"""
        inspection = {
            'name': var_name,
            'type': type(var_value).__name__,
            'value': None,
            'size': 0,
            'memory_usage': 0,
            'attributes': [],
            'methods': [],
            'is_callable': callable(var_value),
            'module': getattr(type(var_value), '__module__', 'unknown'),
            'doc': getattr(var_value, '__doc__', None)
        }
        
        try:
            # Valor como string (limitado)
            str_value = str(var_value)
            if len(str_value) > 500:
                inspection['value'] = str_value[:500] + '... (truncated)'
                inspection['truncated'] = True
            else:
                inspection['value'] = str_value
                inspection['truncated'] = False
            
            inspection['size'] = len(str_value)
            
            # Uso de memória
            try:
                inspection['memory_usage'] = sys.getsizeof(var_value)
            except:
                inspection['memory_usage'] = 0
            
            # Atributos e métodos
            if hasattr(var_value, '__dict__'):
                inspection['attributes'] = [attr for attr in dir(var_value) 
                                          if not attr.startswith('_') and not callable(getattr(var_value, attr, None))]
                inspection['methods'] = [method for method in dir(var_value) 
                                       if not method.startswith('_') and callable(getattr(var_value, method, None))]
            
            # Informações específicas por tipo
            if isinstance(var_value, (list, tuple, set)):
                inspection['length'] = len(var_value)
                if len(var_value) > 0:
                    inspection['element_types'] = list(set(type(item).__name__ for item in var_value))
            
            elif isinstance(var_value, dict):
                inspection['length'] = len(var_value)
                inspection['keys'] = list(var_value.keys())[:10]  # Primeiras 10 chaves
                if len(var_value) > 10:
                    inspection['keys'].append('... (more keys)')
            
            elif hasattr(var_value, 'shape'):  # Arrays numpy/pandas
                inspection['shape'] = getattr(var_value, 'shape', None)
                inspection['dtype'] = getattr(var_value, 'dtype', None)
            
        except Exception as e:
            inspection['error'] = str(e)
            inspection['value'] = '<inspection failed>'
        
        # Adicionar ao histórico se estiver sendo observada
        if var_name in self.watch_list:
            self.variable_history[var_name].append({
                'timestamp': datetime.now(),
                'value': inspection['value'],
                'type': inspection['type'],
                'size': inspection['size']
            })
            
            # Limitar histórico
            if len(self.variable_history[var_name]) > self.max_history:
                self.variable_history[var_name].popleft()
        
        return inspection
    
    def get_variable_history(self, var_name: str) -> List[Dict[str, Any]]:
        """Retorna histórico de uma variável"""
        return list(self.variable_history.get(var_name, []))
    
    def analyze_memory_usage(self, frame_locals: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa uso de memória das variáveis"""
        memory_analysis = {
            'total_variables': len(frame_locals),
            'memory_by_variable': {},
            'largest_variables': [],
            'total_memory': 0
        }
        
        variable_sizes = []
        
        for name, value in frame_locals.items():
            if not name.startswith('_'):
                try:
                    size = sys.getsizeof(value)
                    memory_analysis['memory_by_variable'][name] = {
                        'size': size,
                        'type': type(value).__name__
                    }
                    variable_sizes.append((name, size, type(value).__name__))
                    memory_analysis['total_memory'] += size
                except:
                    pass
        
        # Ordenar por tamanho
        variable_sizes.sort(key=lambda x: x[1], reverse=True)
        memory_analysis['largest_variables'] = variable_sizes[:10]
        
        return memory_analysis

class InteractiveDebugger:
    """Debugger interativo com funcionalidades avançadas"""
    
    def __init__(self):
        self.logging_system = get_logging_system()
        self.logger = self.logging_system.get_logger('forex_ml.debug')
        self.variable_inspector = VariableInspector()
        self.active_sessions = {}
        self.command_history = deque(maxlen=1000)
        self.is_debugging = False
        self.debug_commands = self._setup_debug_commands()
        
        # Configurar handler de sinal para debugging remoto (apenas Unix/Linux)
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, self._signal_debug_handler)
        else:
            self.logger.debug("Sinais SIGUSR1 não disponíveis no Windows")
    
    def _setup_debug_commands(self) -> Dict[str, Callable]:
        """Configura comandos de debugging disponíveis"""
        return {
            'help': self._cmd_help,
            'h': self._cmd_help,
            'list': self._cmd_list_variables,
            'l': self._cmd_list_variables,
            'inspect': self._cmd_inspect_variable,
            'i': self._cmd_inspect_variable,
            'watch': self._cmd_watch_variable,
            'w': self._cmd_watch_variable,
            'unwatch': self._cmd_unwatch_variable,
            'uw': self._cmd_unwatch_variable,
            'stack': self._cmd_show_stack,
            's': self._cmd_show_stack,
            'frame': self._cmd_change_frame,
            'f': self._cmd_change_frame,
            'memory': self._cmd_memory_analysis,
            'm': self._cmd_memory_analysis,
            'history': self._cmd_variable_history,
            'hist': self._cmd_variable_history,
            'execute': self._cmd_execute_code,
            'exec': self._cmd_execute_code,
            'continue': self._cmd_continue,
            'c': self._cmd_continue,
            'quit': self._cmd_quit,
            'q': self._cmd_quit,
            'breakpoints': self._cmd_list_breakpoints,
            'bp': self._cmd_list_breakpoints,
            'add_bp': self._cmd_add_breakpoint,
            'remove_bp': self._cmd_remove_breakpoint,
            'enable_bp': self._cmd_enable_breakpoint,
            'disable_bp': self._cmd_disable_breakpoint
        }
    
    def _signal_debug_handler(self, signum, frame):
        """Handler para sinal de debugging"""
        self.start_interactive_session(frame)
    
    def start_interactive_session(self, frame=None):
        """Inicia sessão interativa de debugging"""
        if frame is None:
            frame = inspect.currentframe().f_back
        
        session_id = f"debug_{int(time.time())}"
        session = DebugSession(
            session_id=session_id,
            start_time=datetime.now()
        )
        
        # Capturar stack frames
        session.stack_frames = self._capture_stack_frames(frame)
        if session.stack_frames:
            session.current_frame = session.stack_frames[0]
        
        self.active_sessions[session_id] = session
        self.is_debugging = True
        
        self.logger.info(f"Iniciando sessão de debugging: {session_id}")
        
        try:
            self._interactive_loop(session)
        finally:
            self.is_debugging = False
            session.is_active = False
            self.logger.info(f"Finalizando sessão de debugging: {session_id}")
    
    def _capture_stack_frames(self, frame) -> List[StackFrame]:
        """Captura informações dos frames da pilha"""
        frames = []
        current_frame = frame
        frame_id = 0
        
        while current_frame and frame_id < 20:  # Limitar a 20 frames
            try:
                code_obj = current_frame.f_code
                filename = code_obj.co_filename
                function_name = code_obj.co_name
                line_number = current_frame.f_lineno
                
                # Capturar contexto do código
                code_context = []
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        start_line = max(0, line_number - 3)
                        end_line = min(len(lines), line_number + 2)
                        code_context = [f"{i+1:4d}: {lines[i].rstrip()}" 
                                      for i in range(start_line, end_line)]
                except:
                    code_context = [f"{line_number}: <código não disponível>"]
                
                # Capturar variáveis (limitado para evitar problemas de memória)
                local_vars = {}
                global_vars = {}
                
                try:
                    for name, value in current_frame.f_locals.items():
                        if not name.startswith('__'):
                            try:
                                str_value = str(value)[:200]
                                local_vars[name] = {
                                    'type': type(value).__name__,
                                    'value': str_value
                                }
                            except:
                                local_vars[name] = {
                                    'type': type(value).__name__,
                                    'value': '<não serializável>'
                                }
                    
                    # Capturar algumas variáveis globais importantes
                    for name in ['__name__', '__file__']:
                        if name in current_frame.f_globals:
                            global_vars[name] = str(current_frame.f_globals[name])
                
                except Exception as e:
                    self.logger.warning(f"Erro ao capturar variáveis do frame {frame_id}: {e}")
                
                stack_frame = StackFrame(
                    filename=filename,
                    function_name=function_name,
                    line_number=line_number,
                    code_context=code_context,
                    local_variables=local_vars,
                    global_variables=global_vars,
                    frame_id=str(frame_id)
                )
                
                frames.append(stack_frame)
                
            except Exception as e:
                self.logger.warning(f"Erro ao processar frame {frame_id}: {e}")
            
            current_frame = current_frame.f_back
            frame_id += 1
        
        return frames
    
    def _interactive_loop(self, session: DebugSession):
        """Loop principal da sessão interativa"""
        print(f"\n=== Debugger Interativo Forex ML ===")
        print(f"Sessão: {session.session_id}")
        print(f"Digite 'help' para ver comandos disponíveis")
        print(f"Digite 'quit' para sair\n")
        
        if session.current_frame:
            self._show_current_context(session.current_frame)
        
        while session.is_active:
            try:
                command = input("(forex-debug) ").strip()
                
                if not command:
                    continue
                
                self.command_history.append(command)
                session.commands_executed.append(command)
                
                # Parsear comando
                parts = command.split()
                cmd_name = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                if cmd_name in self.debug_commands:
                    try:
                        result = self.debug_commands[cmd_name](session, args)
                        if result is False:  # Comando de saída
                            break
                    except Exception as e:
                        print(f"Erro ao executar comando '{cmd_name}': {e}")
                        self.logger.error(f"Erro no comando debug '{cmd_name}': {e}")
                else:
                    print(f"Comando desconhecido: {cmd_name}. Digite 'help' para ver comandos disponíveis.")
            
            except (EOFError, KeyboardInterrupt):
                print("\nSaindo do debugger...")
                break
            except Exception as e:
                print(f"Erro no loop de debugging: {e}")
                self.logger.error(f"Erro no loop de debugging: {e}")
    
    def _show_current_context(self, frame: StackFrame):
        """Mostra contexto atual do frame"""
        print(f"\nFrame atual: {frame.function_name} ({Path(frame.filename).name}:{frame.line_number})")
        print("Código:")
        for line in frame.code_context:
            marker = ">>>" if str(frame.line_number) in line else "   "
            print(f"{marker} {line}")
        print()
    
    # Comandos de debugging
    def _cmd_help(self, session: DebugSession, args: List[str]):
        """Mostra ajuda dos comandos"""
        help_text = """
Comandos disponíveis:

  help, h                    - Mostra esta ajuda
  list, l                    - Lista variáveis locais
  inspect <var>, i <var>     - Inspeciona variável detalhadamente
  watch <var>, w <var>       - Adiciona variável à observação
  unwatch <var>, uw <var>    - Remove variável da observação
  stack, s                   - Mostra stack trace
  frame <n>, f <n>           - Muda para frame N
  memory, m                  - Análise de uso de memória
  history <var>, hist <var>  - Histórico de uma variável
  execute <code>, exec <code>- Executa código Python
  continue, c                - Continua execução
  quit, q                    - Sai do debugger
  
  breakpoints, bp            - Lista breakpoints
  add_bp <file> <line>       - Adiciona breakpoint
  remove_bp <id>             - Remove breakpoint
  enable_bp <id>             - Habilita breakpoint
  disable_bp <id>            - Desabilita breakpoint
"""
        print(help_text)
    
    def _cmd_list_variables(self, session: DebugSession, args: List[str]):
        """Lista variáveis do frame atual"""
        if not session.current_frame:
            print("Nenhum frame ativo")
            return
        
        print(f"\nVariáveis locais em {session.current_frame.function_name}:")
        print("-" * 50)
        
        for name, info in session.current_frame.local_variables.items():
            watch_marker = "[W]" if name in self.variable_inspector.watch_list else "   "
            print(f"{watch_marker} {name:20} = {info['value'][:50]}... ({info['type']})")
    
    def _cmd_inspect_variable(self, session: DebugSession, args: List[str]):
        """Inspeciona variável detalhadamente"""
        if not args:
            print("Uso: inspect <nome_da_variável>")
            return
        
        var_name = args[0]
        
        if not session.current_frame:
            print("Nenhum frame ativo")
            return
        
        if var_name not in session.current_frame.local_variables:
            print(f"Variável '{var_name}' não encontrada no frame atual")
            return
        
        # Para inspeção completa, precisaríamos do valor real da variável
        # Por enquanto, mostramos as informações disponíveis
        var_info = session.current_frame.local_variables[var_name]
        
        print(f"\nInspeção da variável '{var_name}':")
        print("-" * 40)
        print(f"Tipo: {var_info['type']}")
        print(f"Valor: {var_info['value']}")
        
        if var_name in self.variable_inspector.watch_list:
            history = self.variable_inspector.get_variable_history(var_name)
            if history:
                print(f"\nHistórico (últimas {len(history)} alterações):")
                for i, entry in enumerate(history[-5:]):
                    print(f"  {i+1}. {entry['timestamp'].strftime('%H:%M:%S')} - {entry['value'][:30]}...")
    
    def _cmd_watch_variable(self, session: DebugSession, args: List[str]):
        """Adiciona variável à observação"""
        if not args:
            print("Uso: watch <nome_da_variável>")
            return
        
        var_name = args[0]
        self.variable_inspector.add_watch(var_name)
        print(f"Variável '{var_name}' adicionada à observação")
    
    def _cmd_unwatch_variable(self, session: DebugSession, args: List[str]):
        """Remove variável da observação"""
        if not args:
            print("Uso: unwatch <nome_da_variável>")
            return
        
        var_name = args[0]
        self.variable_inspector.remove_watch(var_name)
        print(f"Variável '{var_name}' removida da observação")
    
    def _cmd_show_stack(self, session: DebugSession, args: List[str]):
        """Mostra stack trace"""
        print("\nStack Trace:")
        print("-" * 60)
        
        for i, frame in enumerate(session.stack_frames):
            marker = ">>>" if frame == session.current_frame else "   "
            print(f"{marker} {i:2d}. {frame.function_name} ({Path(frame.filename).name}:{frame.line_number})")
    
    def _cmd_change_frame(self, session: DebugSession, args: List[str]):
        """Muda para frame específico"""
        if not args:
            print("Uso: frame <número>")
            return
        
        try:
            frame_num = int(args[0])
            if 0 <= frame_num < len(session.stack_frames):
                session.current_frame = session.stack_frames[frame_num]
                self._show_current_context(session.current_frame)
            else:
                print(f"Frame {frame_num} não existe. Disponíveis: 0-{len(session.stack_frames)-1}")
        except ValueError:
            print("Número de frame inválido")
    
    def _cmd_memory_analysis(self, session: DebugSession, args: List[str]):
        """Análise de uso de memória"""
        if not session.current_frame:
            print("Nenhum frame ativo")
            return
        
        # Simular análise com dados disponíveis
        print("\nAnálise de Memória:")
        print("-" * 40)
        print(f"Total de variáveis: {len(session.current_frame.local_variables)}")
        
        # Mostrar variáveis por tipo
        type_counts = defaultdict(int)
        for var_info in session.current_frame.local_variables.values():
            type_counts[var_info['type']] += 1
        
        print("\nVariáveis por tipo:")
        for var_type, count in sorted(type_counts.items()):
            print(f"  {var_type}: {count}")
    
    def _cmd_variable_history(self, session: DebugSession, args: List[str]):
        """Mostra histórico de uma variável"""
        if not args:
            print("Uso: history <nome_da_variável>")
            return
        
        var_name = args[0]
        history = self.variable_inspector.get_variable_history(var_name)
        
        if not history:
            print(f"Nenhum histórico encontrado para '{var_name}'")
            return
        
        print(f"\nHistórico da variável '{var_name}':")
        print("-" * 50)
        
        for i, entry in enumerate(history):
            print(f"{i+1:3d}. {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {entry['value'][:50]}... ({entry['type']})")
    
    def _cmd_execute_code(self, session: DebugSession, args: List[str]):
        """Executa código Python no contexto atual"""
        if not args:
            print("Uso: execute <código_python>")
            return
        
        code_to_execute = ' '.join(args)
        
        try:
            # Simular execução (em implementação real, usaríamos o frame atual)
            print(f"Executando: {code_to_execute}")
            print("(Execução simulada - implementação completa requer acesso ao frame real)")
        except Exception as e:
            print(f"Erro na execução: {e}")
    
    def _cmd_continue(self, session: DebugSession, args: List[str]):
        """Continua execução"""
        print("Continuando execução...")
        session.is_active = False
        return False
    
    def _cmd_quit(self, session: DebugSession, args: List[str]):
        """Sai do debugger"""
        print("Saindo do debugger...")
        session.is_active = False
        return False
    
    def _cmd_list_breakpoints(self, session: DebugSession, args: List[str]):
        """Lista breakpoints ativos"""
        breakpoints = self.logging_system.breakpoints
        
        if not breakpoints:
            print("Nenhum breakpoint ativo")
            return
        
        print("\nBreakpoints ativos:")
        print("-" * 60)
        
        for bp_id, bp in breakpoints.items():
            status = "Ativo" if bp.enabled else "Inativo"
            condition = f" (condição: {bp.condition})" if bp.condition else ""
            print(f"{bp_id}: {Path(bp.file_path).name}:{bp.line_number} - {status}{condition}")
            print(f"    Hits: {bp.hit_count}")
    
    def _cmd_add_breakpoint(self, session: DebugSession, args: List[str]):
        """Adiciona breakpoint"""
        if len(args) < 2:
            print("Uso: add_bp <arquivo> <linha> [condição]")
            return
        
        file_path = args[0]
        try:
            line_number = int(args[1])
            condition = ' '.join(args[2:]) if len(args) > 2 else ""
            
            bp_id = self.logging_system.add_breakpoint(file_path, line_number, condition)
            print(f"Breakpoint adicionado: {bp_id}")
        except ValueError:
            print("Número da linha deve ser um inteiro")
    
    def _cmd_remove_breakpoint(self, session: DebugSession, args: List[str]):
        """Remove breakpoint"""
        if not args:
            print("Uso: remove_bp <id_do_breakpoint>")
            return
        
        bp_id = args[0]
        self.logging_system.remove_breakpoint(bp_id)
        print(f"Breakpoint removido: {bp_id}")
    
    def _cmd_enable_breakpoint(self, session: DebugSession, args: List[str]):
        """Habilita breakpoint"""
        if not args:
            print("Uso: enable_bp <id_do_breakpoint>")
            return
        
        bp_id = args[0]
        if bp_id in self.logging_system.breakpoints:
            self.logging_system.breakpoints[bp_id].enabled = True
            print(f"Breakpoint habilitado: {bp_id}")
        else:
            print(f"Breakpoint não encontrado: {bp_id}")
    
    def _cmd_disable_breakpoint(self, session: DebugSession, args: List[str]):
        """Desabilita breakpoint"""
        if not args:
            print("Uso: disable_bp <id_do_breakpoint>")
            return
        
        bp_id = args[0]
        if bp_id in self.logging_system.breakpoints:
            self.logging_system.breakpoints[bp_id].enabled = False
            print(f"Breakpoint desabilitado: {bp_id}")
        else:
            print(f"Breakpoint não encontrado: {bp_id}")

# Instância global do debugger
_debugger = None

def get_debugger() -> InteractiveDebugger:
    """Obtém instância global do debugger"""
    global _debugger
    if _debugger is None:
        _debugger = InteractiveDebugger()
    return _debugger

def debug_here():
    """Inicia debugging no ponto atual"""
    frame = inspect.currentframe().f_back
    get_debugger().start_interactive_session(frame)

def set_trace():
    """Alias para debug_here() compatível com pdb"""
    debug_here()

if __name__ == "__main__":
    # Exemplo de uso
    def test_function():
        x = 10
        y = [1, 2, 3, 4, 5]
        z = {'a': 1, 'b': 2}
        
        # Iniciar debugging aqui
        debug_here()
        
        result = x * sum(y)
        return result
    
    print("Testando debugger interativo...")
    result = test_function()
    print(f"Resultado: {result}")