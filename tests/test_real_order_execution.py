#!/usr/bin/env python3
"""
Teste de Execução Real de Ordem MT5
Testa execução real de ordem com volume mínimo para identificar problemas
"""

import MetaTrader5 as mt5
import time
import logging
from datetime import datetime
from typing import Dict, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealOrderTest:
    """Teste de execução real de ordens"""
    
    def __init__(self):
        self.magic_number = 234000
        self.test_results = []
    
    def connect_mt5(self) -> bool:
        """Conecta ao MT5"""
        try:
            if not mt5.initialize():
                logger.error(f"Falha na inicialização MT5: {mt5.last_error()}")
                return False
            
            logger.info("✅ Conectado ao MT5")
            return True
            
        except Exception as e:
            logger.error(f"Erro na conexão MT5: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Obtém informações da conta"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return {}
            
            info = {
                "login": account_info.login,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "free_margin": account_info.margin_free,
                "currency": account_info.currency
            }
            
            logger.info(f"Conta: {info['login']} | Saldo: {info['balance']} {info['currency']}")
            logger.info(f"Margem livre: {info['free_margin']} | Margem usada: {info['margin']}")
            
            return info
            
        except Exception as e:
            logger.error(f"Erro ao obter informações da conta: {e}")
            return {}
    
    def test_micro_order(self, symbol: str, volume: float = 0.01) -> Dict:
        """Testa ordem com volume mínimo"""
        logger.info(f"\n🧪 Testando ordem real: {symbol} - Volume: {volume}")
        
        try:
            # Verificar símbolo
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {
                    "success": False,
                    "error": "Símbolo não encontrado"
                }
            
            # Obter tick atual
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {
                    "success": False,
                    "error": "Não foi possível obter tick"
                }
            
            logger.info(f"Preços atuais: Bid={tick.bid}, Ask={tick.ask}")
            logger.info(f"Volume mín: {symbol_info.volume_min}, máx: {symbol_info.volume_max}")
            
            # Preparar requisição de compra
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": "Teste real ML Bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            logger.info(f"Enviando ordem: {request}")
            
            # Executar ordem
            result = mt5.order_send(request)
            
            if result is None:
                return {
                    "success": False,
                    "error": "order_send retornou None",
                    "last_error": mt5.last_error()
                }
            
            # Analisar resultado
            success = result.retcode == mt5.TRADE_RETCODE_DONE
            
            test_result = {
                "success": success,
                "retcode": result.retcode,
                "retcode_name": self.get_retcode_name(result.retcode),
                "comment": result.comment,
                "order_id": getattr(result, 'order', None),
                "deal_id": getattr(result, 'deal', None),
                "volume": getattr(result, 'volume', None),
                "price": getattr(result, 'price', None),
                "request_id": getattr(result, 'request_id', None),
                "request": request
            }
            
            if success:
                logger.info(f"✅ Ordem executada com sucesso!")
                logger.info(f"Order ID: {test_result['order_id']}, Deal ID: {test_result['deal_id']}")
                logger.info(f"Volume: {test_result['volume']}, Preço: {test_result['price']}")
                
                # Aguardar um pouco e fechar a posição
                time.sleep(2)
                close_result = self.close_position(symbol)
                test_result["close_result"] = close_result
                
            else:
                logger.error(f"❌ Falha na execução: {test_result['retcode_name']} - {test_result['comment']}")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Erro ao testar ordem: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def close_position(self, symbol: str) -> Dict:
        """Fecha posição aberta"""
        try:
            logger.info(f"🔄 Fechando posição: {symbol}")
            
            # Obter posições abertas
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return {
                    "success": False,
                    "error": "Nenhuma posição encontrada"
                }
            
            position = positions[0]  # Primeira posição
            
            # Obter tick atual
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {
                    "success": False,
                    "error": "Não foi possível obter tick para fechamento"
                }
            
            # Determinar tipo de ordem para fechamento
            if position.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            
            # Preparar requisição de fechamento
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": price,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": "Fechamento teste",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Executar fechamento
            result = mt5.order_send(close_request)
            
            if result is None:
                return {
                    "success": False,
                    "error": "Falha no fechamento - resultado None"
                }
            
            success = result.retcode == mt5.TRADE_RETCODE_DONE
            
            if success:
                logger.info(f"✅ Posição fechada com sucesso!")
                logger.info(f"Deal ID: {getattr(result, 'deal', None)}")
            else:
                logger.error(f"❌ Falha no fechamento: {result.retcode} - {result.comment}")
            
            return {
                "success": success,
                "retcode": result.retcode,
                "retcode_name": self.get_retcode_name(result.retcode),
                "comment": result.comment,
                "deal_id": getattr(result, 'deal', None)
            }
            
        except Exception as e:
            logger.error(f"Erro ao fechar posição: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_retcode_name(self, retcode: int) -> str:
        """Converte código de retorno em nome"""
        retcodes = {
            mt5.TRADE_RETCODE_REQUOTE: "REQUOTE",
            mt5.TRADE_RETCODE_REJECT: "REJECT",
            mt5.TRADE_RETCODE_CANCEL: "CANCEL",
            mt5.TRADE_RETCODE_PLACED: "PLACED",
            mt5.TRADE_RETCODE_DONE: "DONE",
            mt5.TRADE_RETCODE_DONE_PARTIAL: "DONE_PARTIAL",
            mt5.TRADE_RETCODE_ERROR: "ERROR",
            mt5.TRADE_RETCODE_TIMEOUT: "TIMEOUT",
            mt5.TRADE_RETCODE_INVALID: "INVALID",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "INVALID_VOLUME",
            mt5.TRADE_RETCODE_INVALID_PRICE: "INVALID_PRICE",
            mt5.TRADE_RETCODE_INVALID_STOPS: "INVALID_STOPS",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "TRADE_DISABLED",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "MARKET_CLOSED",
            mt5.TRADE_RETCODE_NO_MONEY: "NO_MONEY",
            mt5.TRADE_RETCODE_PRICE_CHANGED: "PRICE_CHANGED",
            mt5.TRADE_RETCODE_PRICE_OFF: "PRICE_OFF",
            mt5.TRADE_RETCODE_INVALID_EXPIRATION: "INVALID_EXPIRATION",
            mt5.TRADE_RETCODE_ORDER_CHANGED: "ORDER_CHANGED",
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "TOO_MANY_REQUESTS",
            mt5.TRADE_RETCODE_NO_CHANGES: "NO_CHANGES",
            mt5.TRADE_RETCODE_SERVER_DISABLES_AT: "SERVER_DISABLES_AT",
            mt5.TRADE_RETCODE_CLIENT_DISABLES_AT: "CLIENT_DISABLES_AT",
            mt5.TRADE_RETCODE_LOCKED: "LOCKED",
            mt5.TRADE_RETCODE_FROZEN: "FROZEN",
            mt5.TRADE_RETCODE_INVALID_FILL: "INVALID_FILL",
            mt5.TRADE_RETCODE_CONNECTION: "CONNECTION",
            mt5.TRADE_RETCODE_ONLY_REAL: "ONLY_REAL",
            mt5.TRADE_RETCODE_LIMIT_ORDERS: "LIMIT_ORDERS",
            mt5.TRADE_RETCODE_LIMIT_VOLUME: "LIMIT_VOLUME",
            mt5.TRADE_RETCODE_INVALID_ORDER: "INVALID_ORDER",
            mt5.TRADE_RETCODE_POSITION_CLOSED: "POSITION_CLOSED"
        }
        return retcodes.get(retcode, f"UNKNOWN_{retcode}")
    
    def run_test(self, symbol: str = "GOLD") -> Dict:
        """Executa teste completo"""
        logger.info("🚀 Iniciando teste de execução real...")
        
        # Conectar
        if not self.connect_mt5():
            return {"success": False, "error": "Falha na conexão"}
        
        # Informações da conta
        account_info = self.get_account_info()
        
        # Testar ordem
        test_result = self.test_micro_order(symbol)
        
        # Desconectar
        mt5.shutdown()
        
        # Resultado final
        final_result = {
            "account_info": account_info,
            "test_result": test_result,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("\n" + "="*50)
        logger.info("📋 RESULTADO DO TESTE")
        logger.info("="*50)
        
        if test_result.get("success"):
            logger.info("✅ TESTE BEM-SUCEDIDO - Sistema MT5 funcionando corretamente")
            logger.info("✅ Ordem executada e fechada com sucesso")
        else:
            logger.error("❌ TESTE FALHADO - Problema identificado:")
            logger.error(f"   Erro: {test_result.get('error', 'Desconhecido')}")
            logger.error(f"   Código: {test_result.get('retcode_name', 'N/A')}")
            logger.error(f"   Comentário: {test_result.get('comment', 'N/A')}")
        
        return final_result

def main():
    """Função principal"""
    tester = RealOrderTest()
    result = tester.run_test()
    
    # Salvar resultado
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"real_order_test_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"📄 Resultado salvo: {filename}")
    except Exception as e:
        logger.error(f"Erro ao salvar resultado: {e}")

if __name__ == "__main__":
    main()