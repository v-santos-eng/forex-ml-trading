import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from datetime import datetime, timedelta
import requests
import json
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from newsapi import NewsApiClient
try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    tweepy = None
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pickle
from pathlib import Path
from collections import defaultdict, deque
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Suprimir warnings
warnings.filterwarnings('ignore')

@dataclass
class SentimentConfig:
    """Configuração para análise de sentimento"""
    
    # APIs e credenciais
    news_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    alpha_vantage_key: Optional[str] = None
    
    # Parâmetros de coleta
    news_sources: List[str] = field(default_factory=lambda: [
        'reuters', 'bloomberg', 'cnbc', 'marketwatch', 'yahoo-finance'
    ])
    max_articles_per_source: int = 50
    lookback_days: int = 7
    
    # Parâmetros de processamento
    sentiment_methods: List[str] = field(default_factory=lambda: [
        'textblob', 'vader', 'custom'
    ])
    text_preprocessing: bool = True
    remove_stopwords: bool = True
    
    # Parâmetros de features
    sentiment_window: int = 5  # Janela para médias móveis
    volatility_adjustment: bool = True
    market_hours_only: bool = False
    
    # Parâmetros de ML
    train_sentiment_model: bool = True
    model_retrain_frequency: int = 30  # dias
    feature_importance_threshold: float = 0.01
    
    # Parâmetros de cache
    use_cache: bool = True
    cache_duration_hours: int = 6
    database_path: str = "sentiment_data.db"
    
    # Parâmetros de performance
    enable_parallel: bool = True
    max_workers: int = 4
    request_delay: float = 1.0  # segundos entre requests
    
    # Símbolos para análise
    symbols: List[str] = field(default_factory=lambda: [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD'
    ])
    
    # Palavras-chave por símbolo
    symbol_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        'EURUSD': ['EUR', 'Euro', 'European Central Bank', 'ECB', 'Eurozone'],
        'GBPUSD': ['GBP', 'Pound', 'Sterling', 'Bank of England', 'BOE', 'Brexit'],
        'USDJPY': ['JPY', 'Yen', 'Bank of Japan', 'BOJ', 'Japan'],
        'USDCHF': ['CHF', 'Franc', 'Swiss National Bank', 'SNB', 'Switzerland'],
        'AUDUSD': ['AUD', 'Australian Dollar', 'RBA', 'Reserve Bank Australia']
    })

Base = declarative_base()

class NewsArticle(Base):
    """Modelo de dados para artigos de notícias"""
    __tablename__ = 'news_articles'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    content = Column(Text)
    source = Column(String(100))
    published_at = Column(DateTime)
    url = Column(String(500))
    symbol = Column(String(10))
    sentiment_textblob = Column(Float)
    sentiment_vader = Column(Float)
    sentiment_custom = Column(Float)
    relevance_score = Column(Float)
    created_at = Column(DateTime, default=datetime.now)

class SentimentAnalyzer:
    """
    Sistema avançado de análise de sentimento para trading
    
    Funcionalidades:
    1. Coleta de notícias de múltiplas fontes
    2. Análise de sentimento com múltiplos métodos
    3. Processamento de texto avançado
    4. Features de sentimento para ML
    5. Modelo de sentimento customizado
    6. Cache e persistência de dados
    7. Análise de relevância por símbolo
    8. Agregação temporal de sentimento
    """
    
    def __init__(self, 
                 config: Optional[SentimentConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            config: Configuração do sistema
            logger: Logger para debug
        """
        self.config = config or SentimentConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Inicializar analisadores de sentimento
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Modelo customizado
        self.custom_model = None
        self.model_last_trained = None
        
        # Cache de sentimentos
        self.sentiment_cache = {}
        
        # Inicializar APIs
        self._init_apis()
        
        # Inicializar banco de dados
        self._init_database()
        
        # Carregar modelo se existir
        self._load_custom_model()
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analisa o sentimento de um texto e retorna um score agregado
        
        Args:
            text: Texto para análise de sentimento
            
        Returns:
            float: Score de sentimento entre -1 (muito negativo) e 1 (muito positivo)
        """
        try:
            # Usar o método interno existente
            sentiments = self._analyze_sentiment(text)
            
            # Calcular média ponderada dos diferentes métodos
            scores = []
            weights = []
            
            if 'sentiment_textblob' in sentiments:
                scores.append(sentiments['sentiment_textblob'])
                weights.append(0.3)
                
            if 'sentiment_vader' in sentiments:
                scores.append(sentiments['sentiment_vader'])
                weights.append(0.4)
                
            if 'sentiment_custom' in sentiments and sentiments['sentiment_custom'] != 0.0:
                scores.append(sentiments['sentiment_custom'])
                weights.append(0.3)
            
            if not scores:
                return 0.0
                
            # Média ponderada
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Erro na análise de sentimento: {e}")
            return 0.0
        
    def _init_apis(self) -> None:
        """
        Inicializa APIs externas
        """
        # News API
        if self.config.news_api_key:
            try:
                self.news_api = NewsApiClient(api_key=self.config.news_api_key)
            except Exception as e:
                self.logger.warning(f"Erro ao inicializar News API: {e}")
                self.news_api = None
        else:
            self.news_api = None
            
        # Twitter API (usando tweepy v2)
        if self.config.twitter_bearer_token:
            try:
                self.twitter_api = tweepy.Client(
                    bearer_token=self.config.twitter_bearer_token,
                    wait_on_rate_limit=True
                )
            except Exception as e:
                self.logger.warning(f"Erro ao inicializar Twitter API: {e}")
                self.twitter_api = None
        else:
            self.twitter_api = None
            
    def _init_database(self) -> None:
        """
        Inicializa banco de dados SQLite
        """
        try:
            self.engine = create_engine(f'sqlite:///{self.config.database_path}')
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            self.logger.info("Banco de dados inicializado")
        except Exception as e:
            self.logger.error(f"Erro ao inicializar banco: {e}")
            self.session = None
            
    def collect_news_data(self, 
                         symbols: Optional[List[str]] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Coleta dados de notícias para os símbolos especificados
        
        Args:
            symbols: Lista de símbolos para coletar notícias
            start_date: Data de início
            end_date: Data de fim
            
        Returns:
            DataFrame com notícias coletadas
        """
        if symbols is None:
            symbols = self.config.symbols
            
        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.config.lookback_days)
            
        if end_date is None:
            end_date = datetime.now()
            
        all_articles = []
        
        # Coletar notícias para cada símbolo
        for symbol in symbols:
            self.logger.info(f"Coletando notícias para {symbol}")
            
            # Verificar cache primeiro
            if self.config.use_cache:
                cached_articles = self._get_cached_articles(symbol, start_date, end_date)
                if cached_articles:
                    all_articles.extend(cached_articles)
                    continue
                    
            # Coletar de múltiplas fontes
            symbol_articles = []
            
            if self.config.enable_parallel:
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = []
                    
                    # News API
                    if self.news_api:
                        future = executor.submit(
                            self._collect_from_news_api, 
                            symbol, start_date, end_date
                        )
                        futures.append(future)
                        
                    # Yahoo Finance News
                    future = executor.submit(
                        self._collect_from_yahoo_finance, 
                        symbol, start_date, end_date
                    )
                    futures.append(future)
                    
                    # Coletar resultados
                    for future in as_completed(futures):
                        try:
                            articles = future.result()
                            symbol_articles.extend(articles)
                        except Exception as e:
                            self.logger.error(f"Erro na coleta paralela: {e}")
            else:
                # Coleta sequencial
                if self.news_api:
                    articles = self._collect_from_news_api(symbol, start_date, end_date)
                    symbol_articles.extend(articles)
                    
                articles = self._collect_from_yahoo_finance(symbol, start_date, end_date)
                symbol_articles.extend(articles)
                
            # Processar e analisar sentimento
            processed_articles = self._process_articles(symbol_articles, symbol)
            all_articles.extend(processed_articles)
            
            # Salvar no banco
            if self.session:
                self._save_articles_to_db(processed_articles)
                
            # Delay entre símbolos
            time.sleep(self.config.request_delay)
            
        # Converter para DataFrame
        if all_articles:
            df = pd.DataFrame(all_articles)
            df['published_at'] = pd.to_datetime(df['published_at'])
            return df.sort_values('published_at')
        else:
            return pd.DataFrame()
    
    def _collect_from_news_api(self, symbol: str, 
                              start_date: datetime, 
                              end_date: datetime) -> List[Dict]:
        """
        Coleta notícias da News API
        """
        articles = []
        
        try:
            keywords = self.config.symbol_keywords.get(symbol, [symbol])
            query = ' OR '.join(keywords)
            
            # Buscar artigos
            response = self.news_api.get_everything(
                q=query,
                sources=','.join(self.config.news_sources),
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=self.config.max_articles_per_source
            )
            
            for article in response['articles']:
                if article['content'] and article['title']:
                    articles.append({
                        'title': article['title'],
                        'content': article['content'],
                        'source': article['source']['name'],
                        'published_at': article['publishedAt'],
                        'url': article['url'],
                        'symbol': symbol
                    })
                    
        except Exception as e:
            self.logger.error(f"Erro ao coletar da News API para {symbol}: {e}")
            
        return articles
    
    def _collect_from_yahoo_finance(self, symbol: str,
                                   start_date: datetime,
                                   end_date: datetime) -> List[Dict]:
        """
        Coleta notícias do Yahoo Finance
        """
        articles = []
        
        try:
            # Mapear símbolos forex para tickers do Yahoo
            yahoo_symbol_map = {
                'EURUSD': 'EURUSD=X',
                'GBPUSD': 'GBPUSD=X', 
                'USDJPY': 'USDJPY=X',
                'USDCHF': 'USDCHF=X',
                'AUDUSD': 'AUDUSD=X'
            }
            
            yahoo_symbol = yahoo_symbol_map.get(symbol, symbol)
            ticker = yf.Ticker(yahoo_symbol)
            
            # Obter notícias
            news = ticker.news
            
            for item in news[:self.config.max_articles_per_source]:
                # Verificar se está no período
                pub_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                
                if start_date <= pub_date <= end_date:
                    articles.append({
                        'title': item.get('title', ''),
                        'content': item.get('summary', ''),
                        'source': 'Yahoo Finance',
                        'published_at': pub_date,
                        'url': item.get('link', ''),
                        'symbol': symbol
                    })
                    
        except Exception as e:
            self.logger.error(f"Erro ao coletar do Yahoo Finance para {symbol}: {e}")
            
        return articles
    
    def _process_articles(self, articles: List[Dict], symbol: str) -> List[Dict]:
        """
        Processa artigos e calcula sentimento
        """
        processed = []
        
        for article in articles:
            try:
                # Preprocessar texto
                text = self._preprocess_text(article['title'] + ' ' + article['content'])
                
                # Calcular relevância
                relevance = self._calculate_relevance(text, symbol)
                
                # Calcular sentimentos
                sentiments = self._analyze_sentiment(text)
                
                # Adicionar ao artigo processado
                processed_article = article.copy()
                processed_article.update({
                    'processed_text': text,
                    'relevance_score': relevance,
                    **sentiments
                })
                
                processed.append(processed_article)
                
            except Exception as e:
                self.logger.error(f"Erro ao processar artigo: {e}")
                continue
                
        return processed
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocessa texto para análise
        """
        if not self.config.text_preprocessing:
            return text
            
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remover stopwords (implementação simples)
        if self.config.remove_stopwords:
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = text.split()
            text = ' '.join([word for word in words if word not in stopwords])
            
        return text
    
    def _calculate_relevance(self, text: str, symbol: str) -> float:
        """
        Calcula relevância do texto para o símbolo
        """
        keywords = self.config.symbol_keywords.get(symbol, [symbol])
        
        relevance_score = 0.0
        text_lower = text.lower()
        
        for keyword in keywords:
            # Contar ocorrências da palavra-chave
            count = text_lower.count(keyword.lower())
            
            # Peso baseado na importância da palavra-chave
            if keyword == symbol:
                weight = 2.0  # Símbolo principal tem peso maior
            else:
                weight = 1.0
                
            relevance_score += count * weight
            
        # Normalizar pela length do texto
        text_length = len(text.split())
        if text_length > 0:
            relevance_score = relevance_score / text_length
            
        return min(relevance_score, 1.0)  # Cap em 1.0
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analisa sentimento usando múltiplos métodos
        """
        sentiments = {}
        
        # TextBlob
        if 'textblob' in self.config.sentiment_methods:
            try:
                blob = TextBlob(text)
                sentiments['sentiment_textblob'] = blob.sentiment.polarity
            except Exception as e:
                self.logger.error(f"Erro no TextBlob: {e}")
                sentiments['sentiment_textblob'] = 0.0
                
        # VADER
        if 'vader' in self.config.sentiment_methods:
            try:
                scores = self.vader_analyzer.polarity_scores(text)
                sentiments['sentiment_vader'] = scores['compound']
            except Exception as e:
                self.logger.error(f"Erro no VADER: {e}")
                sentiments['sentiment_vader'] = 0.0
                
        # Modelo customizado
        if 'custom' in self.config.sentiment_methods and self.custom_model:
            try:
                sentiment_custom = self._predict_custom_sentiment(text)
                sentiments['sentiment_custom'] = sentiment_custom
            except Exception as e:
                self.logger.error(f"Erro no modelo customizado: {e}")
                sentiments['sentiment_custom'] = 0.0
        else:
            sentiments['sentiment_custom'] = 0.0
            
        return sentiments
    
    def _predict_custom_sentiment(self, text: str) -> float:
        """
        Prediz sentimento usando modelo customizado
        """
        if self.custom_model is None:
            return 0.0
            
        # Vetorizar texto
        text_vector = self.tfidf_vectorizer.transform([text])
        
        # Predizer
        prediction = self.custom_model.predict_proba(text_vector)[0]
        
        # Converter para escala -1 a 1
        # Assumindo classes [negativo, neutro, positivo]
        if len(prediction) == 3:
            sentiment_score = prediction[2] - prediction[0]  # positivo - negativo
        else:
            sentiment_score = prediction[1] - prediction[0]  # positivo - negativo
            
        return np.clip(sentiment_score, -1.0, 1.0)
    
    def train_custom_model(self, training_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Treina modelo customizado de sentimento
        
        Args:
            training_data: Dados de treino com colunas 'text' e 'sentiment'
            
        Returns:
            Métricas do modelo treinado
        """
        if training_data is None:
            # Usar dados do banco como treino
            training_data = self._prepare_training_data()
            
        if len(training_data) < 100:
            self.logger.warning("Dados insuficientes para treinar modelo customizado")
            return {'status': 'insufficient_data'}
            
        try:
            # Preparar features
            X = training_data['text'].values
            y = training_data['sentiment_label'].values
            
            # Vetorizar texto
            X_tfidf = self.tfidf_vectorizer.fit_transform(X)
            
            # Split treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Treinar modelo
            self.custom_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.custom_model.fit(X_train, y_train)
            
            # Avaliar modelo
            y_pred = self.custom_model.predict(X_test)
            
            # Métricas
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Salvar modelo
            self._save_custom_model()
            self.model_last_trained = datetime.now()
            
            self.logger.info("Modelo customizado treinado com sucesso")
            
            return {
                'status': 'success',
                'accuracy': report['accuracy'],
                'classification_report': report,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao treinar modelo customizado: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _prepare_training_data(self) -> pd.DataFrame:
        """
        Prepara dados de treino a partir do banco
        """
        if not self.session:
            return pd.DataFrame()
            
        try:
            # Buscar artigos com sentimento
            query = """
            SELECT processed_text, sentiment_textblob, sentiment_vader
            FROM news_articles 
            WHERE processed_text IS NOT NULL 
            AND sentiment_textblob IS NOT NULL
            AND sentiment_vader IS NOT NULL
            """
            
            df = pd.read_sql(query, self.engine)
            
            if len(df) == 0:
                return pd.DataFrame()
                
            # Criar labels baseados no consenso dos métodos
            df['avg_sentiment'] = (df['sentiment_textblob'] + df['sentiment_vader']) / 2
            
            # Discretizar sentimento
            def sentiment_to_label(score):
                if score < -0.1:
                    return 'negative'
                elif score > 0.1:
                    return 'positive'
                else:
                    return 'neutral'
                    
            df['sentiment_label'] = df['avg_sentiment'].apply(sentiment_to_label)
            
            # Renomear coluna
            df = df.rename(columns={'processed_text': 'text'})
            
            return df[['text', 'sentiment_label']]
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar dados de treino: {e}")
            return pd.DataFrame()
    
    def generate_sentiment_features(self, 
                                  symbols: Optional[List[str]] = None,
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Gera features de sentimento para uso em modelos de ML
        
        Args:
            symbols: Lista de símbolos
            start_date: Data de início
            end_date: Data de fim
            
        Returns:
            DataFrame com features de sentimento
        """
        if symbols is None:
            symbols = self.config.symbols
            
        # Coletar dados de sentimento
        sentiment_data = self.collect_news_data(symbols, start_date, end_date)
        
        if len(sentiment_data) == 0:
            self.logger.warning("Nenhum dado de sentimento coletado")
            return pd.DataFrame()
            
        # Gerar features por símbolo
        all_features = []
        
        for symbol in symbols:
            symbol_data = sentiment_data[sentiment_data['symbol'] == symbol].copy()
            
            if len(symbol_data) == 0:
                continue
                
            # Ordenar por data
            symbol_data = symbol_data.sort_values('published_at')
            
            # Agregar por dia
            daily_features = self._aggregate_daily_sentiment(symbol_data, symbol)
            
            all_features.append(daily_features)
            
        if all_features:
            # Combinar features de todos os símbolos
            combined_features = pd.concat(all_features, axis=1)
            
            # Preencher valores faltantes
            combined_features = combined_features.fillna(method='ffill').fillna(0)
            
            return combined_features
        else:
            return pd.DataFrame()
    
    def _aggregate_daily_sentiment(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Agrega sentimento por dia para um símbolo
        """
        # Criar índice diário
        data['date'] = data['published_at'].dt.date
        
        # Agregar métricas por dia
        daily_agg = data.groupby('date').agg({
            'sentiment_textblob': ['mean', 'std', 'count'],
            'sentiment_vader': ['mean', 'std', 'count'],
            'sentiment_custom': ['mean', 'std', 'count'],
            'relevance_score': ['mean', 'sum']
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in daily_agg.columns]
        daily_agg = daily_agg.rename(columns={'date_': 'date'})
        
        # Adicionar prefixo do símbolo
        feature_cols = [col for col in daily_agg.columns if col != 'date']
        for col in feature_cols:
            daily_agg = daily_agg.rename(columns={col: f'{symbol}_{col}'})
            
        # Converter date para datetime
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])
        daily_agg = daily_agg.set_index('date')
        
        # Adicionar features de janela móvel
        window = self.config.sentiment_window
        
        for method in ['textblob', 'vader', 'custom']:
            mean_col = f'{symbol}_sentiment_{method}_mean'
            if mean_col in daily_agg.columns:
                # Média móvel
                daily_agg[f'{symbol}_sentiment_{method}_ma{window}'] = (
                    daily_agg[mean_col].rolling(window=window, min_periods=1).mean()
                )
                
                # Volatilidade do sentimento
                daily_agg[f'{symbol}_sentiment_{method}_vol{window}'] = (
                    daily_agg[mean_col].rolling(window=window, min_periods=1).std()
                )
                
                # Momentum do sentimento
                daily_agg[f'{symbol}_sentiment_{method}_momentum'] = (
                    daily_agg[mean_col] - daily_agg[mean_col].shift(1)
                )
                
        # Features de relevância
        relevance_mean_col = f'{symbol}_relevance_score_mean'
        if relevance_mean_col in daily_agg.columns:
            daily_agg[f'{symbol}_relevance_ma{window}'] = (
                daily_agg[relevance_mean_col].rolling(window=window, min_periods=1).mean()
            )
            
        # Features de volume de notícias
        count_col = f'{symbol}_sentiment_textblob_count'
        if count_col in daily_agg.columns:
            daily_agg[f'{symbol}_news_volume'] = daily_agg[count_col]
            daily_agg[f'{symbol}_news_volume_ma{window}'] = (
                daily_agg[count_col].rolling(window=window, min_periods=1).mean()
            )
            
        return daily_agg
    
    def _get_cached_articles(self, symbol: str, 
                           start_date: datetime, 
                           end_date: datetime) -> List[Dict]:
        """
        Recupera artigos do cache/banco
        """
        if not self.session:
            return []
            
        try:
            articles = self.session.query(NewsArticle).filter(
                NewsArticle.symbol == symbol,
                NewsArticle.published_at >= start_date,
                NewsArticle.published_at <= end_date,
                NewsArticle.created_at >= datetime.now() - timedelta(hours=self.config.cache_duration_hours)
            ).all()
            
            return [{
                'title': article.title,
                'content': article.content,
                'source': article.source,
                'published_at': article.published_at,
                'url': article.url,
                'symbol': article.symbol,
                'sentiment_textblob': article.sentiment_textblob,
                'sentiment_vader': article.sentiment_vader,
                'sentiment_custom': article.sentiment_custom,
                'relevance_score': article.relevance_score
            } for article in articles]
            
        except Exception as e:
            self.logger.error(f"Erro ao recuperar cache: {e}")
            return []
    
    def _save_articles_to_db(self, articles: List[Dict]) -> None:
        """
        Salva artigos no banco de dados
        """
        if not self.session:
            return
            
        try:
            for article_data in articles:
                article = NewsArticle(
                    title=article_data.get('title'),
                    content=article_data.get('content'),
                    source=article_data.get('source'),
                    published_at=article_data.get('published_at'),
                    url=article_data.get('url'),
                    symbol=article_data.get('symbol'),
                    sentiment_textblob=article_data.get('sentiment_textblob'),
                    sentiment_vader=article_data.get('sentiment_vader'),
                    sentiment_custom=article_data.get('sentiment_custom'),
                    relevance_score=article_data.get('relevance_score')
                )
                
                self.session.add(article)
                
            self.session.commit()
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar no banco: {e}")
            self.session.rollback()
    
    def _save_custom_model(self) -> None:
        """
        Salva modelo customizado
        """
        try:
            model_data = {
                'model': self.custom_model,
                'vectorizer': self.tfidf_vectorizer,
                'trained_at': datetime.now()
            }
            
            with open('custom_sentiment_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
                
            self.logger.info("Modelo customizado salvo")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {e}")
    
    def _load_custom_model(self) -> None:
        """
        Carrega modelo customizado
        """
        try:
            if Path('custom_sentiment_model.pkl').exists():
                with open('custom_sentiment_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                    
                self.custom_model = model_data['model']
                self.tfidf_vectorizer = model_data['vectorizer']
                self.model_last_trained = model_data['trained_at']
                
                self.logger.info("Modelo customizado carregado")
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
    
    def plot_sentiment_analysis(self, 
                              sentiment_data: pd.DataFrame,
                              symbol: str,
                              save_plot: bool = False) -> None:
        """
        Plota análise de sentimento
        """
        symbol_data = sentiment_data[sentiment_data['symbol'] == symbol].copy()
        
        if len(symbol_data) == 0:
            self.logger.warning(f"Nenhum dado de sentimento para {symbol}")
            return
            
        # Criar plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Sentiment Analysis - {symbol}', fontsize=16)
        
        # 1. Sentimento ao longo do tempo
        ax1 = axes[0, 0]
        symbol_data = symbol_data.sort_values('published_at')
        
        ax1.plot(symbol_data['published_at'], symbol_data['sentiment_textblob'], 
                label='TextBlob', alpha=0.7)
        ax1.plot(symbol_data['published_at'], symbol_data['sentiment_vader'], 
                label='VADER', alpha=0.7)
        
        if 'sentiment_custom' in symbol_data.columns:
            ax1.plot(symbol_data['published_at'], symbol_data['sentiment_custom'], 
                    label='Custom', alpha=0.7)
            
        ax1.set_title('Sentiment Over Time')
        ax1.set_ylabel('Sentiment Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 2. Distribuição de sentimento
        ax2 = axes[0, 1]
        ax2.hist(symbol_data['sentiment_textblob'], bins=20, alpha=0.7, label='TextBlob')
        ax2.hist(symbol_data['sentiment_vader'], bins=20, alpha=0.7, label='VADER')
        ax2.set_title('Sentiment Distribution')
        ax2.set_xlabel('Sentiment Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. Relevância vs Sentimento
        ax3 = axes[1, 0]
        scatter = ax3.scatter(symbol_data['relevance_score'], 
                            symbol_data['sentiment_textblob'],
                            c=symbol_data['sentiment_vader'], 
                            cmap='RdYlBu', alpha=0.6)
        ax3.set_title('Relevance vs Sentiment')
        ax3.set_xlabel('Relevance Score')
        ax3.set_ylabel('TextBlob Sentiment')
        plt.colorbar(scatter, ax=ax3, label='VADER Sentiment')
        
        # 4. Volume de notícias por dia
        ax4 = axes[1, 1]
        daily_counts = symbol_data.groupby(symbol_data['published_at'].dt.date).size()
        ax4.bar(daily_counts.index, daily_counts.values)
        ax4.set_title('News Volume by Day')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Number of Articles')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{symbol}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def get_sentiment_summary(self, 
                            symbols: Optional[List[str]] = None,
                            days: int = 7) -> Dict[str, Any]:
        """
        Retorna resumo de sentimento
        """
        if symbols is None:
            symbols = self.config.symbols
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        sentiment_data = self.collect_news_data(symbols, start_date, end_date)
        
        if len(sentiment_data) == 0:
            return {'status': 'no_data'}
            
        summary = {}
        
        for symbol in symbols:
            symbol_data = sentiment_data[sentiment_data['symbol'] == symbol]
            
            if len(symbol_data) == 0:
                summary[symbol] = {'status': 'no_data'}
                continue
                
            # Calcular métricas
            summary[symbol] = {
                'articles_count': len(symbol_data),
                'avg_sentiment_textblob': symbol_data['sentiment_textblob'].mean(),
                'avg_sentiment_vader': symbol_data['sentiment_vader'].mean(),
                'avg_relevance': symbol_data['relevance_score'].mean(),
                'sentiment_std_textblob': symbol_data['sentiment_textblob'].std(),
                'sentiment_std_vader': symbol_data['sentiment_vader'].std(),
                'latest_sentiment': {
                    'textblob': symbol_data.iloc[-1]['sentiment_textblob'],
                    'vader': symbol_data.iloc[-1]['sentiment_vader'],
                    'published_at': symbol_data.iloc[-1]['published_at']
                },
                'sentiment_trend': self._calculate_sentiment_trend(symbol_data)
            }
            
        return summary
    
    def _calculate_sentiment_trend(self, data: pd.DataFrame) -> str:
        """
        Calcula tendência do sentimento
        """
        if len(data) < 2:
            return 'insufficient_data'
            
        # Usar média dos métodos
        data = data.sort_values('published_at')
        data['avg_sentiment'] = (data['sentiment_textblob'] + data['sentiment_vader']) / 2
        
        # Comparar primeira e segunda metade
        mid_point = len(data) // 2
        first_half = data.iloc[:mid_point]['avg_sentiment'].mean()
        second_half = data.iloc[mid_point:]['avg_sentiment'].mean()
        
        diff = second_half - first_half
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'deteriorating'
        else:
            return 'stable'

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Configuração (adicione suas chaves de API)
    config = SentimentConfig(
        # news_api_key="sua_chave_news_api",
        # twitter_bearer_token="seu_token_twitter",
        symbols=['EURUSD', 'GBPUSD'],
        lookback_days=3,
        max_articles_per_source=20
    )
    
    # Criar analisador
    analyzer = SentimentAnalyzer(config=config, logger=logger)
    
    print("\n=== COLETANDO DADOS DE SENTIMENTO ===")
    
    # Coletar dados de sentimento
    sentiment_data = analyzer.collect_news_data()
    
    if len(sentiment_data) > 0:
        print(f"Coletados {len(sentiment_data)} artigos")
        print(f"Símbolos: {sentiment_data['symbol'].unique()}")
        print(f"Fontes: {sentiment_data['source'].unique()}")
        
        # Mostrar estatísticas por símbolo
        for symbol in sentiment_data['symbol'].unique():
            symbol_data = sentiment_data[sentiment_data['symbol'] == symbol]
            print(f"\n{symbol}:")
            print(f"  Artigos: {len(symbol_data)}")
            print(f"  Sentimento TextBlob médio: {symbol_data['sentiment_textblob'].mean():.3f}")
            print(f"  Sentimento VADER médio: {symbol_data['sentiment_vader'].mean():.3f}")
            print(f"  Relevância média: {symbol_data['relevance_score'].mean():.3f}")
            
        # Treinar modelo customizado
        print("\n=== TREINANDO MODELO CUSTOMIZADO ===")
        training_result = analyzer.train_custom_model()
        print(f"Resultado do treino: {training_result}")
        
        # Gerar features de sentimento
        print("\n=== GERANDO FEATURES DE SENTIMENTO ===")
        features = analyzer.generate_sentiment_features()
        
        if len(features) > 0:
            print(f"Features geradas: {features.shape}")
            print(f"Colunas: {list(features.columns)}")
            print("\nPrimeiras linhas:")
            print(features.head())
            
        # Resumo de sentimento
        print("\n=== RESUMO DE SENTIMENTO ===")
        summary = analyzer.get_sentiment_summary()
        
        for symbol, data in summary.items():
            if isinstance(data, dict) and 'status' not in data:
                print(f"\n{symbol}:")
                print(f"  Artigos: {data['articles_count']}")
                print(f"  Sentimento médio (TextBlob): {data['avg_sentiment_textblob']:.3f}")
                print(f"  Sentimento médio (VADER): {data['avg_sentiment_vader']:.3f}")
                print(f"  Tendência: {data['sentiment_trend']}")
                
        # Plotar análise para primeiro símbolo
        if len(sentiment_data) > 0:
            first_symbol = sentiment_data['symbol'].iloc[0]
            print(f"\n=== PLOTANDO ANÁLISE PARA {first_symbol} ===")
            analyzer.plot_sentiment_analysis(sentiment_data, first_symbol, save_plot=False)
            
    else:
        print("Nenhum dado de sentimento coletado")
        print("Verifique as configurações de API e conectividade")