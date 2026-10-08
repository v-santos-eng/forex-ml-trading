import logging
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from pathlib import Path
import pickle
import warnings
import copy
from collections import defaultdict
import joblib

# Bibliotecas de otimização
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna não disponível. Usando otimização básica.")

try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

class OptimizationMethod(Enum):
    """Métodos de otimização disponíveis"""
    OPTUNA_TPE = "optuna_tpe"
    OPTUNA_CMAES = "optuna_cmaes"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    HYBRID_RS_BO = "hybrid_rs_bo"  # Random Search seguido de Bayesian (Optuna TPE)

class ModelType(Enum):
    """Tipos de modelos suportados"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"

class OptimizationObjective(Enum):
    """Objetivos de otimização"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    CUSTOM = "custom"

@dataclass
class OptimizationResult:
    """Resultado de uma otimização"""
    optimization_id: str
    pair: str
    model_type: ModelType
    method: OptimizationMethod
    objective: OptimizationObjective
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    optimization_time: float
    timestamp: datetime
    cv_scores: List[float]
    feature_importance: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None
    study_path: Optional[str] = None

@dataclass
class OptimizationConfig:
    """Configuração para otimização"""
    model_type: ModelType
    method: OptimizationMethod = OptimizationMethod.OPTUNA_TPE
    objective: OptimizationObjective = OptimizationObjective.ACCURACY
    n_trials: int = 100
    timeout_seconds: Optional[int] = 3600  # 1 hora
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    early_stopping: bool = True
    pruning: bool = True
    custom_param_space: Optional[Dict] = None
    custom_objective_func: Optional[Callable] = None

class HyperparameterOptimizer:
    """Sistema de otimização automática de hiperparâmetros"""
    
    def __init__(self, db_path: str = "hyperparameter_optimization.db"):
        self.logger = self._setup_logger()
        self.db_path = db_path
        self.optimization_results: List[OptimizationResult] = []
        
        # Configurações padrão
        self.default_configs = self._initialize_default_configs()
        
        # Espaços de parâmetros padrão
        self.param_spaces = self._initialize_param_spaces()
        
        # Cache de estudos Optuna
        self.studies_cache: Dict[str, Any] = {}
        
        # Estado de otimizações em andamento
        self.active_optimizations: Dict[str, threading.Thread] = {}
        
        self._initialize_database()
        self._load_historical_results()
    
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger("HyperparameterOptimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            handler = logging.FileHandler("logs/hyperparameter_optimizer.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def _initialize_default_configs(self) -> Dict[ModelType, OptimizationConfig]:
        """Inicializa configurações padrão para cada tipo de modelo"""
        return {
            ModelType.RANDOM_FOREST: OptimizationConfig(
                model_type=ModelType.RANDOM_FOREST,
                method=OptimizationMethod.OPTUNA_TPE,
                n_trials=50,
                timeout_seconds=1800
            ),
            ModelType.GRADIENT_BOOSTING: OptimizationConfig(
                model_type=ModelType.GRADIENT_BOOSTING,
                method=OptimizationMethod.OPTUNA_TPE,
                n_trials=75,
                timeout_seconds=2400
            ),
            ModelType.XGBOOST: OptimizationConfig(
                model_type=ModelType.XGBOOST,
                method=OptimizationMethod.OPTUNA_TPE,
                n_trials=100,
                timeout_seconds=3600
            ),
            ModelType.LIGHTGBM: OptimizationConfig(
                model_type=ModelType.LIGHTGBM,
                method=OptimizationMethod.OPTUNA_TPE,
                n_trials=100,
                timeout_seconds=3600
            ),
            ModelType.LOGISTIC_REGRESSION: OptimizationConfig(
                model_type=ModelType.LOGISTIC_REGRESSION,
                method=OptimizationMethod.GRID_SEARCH,
                n_trials=25,
                timeout_seconds=600
            ),
            ModelType.SVM: OptimizationConfig(
                model_type=ModelType.SVM,
                method=OptimizationMethod.OPTUNA_TPE,
                n_trials=50,
                timeout_seconds=1800
            )
        }
    
    def _initialize_param_spaces(self) -> Dict[ModelType, Dict]:
        """Define espaços de parâmetros para cada modelo"""
        return {
            ModelType.RANDOM_FOREST: {
                'n_estimators': (50, 500),
                'max_depth': (3, 20),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'criterion': ['gini', 'entropy']
            },
            ModelType.GRADIENT_BOOSTING: {
                'n_estimators': (50, 300),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'subsample': (0.6, 1.0),
                'max_features': ['sqrt', 'log2', None]
            },
            ModelType.XGBOOST: {
                'n_estimators': (50, 500),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 12),
                'min_child_weight': (1, 10),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'gamma': (0, 5),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10)
            },
            ModelType.LIGHTGBM: {
                'n_estimators': (50, 500),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 12),
                'num_leaves': (10, 300),
                'min_child_samples': (5, 100),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10)
            },
            ModelType.LOGISTIC_REGRESSION: {
                'C': (0.001, 100),
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'max_iter': (100, 1000)
            },
            ModelType.SVM: {
                'C': (0.1, 100),
                'gamma': ['scale', 'auto'] + [(0.001, 10)],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'degree': (2, 5),  # Para kernel poly
                'probability': [True]
            }
        }
    
    def _initialize_database(self) -> None:
        """Inicializa o banco de dados SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Tabela de resultados de otimização
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS optimization_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        optimization_id TEXT UNIQUE NOT NULL,
                        pair TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        method TEXT NOT NULL,
                        objective TEXT NOT NULL,
                        best_params TEXT NOT NULL,
                        best_score REAL NOT NULL,
                        n_trials INTEGER NOT NULL,
                        optimization_time REAL NOT NULL,
                        timestamp DATETIME NOT NULL,
                        cv_scores TEXT,
                        feature_importance TEXT,
                        model_path TEXT,
                        study_path TEXT
                    )
                """)
                
                # Tabela de configurações de otimização
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS optimization_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        config_data TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    )
                """)
                
                # Índices para performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_pair_model ON optimization_results(pair, model_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_timestamp ON optimization_results(timestamp)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar banco de dados: {e}")
    
    def _load_historical_results(self) -> None:
        """Carrega resultados históricos do banco"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT optimization_id, pair, model_type, method, objective,
                           best_params, best_score, n_trials, optimization_time,
                           timestamp, cv_scores, feature_importance, model_path, study_path
                    FROM optimization_results
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """)
                
                for row in cursor.fetchall():
                    result = OptimizationResult(
                        optimization_id=row[0],
                        pair=row[1],
                        model_type=ModelType(row[2]),
                        method=OptimizationMethod(row[3]),
                        objective=OptimizationObjective(row[4]),
                        best_params=json.loads(row[5]),
                        best_score=row[6],
                        n_trials=row[7],
                        optimization_time=row[8],
                        timestamp=datetime.fromisoformat(row[9]),
                        cv_scores=json.loads(row[10]) if row[10] else [],
                        feature_importance=json.loads(row[11]) if row[11] else None,
                        model_path=row[12],
                        study_path=row[13]
                    )
                    self.optimization_results.append(result)
                
                self.logger.info(f"Carregados {len(self.optimization_results)} resultados históricos")
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar resultados históricos: {e}")
    
    def optimize_model(self, pair: str, X: np.ndarray, y: np.ndarray, 
                      model_type: ModelType, config: Optional[OptimizationConfig] = None,
                      async_optimization: bool = False) -> Union[OptimizationResult, str]:
        """Otimiza hiperparâmetros de um modelo"""
        if config is None:
            config = self.default_configs.get(model_type)
            if config is None:
                raise ValueError(f"Configuração não encontrada para {model_type}")
        
        optimization_id = f"{pair}_{model_type.value}_{int(datetime.now().timestamp())}"
        
        if async_optimization:
            # Execução assíncrona
            thread = threading.Thread(
                target=self._run_optimization,
                args=(optimization_id, pair, X, y, model_type, config),
                daemon=True
            )
            self.active_optimizations[optimization_id] = thread
            thread.start()
            return optimization_id
        else:
            # Execução síncrona
            return self._run_optimization(optimization_id, pair, X, y, model_type, config)
    
    def _run_optimization(self, optimization_id: str, pair: str, X: np.ndarray, y: np.ndarray,
                         model_type: ModelType, config: OptimizationConfig) -> OptimizationResult:
        """Executa a otimização de hiperparâmetros"""
        start_time = time.time()
        
        self.logger.info(f"Iniciando otimização {optimization_id} para {pair} com {model_type.value}")
        
        try:
            if config.method == OptimizationMethod.OPTUNA_TPE and OPTUNA_AVAILABLE:
                result = self._optimize_with_optuna(optimization_id, pair, X, y, model_type, config)
            elif config.method == OptimizationMethod.GRID_SEARCH:
                result = self._optimize_with_grid_search(optimization_id, pair, X, y, model_type, config)
            elif config.method == OptimizationMethod.RANDOM_SEARCH:
                result = self._optimize_with_random_search(optimization_id, pair, X, y, model_type, config)
            elif config.method == OptimizationMethod.HYBRID_RS_BO:
                result = self._optimize_with_hybrid(optimization_id, pair, X, y, model_type, config)
            else:
                raise ValueError(f"Método de otimização não suportado: {config.method}")
            
            # Calcula tempo total
            result.optimization_time = time.time() - start_time
            
            # Salva resultado
            self._save_optimization_result(result)
            self.optimization_results.append(result)
            
            # Remove da lista de otimizações ativas
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
            
            self.logger.info(f"Otimização {optimization_id} concluída. Score: {result.best_score:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na otimização {optimization_id}: {e}")
            # Remove da lista de otimizações ativas
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
            raise
    
    def _optimize_with_optuna(self, optimization_id: str, pair: str, X: np.ndarray, y: np.ndarray,
                             model_type: ModelType, config: OptimizationConfig) -> OptimizationResult:
        """Otimização usando Optuna"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna não está disponível")
        
        # Configura sampler
        if config.method == OptimizationMethod.OPTUNA_TPE:
            sampler = TPESampler(seed=config.random_state)
        elif config.method == OptimizationMethod.OPTUNA_CMAES:
            sampler = CmaEsSampler(seed=config.random_state)
        else:
            sampler = TPESampler(seed=config.random_state)
        
        # Configura pruner se habilitado
        pruner = MedianPruner() if config.pruning else optuna.pruners.NopPruner()
        
        # Cria estudo
        study_name = f"study_{optimization_id}"
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        # Define função objetivo
        def objective(trial):
            params = self._suggest_params(trial, model_type, config)
            return self._evaluate_params(X, y, model_type, params, config)
        
        # Executa otimização
        study.optimize(
            objective,
            n_trials=config.n_trials,
            timeout=config.timeout_seconds,
            show_progress_bar=False
        )
        
        # Avalia melhor modelo com validação cruzada completa
        cv_scores = self._full_cross_validation(X, y, model_type, study.best_params, config)
        
        # Treina modelo final e extrai importância das features
        final_model = self._train_final_model(X, y, model_type, study.best_params)
        feature_importance = self._extract_feature_importance(final_model, X.shape[1])
        
        # Salva modelo e estudo
        model_path = self._save_model(final_model, optimization_id)
        study_path = self._save_study(study, optimization_id)
        
        return OptimizationResult(
            optimization_id=optimization_id,
            pair=pair,
            model_type=model_type,
            method=config.method,
            objective=config.objective,
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            optimization_time=0,  # Será preenchido depois
            timestamp=datetime.now(),
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            model_path=model_path,
            study_path=study_path
        )
    
    def _optimize_with_grid_search(self, optimization_id: str, pair: str, X: np.ndarray, y: np.ndarray,
                                  model_type: ModelType, config: OptimizationConfig) -> OptimizationResult:
        """Otimização usando Grid Search"""
        from sklearn.model_selection import GridSearchCV
        
        # Obtém espaço de parâmetros
        param_space = self._get_grid_search_space(model_type)
        
        # Cria modelo base
        model = self._create_model(model_type, {})
        
        # Configura validação cruzada
        cv = self._get_cv_strategy(config)
        
        # Executa grid search
        grid_search = GridSearchCV(
            model,
            param_space,
            cv=cv,
            scoring=self._get_scoring_metric(config.objective),
            n_jobs=config.n_jobs,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Extrai resultados
        cv_scores = list(grid_search.cv_results_['mean_test_score'])
        feature_importance = self._extract_feature_importance(grid_search.best_estimator_, X.shape[1])
        
        # Salva modelo
        model_path = self._save_model(grid_search.best_estimator_, optimization_id)
        
        return OptimizationResult(
            optimization_id=optimization_id,
            pair=pair,
            model_type=model_type,
            method=config.method,
            objective=config.objective,
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            n_trials=len(grid_search.cv_results_['params']),
            optimization_time=0,
            timestamp=datetime.now(),
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            model_path=model_path
        )
    
    def _optimize_with_random_search(self, optimization_id: str, pair: str, X: np.ndarray, y: np.ndarray,
                                    model_type: ModelType, config: OptimizationConfig) -> OptimizationResult:
        """Otimização usando Random Search"""
        from sklearn.model_selection import RandomizedSearchCV
        
        # Obtém espaço de parâmetros
        param_space = self._get_random_search_space(model_type)
        
        # Cria modelo base
        model = self._create_model(model_type, {})
        
        # Configura validação cruzada
        cv = self._get_cv_strategy(config)
        
        # Executa random search
        random_search = RandomizedSearchCV(
            model,
            param_space,
            n_iter=config.n_trials,
            cv=cv,
            scoring=self._get_scoring_metric(config.objective),
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        # Extrai resultados
        cv_scores = list(random_search.cv_results_['mean_test_score'])
        feature_importance = self._extract_feature_importance(random_search.best_estimator_, X.shape[1])
        
        # Salva modelo
        model_path = self._save_model(random_search.best_estimator_, optimization_id)
        
        return OptimizationResult(
            optimization_id=optimization_id,
            pair=pair,
            model_type=model_type,
            method=config.method,
            objective=config.objective,
            best_params=random_search.best_params_,
            best_score=random_search.best_score_,
            n_trials=config.n_trials,
            optimization_time=0,
            timestamp=datetime.now(),
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            model_path=model_path
        )
    
    def _suggest_params(self, trial, model_type: ModelType, config: OptimizationConfig) -> Dict[str, Any]:
        """Sugere parâmetros para um trial do Optuna"""
        param_space = config.custom_param_space or self.param_spaces[model_type]
        params = {}
        
        for param_name, param_config in param_space.items():
            if isinstance(param_config, tuple) and len(param_config) == 2:
                # Range numérico
                if isinstance(param_config[0], int) and isinstance(param_config[1], int):
                    params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[1])
            elif isinstance(param_config, list):
                # Escolha categórica
                params[param_name] = trial.suggest_categorical(param_name, param_config)
        
        return params
    
    def _evaluate_params(self, X: np.ndarray, y: np.ndarray, model_type: ModelType, 
                        params: Dict[str, Any], config: OptimizationConfig) -> float:
        """Avalia um conjunto de parâmetros"""
        try:
            # Cria modelo com parâmetros
            model = self._create_model(model_type, params)
            
            # Configura validação cruzada
            cv = self._get_cv_strategy(config)
            
            # Avalia com validação cruzada
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring=self._get_scoring_metric(config.objective),
                n_jobs=1  # Para evitar conflitos com Optuna
            )
            
            return scores.mean()
            
        except Exception as e:
            self.logger.warning(f"Erro ao avaliar parâmetros {params}: {e}")
            return 0.0
    
    def _create_model(self, model_type: ModelType, params: Dict[str, Any]):
        """Cria uma instância do modelo com os parâmetros especificados"""
        if model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(random_state=42, **params)
        elif model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(random_state=42, **params)
        elif model_type == ModelType.XGBOOST and XGB_AVAILABLE:
            return xgb.XGBClassifier(random_state=42, **params)
        elif model_type == ModelType.LIGHTGBM and LGB_AVAILABLE:
            return lgb.LGBMClassifier(random_state=42, **params)
        elif model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(random_state=42, **params)
        elif model_type == ModelType.SVM:
            return SVC(random_state=42, **params)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    def _get_cv_strategy(self, config: OptimizationConfig):
        """Retorna estratégia de validação cruzada apropriada"""
        # Para dados de séries temporais, usa TimeSeriesSplit
        # Para dados balanceados, usa StratifiedKFold
        return StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    
    def _get_scoring_metric(self, objective: OptimizationObjective) -> str:
        """Retorna métrica de scoring para sklearn"""
        mapping = {
            OptimizationObjective.ACCURACY: 'accuracy',
            OptimizationObjective.PRECISION: 'precision',
            OptimizationObjective.RECALL: 'recall',
            OptimizationObjective.F1_SCORE: 'f1',
            OptimizationObjective.ROC_AUC: 'roc_auc'
        }
        return mapping.get(objective, 'accuracy')
    
    def _full_cross_validation(self, X: np.ndarray, y: np.ndarray, model_type: ModelType,
                              params: Dict[str, Any], config: OptimizationConfig) -> List[float]:
        """Executa validação cruzada completa com os melhores parâmetros"""
        model = self._create_model(model_type, params)
        cv = self._get_cv_strategy(config)
        
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring=self._get_scoring_metric(config.objective),
            n_jobs=config.n_jobs
        )
        
        return scores.tolist()
    
    def _train_final_model(self, X: np.ndarray, y: np.ndarray, model_type: ModelType, params: Dict[str, Any]):
        """Treina modelo final com todos os dados"""
        model = self._create_model(model_type, params)
        model.fit(X, y)
        return model
    
    def _extract_feature_importance(self, model, n_features: int) -> Optional[Dict[str, float]]:
        """Extrai importância das features do modelo"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}
            elif hasattr(model, 'coef_'):
                # Para modelos lineares
                coefs = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                return {f"feature_{i}": float(coef) for i, coef in enumerate(coefs)}
        except Exception as e:
            self.logger.warning(f"Erro ao extrair importância das features: {e}")
        
        return None
    
    def _save_model(self, model, optimization_id: str) -> str:
        """Salva modelo otimizado"""
        model_dir = Path("models/optimized")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{optimization_id}_model.pkl"
        
        try:
            joblib.dump(model, model_path)
            return str(model_path)
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {e}")
            return None
    
    def _save_study(self, study, optimization_id: str) -> Optional[str]:
        """Salva estudo Optuna"""
        if not OPTUNA_AVAILABLE:
            return None
        
        study_dir = Path("studies")
        study_dir.mkdir(parents=True, exist_ok=True)
        
        study_path = study_dir / f"{optimization_id}_study.pkl"
        
        try:
            with open(study_path, 'wb') as f:
                pickle.dump(study, f)
            return str(study_path)
        except Exception as e:
            self.logger.error(f"Erro ao salvar estudo: {e}")
            return None
    
    def _get_grid_search_space(self, model_type: ModelType) -> Dict:
        """Converte espaço de parâmetros para Grid Search"""
        param_space = self.param_spaces[model_type]
        grid_space = {}
        
        for param_name, param_config in param_space.items():
            if isinstance(param_config, tuple) and len(param_config) == 2:
                # Para ranges numéricos, cria lista de valores
                if isinstance(param_config[0], int):
                    grid_space[param_name] = list(range(param_config[0], param_config[1] + 1, 
                                                       max(1, (param_config[1] - param_config[0]) // 5)))
                else:
                    grid_space[param_name] = np.linspace(param_config[0], param_config[1], 5).tolist()
            elif isinstance(param_config, list):
                grid_space[param_name] = param_config
        
        return grid_space
    
    def _get_random_search_space(self, model_type: ModelType) -> Dict:
        """Converte espaço de parâmetros para Random Search"""
        from scipy.stats import randint, uniform
        
        param_space = self.param_spaces[model_type]
        random_space = {}
        
        for param_name, param_config in param_space.items():
            if isinstance(param_config, tuple) and len(param_config) == 2:
                if isinstance(param_config[0], int):
                    random_space[param_name] = randint(param_config[0], param_config[1] + 1)
                else:
                    random_space[param_name] = uniform(param_config[0], 
                                                      param_config[1] - param_config[0])
            elif isinstance(param_config, list):
                random_space[param_name] = param_config
        
        return random_space
    
    def _save_optimization_result(self, result: OptimizationResult) -> None:
        """Salva resultado da otimização no banco de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO optimization_results 
                    (optimization_id, pair, model_type, method, objective, best_params,
                     best_score, n_trials, optimization_time, timestamp, cv_scores,
                     feature_importance, model_path, study_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.optimization_id,
                    result.pair,
                    result.model_type.value,
                    result.method.value,
                    result.objective.value,
                    json.dumps(result.best_params),
                    result.best_score,
                    result.n_trials,
                    result.optimization_time,
                    result.timestamp.isoformat(),
                    json.dumps(result.cv_scores),
                    json.dumps(result.feature_importance) if result.feature_importance else None,
                    result.model_path,
                    result.study_path
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Erro ao salvar resultado da otimização: {e}")
    
    def get_best_params(self, pair: str, model_type: ModelType, 
                       objective: OptimizationObjective = None) -> Optional[Dict[str, Any]]:
        """Retorna os melhores parâmetros para um par e modelo"""
        filtered_results = [
            r for r in self.optimization_results
            if r.pair == pair and r.model_type == model_type
        ]
        
        if objective:
            filtered_results = [r for r in filtered_results if r.objective == objective]
        
        if not filtered_results:
            return None
        
        # Retorna resultado com melhor score
        best_result = max(filtered_results, key=lambda x: x.best_score)
        return best_result.best_params
    
    def get_optimization_history(self, pair: str = None, model_type: ModelType = None,
                               days: int = 30) -> List[Dict]:
        """Retorna histórico de otimizações"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        filtered_results = [
            r for r in self.optimization_results
            if r.timestamp > cutoff_date
        ]
        
        if pair:
            filtered_results = [r for r in filtered_results if r.pair == pair]
        
        if model_type:
            filtered_results = [r for r in filtered_results if r.model_type == model_type]
        
        return [asdict(result) for result in filtered_results]
    
    def get_optimization_status(self, optimization_id: str) -> Dict[str, Any]:
        """Retorna status de uma otimização"""
        # Verifica se está em andamento
        if optimization_id in self.active_optimizations:
            thread = self.active_optimizations[optimization_id]
            return {
                "status": "running" if thread.is_alive() else "completed",
                "optimization_id": optimization_id
            }
        
        # Verifica se foi concluída
        for result in self.optimization_results:
            if result.optimization_id == optimization_id:
                return {
                    "status": "completed",
                    "optimization_id": optimization_id,
                    "best_score": result.best_score,
                    "n_trials": result.n_trials,
                    "optimization_time": result.optimization_time
                }
        
        return {"status": "not_found", "optimization_id": optimization_id}
    
    def cancel_optimization(self, optimization_id: str) -> bool:
        """Cancela uma otimização em andamento"""
        if optimization_id in self.active_optimizations:
            # Note: Optuna não suporta cancelamento direto
            # Esta implementação é limitada
            self.logger.warning(f"Tentativa de cancelar otimização {optimization_id}")
            return True
        return False
    
    def get_performance_comparison(self, pair: str, model_types: List[ModelType] = None) -> Dict:
        """Compara performance entre diferentes modelos"""
        if model_types is None:
            model_types = list(ModelType)
        
        comparison = {}
        
        for model_type in model_types:
            results = [
                r for r in self.optimization_results
                if r.pair == pair and r.model_type == model_type
            ]
            
            if results:
                best_result = max(results, key=lambda x: x.best_score)
                comparison[model_type.value] = {
                    "best_score": best_result.best_score,
                    "best_params": best_result.best_params,
                    "optimization_time": best_result.optimization_time,
                    "n_trials": best_result.n_trials,
                    "timestamp": best_result.timestamp.isoformat()
                }
        
        return comparison
    
    def _optimize_with_hybrid(self, optimization_id: str, pair: str, X: np.ndarray, y: np.ndarray,
                             model_type: ModelType, config: OptimizationConfig) -> OptimizationResult:
        """Estratégia híbrida: Random Search seguido de Bayesian Optimization (Optuna TPE)"""
        # Etapa 1: Random Search rápido para mapear espaço
        rs_trials = max(10, int(config.n_trials * 0.3))
        config_rs = copy.deepcopy(config)
        config_rs.method = OptimizationMethod.RANDOM_SEARCH
        config_rs.n_trials = rs_trials
        random_result = self._optimize_with_random_search(optimization_id + "_rs", pair, X, y, model_type, config_rs)

        # Etapa 2: Ajustar espaço de busca ao redor dos melhores parâmetros
        default_space = self.param_spaces[model_type]
        narrowed_space = {}
        for param_name, best_value in random_result.best_params.items():
            if param_name not in default_space:
                continue
            param_def = default_space[param_name]
            if isinstance(param_def, tuple) and len(param_def) == 2:
                low, high = param_def
                if isinstance(best_value, int):
                    span = max(1, int((high - low) * 0.1))
                    narrowed_space[param_name] = (max(low, best_value - span), min(high, best_value + span))
                else:
                    span = (high - low) * 0.1
                    narrowed_space[param_name] = (max(low, best_value - span), min(high, best_value + span))
            else:
                # Mantém espaço original para variáveis categóricas
                narrowed_space[param_name] = param_def

        # Etapa 3: Bayesian Optimization com Optuna TPE usando espaço reduzido
        bo_trials = config.n_trials - rs_trials
        if bo_trials <= 0:
            return random_result
        config_bo = copy.deepcopy(config)
        config_bo.method = OptimizationMethod.OPTUNA_TPE
        config_bo.n_trials = bo_trials
        config_bo.custom_param_space = narrowed_space

        bo_result = self._optimize_with_optuna(optimization_id + "_bo", pair, X, y, model_type, config_bo)

        # Retorna o melhor entre as duas etapas
        return bo_result if bo_result.best_score >= random_result.best_score else random_result

    def suggest_next_optimization(self, pair: str) -> Dict[str, Any]:
        """Sugere próxima otimização baseada no histórico"""
        # Analisa histórico para sugerir melhorias
        recent_results = [
            r for r in self.optimization_results
            if r.pair == pair and r.timestamp > datetime.now() - timedelta(days=7)
        ]
        
        if not recent_results:
            return {
                "suggestion": "Executar otimização inicial",
                "model_type": ModelType.XGBOOST.value,
                "method": OptimizationMethod.OPTUNA_TPE.value,
                "priority": "high"
            }
        
        # Encontra modelo com melhor performance
        best_result = max(recent_results, key=lambda x: x.best_score)
        
        suggestions = []
        
        # Se performance está baixa, sugere otimização mais agressiva
        if best_result.best_score < 0.6:
            suggestions.append({
                "suggestion": "Performance baixa - otimização agressiva recomendada",
                "model_type": ModelType.XGBOOST.value,
                "method": OptimizationMethod.OPTUNA_TPE.value,
                "n_trials": 200,
                "priority": "high"
            })
        
        # Se não há otimização recente, sugere manutenção
        last_optimization = max(recent_results, key=lambda x: x.timestamp)
        if (datetime.now() - last_optimization.timestamp).days > 3:
            suggestions.append({
                "suggestion": "Otimização de manutenção",
                "model_type": best_result.model_type.value,
                "method": OptimizationMethod.OPTUNA_TPE.value,
                "n_trials": 50,
                "priority": "medium"
            })
        
        return suggestions[0] if suggestions else {"suggestion": "Nenhuma otimização necessária"}

if __name__ == "__main__":
    # Exemplo de uso
    optimizer = HyperparameterOptimizer()
    
    # Simula dados de exemplo
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, 1000)
    
    # Configura otimização
    config = OptimizationConfig(
        model_type=ModelType.RANDOM_FOREST,
        method=OptimizationMethod.OPTUNA_TPE,
        n_trials=20,
        timeout_seconds=300
    )
    
    try:
        # Executa otimização
        result = optimizer.optimize_model("EURUSD", X, y, ModelType.RANDOM_FOREST, config)
        
        print(f"Otimização concluída:")
        print(f"- Score: {result.best_score:.4f}")
        print(f"- Parâmetros: {result.best_params}")
        print(f"- Tempo: {result.optimization_time:.2f}s")
        print(f"- Trials: {result.n_trials}")
        
        # Mostra comparação de performance
        comparison = optimizer.get_performance_comparison("EURUSD")
        print(f"\nComparação de modelos: {comparison}")
        
        # Sugere próxima otimização
        suggestion = optimizer.suggest_next_optimization("EURUSD")
        print(f"\nSugestão: {suggestion}")
        
    except Exception as e:
        print(f"Erro: {e}")