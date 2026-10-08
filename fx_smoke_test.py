import json
from pathlib import Path
import logging

class FXSmokeTester:
    def __init__(self, models_dir: str, baseline_report: str, timeframe: str):
        self.models_dir = Path(models_dir)
        self.timeframe = timeframe
        self.baseline = self._load_baseline(baseline_report)
        self.logger = logging.getLogger('FXSmokeTest')

    def _load_baseline(self, path: str) -> dict:
        if not Path(path).exists():
            self.logger.error(f'Baseline não encontrado: {path}')
            exit(1)
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f'Arquivo de baseline não encontrado: {path}')
            return {}

    def _load_current_metrics(self) -> dict:
        eval_dir = f'model_evaluation_{self.timeframe}'
        report_dir = self.models_dir.parent / eval_dir
        
        if not report_dir.exists():
            self.logger.error(f"O diretório de avaliação '{report_dir}' não foi encontrado.")
            exit(1)

        sorted_reports = sorted(report_dir.glob('evaluation_report_*.json'), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not sorted_reports:
            self.logger.error(f'Nenhum relatório encontrado em {report_dir}')
            exit(1)
            
        current_report_path = sorted_reports[0]
        self.logger.info(f"Carregando relatório de avaliação mais recente: {current_report_path}")

        try:
            with open(current_report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f'Relatório atual não encontrado em {current_report_path}')
            return {}
        except json.JSONDecodeError:
            self.logger.error(f"Erro ao decodificar o JSON do relatório: {current_report_path}")
            return {}

    def compare_metrics(self, threshold: float = 0.1) -> dict:
        current = self._load_current_metrics()
        results = {'passed': 0, 'failed': 0, 'details': {}}

        for pair in current.get('results', []):
            pair_name = pair['pair']
            baseline_pair = next((p for p in self.baseline.get('results', []) if p['pair'] == pair_name), None)
            
            if not baseline_pair:
                continue

            for model in pair['models']:
                model_name = model['model_type']
                baseline_model = next((m for m in baseline_pair['models'] if m['model_type'] == model_name), None)
                
                if baseline_model:
                    r2_improvement = model['metrics']['test_r2'] - baseline_model['metrics']['test_r2']
                    f1_improvement = model['metrics']['test_f1'] - baseline_model['metrics']['test_f1']
                    
                    passed = (
                        r2_improvement >= -threshold and
                        f1_improvement >= -threshold/2 and
                        model['metrics']['test_r2'] > -5.0  # Tolerância para R² negativo
                    )
                    
                    results['details'].setdefault(pair_name, {})[model_name] = {
                        'r2_current': model['metrics']['test_r2'],
                        'r2_baseline': baseline_model['metrics']['test_r2'],
                        'f1_current': model['metrics']['test_f1'],
                        'f1_baseline': baseline_model['metrics']['test_f1'],
                        'passed': passed
                    }
                    
                    if passed:
                        results['passed'] += 1
                    else:
                        results['failed'] += 1

        return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FX Model Smoke Test')
    parser.add_argument('--timeframe', type=str, required=True, choices=['M5','M15','M30'],
                        help='Timeframe para análise (M5, M15, M30)')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Diretório com modelos treinados')
    parser.add_argument('--baseline_report', type=str, required=True,
                        help='Caminho completo para o relatório baseline')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Threshold de melhoria para considerar válido')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        encoding='utf-8')
    
    tester = FXSmokeTester(args.models_dir, args.baseline_report, args.timeframe)
    results = tester.compare_metrics(args.threshold)
    
    print('\n=== Resultados do Smoke Test ===')
    print(f'Modelos aprovados: {results["passed"]}')
    print(f'Modelos reprovados: {results["failed"]}')
    
    for pair, models in results['details'].items():
        print(f'\nPar: {pair}')
        for model, data in models.items():
            status = 'APROVADO' if data['passed'] else 'REPROVADO'
            print(f'  {model} ({status}):')
            print(f'    R²: {data["r2_baseline"]:.3f} → {data["r2_current"]:.3f}')
            print(f'    F1: {data["f1_baseline"]:.3f} → {data["f1_current"]:.3f}')