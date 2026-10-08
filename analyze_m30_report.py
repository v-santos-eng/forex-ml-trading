import json

def analyze_report(file_path):
    """
    Reads a JSON training report and prints a formatted summary.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    except FileNotFoundError:
        print(f"Error: Report file not found at '{file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'")
        return

    print(f"--- Analysis of Report: {file_path} ---")
    print(f"Timestamp: {report.get('timestamp')}")
    
    summary = report.get('training_summary', {})
    print("\n--- Training Summary (M30) ---")
    print(f"Total Pairs Trained: {summary.get('total_pairs_trained')}")
    print(f"Models Directory: {summary.get('models_directory')}")

    best_models = summary.get('best_models', {})
    if best_models:
        print("\n--- Best Overall Models (M30) ---")
        reg = best_models.get('regression', {})
        clf = best_models.get('classification', {})
        if reg:
            print(f"Best Regression -> Pair: {reg.get('pair')}, Model: {reg.get('model')}, R2: {reg.get('r2'):.4f}")
        if clf:
            print(f"Best Classification -> Pair: {clf.get('pair')}, Model: {clf.get('model')}, F1: {clf.get('f1'):.4f}")

    print("\n--- Results per Pair (M30) ---")
    pair_results = report.get('pair_results', {})
    if not pair_results:
        print("No pair-specific results found in the report.")
    else:
        for pair, data in pair_results.items():
            s = data.get('summary', {})
            print(f"\n== {pair} ==")
            print(f"  Samples (Train/Val): {s.get('training_samples')} / {s.get('validation_samples')}")
            print(f"  Best R2: {s.get('best_regression_r2', 0.0):.4f} (Model: {data.get('best_regression_model')})")
            print(f"  Best F1: {s.get('best_classification_f1', 0.0):.4f} (Model: {data.get('best_classification_model')})")

if __name__ == "__main__":
    # The latest report file based on previous logs
    report_file = 'model_training_report_20250828_160459.json'
    analyze_report(report_file)