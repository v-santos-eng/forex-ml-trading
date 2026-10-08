import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'detailed_decisions.db')


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def pick_table(cur):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    names = {r[0] for r in cur.fetchall()}
    if 'detailed_decisions' in names:
        return 'detailed_decisions'
    if 'decisions' in names:
        return 'decisions'
    return None


def main():
    result = {
        "db_path": DB_PATH,
        "exists": os.path.exists(DB_PATH),
        "summary": {},
        "samples": {}
    }

    if not os.path.exists(DB_PATH):
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    table_name = pick_table(cur)
    if not table_name:
        result["summary"]["error"] = "Nenhuma tabela de decisões encontrada no DB."
        print(json.dumps(result, indent=2, ensure_ascii=False))
        conn.close()
        return

    # Total decisions
    cur.execute(f"SELECT COUNT(*) FROM {table_name};")
    total_decisions = cur.fetchone()[0]

    # Count NO_ACTION with INVALID_STOPS_PREVENTED (in bot_action or reasoning_factors)
    cur.execute(
        f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE action = 'NO_ACTION' AND (
            (bot_action IS NOT NULL AND bot_action LIKE '%INVALID_STOPS_PREVENTED%') OR
            (reasoning_factors IS NOT NULL AND reasoning_factors LIKE '%INVALID_STOPS_PREVENTED%')
        );
        """
    )
    no_action_invalid_stops = cur.fetchone()[0]

    # Count entries with EXECUTION_FAILED and error_code 10016 (in bot_action JSON text)
    cur.execute(
        f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE bot_action IS NOT NULL
          AND bot_action LIKE '%EXECUTION_FAILED%'
          AND bot_action LIKE '%10016%';
        """
    )
    error_10016 = cur.fetchone()[0]

    # Confidence stats for those categories
    # PREVENTED
    cur.execute(
        f"""
        SELECT confidence FROM {table_name}
        WHERE action = 'NO_ACTION' AND (
            (bot_action IS NOT NULL AND bot_action LIKE '%INVALID_STOPS_PREVENTED%') OR
            (reasoning_factors IS NOT NULL AND reasoning_factors LIKE '%INVALID_STOPS_PREVENTED%')
        )
        """
    )
    conf_prev = [safe_float(r[0]) for r in cur.fetchall() if r[0] is not None]
    conf_prev = [c for c in conf_prev if c is not None]

    # 10016 failed
    cur.execute(
        f"""
        SELECT confidence FROM {table_name}
        WHERE bot_action IS NOT NULL
          AND bot_action LIKE '%EXECUTION_FAILED%'
          AND bot_action LIKE '%10016%'
        """
    )
    conf_10016 = [safe_float(r[0]) for r in cur.fetchall() if r[0] is not None]
    conf_10016 = [c for c in conf_10016 if c is not None]

    def make_stats(vals):
        return {
            "count": len(vals),
            "avg": (sum(vals) / len(vals)) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }

    result["summary"].update({
        "table": table_name,
        "total_decisions": total_decisions,
        "no_action_invalid_stops_prevented": no_action_invalid_stops,
        "error_code_10016_mentions": error_10016,
        "confidence_stats_prevented": make_stats(conf_prev),
        "confidence_stats_10016": make_stats(conf_10016),
    })

    # Samples: last 10 NO_ACTION INVALID_STOPS_PREVENTED
    cur.execute(
        f"""
        SELECT id, timestamp, symbol, action, confidence, reasoning_factors, bot_action
        FROM {table_name}
        WHERE action = 'NO_ACTION' AND (
            (bot_action IS NOT NULL AND bot_action LIKE '%INVALID_STOPS_PREVENTED%') OR
            (reasoning_factors IS NOT NULL AND reasoning_factors LIKE '%INVALID_STOPS_PREVENTED%')
        )
        ORDER BY datetime(timestamp) DESC
        LIMIT 10;
        """
    )
    rows = cur.fetchall()
    na_samples = []
    for r in rows:
        rid, ts, sym, act, conf, reasons, bot_action = r
        snip_reason = None
        try:
            if isinstance(reasons, str):
                # reasons is a JSON array string
                arr = json.loads(reasons)
                if isinstance(arr, list):
                    snip_reason = "; ".join(map(str, arr))[:240]
        except Exception:
            pass
        has_tag = ('INVALID_STOPS_PREVENTED' in (reasons or '')) or ('INVALID_STOPS_PREVENTED' in (bot_action or ''))
        na_samples.append({
            "id": rid,
            "timestamp": ts,
            "symbol": sym,
            "action": act,
            "confidence": safe_float(conf),
            "reasoning_factors": snip_reason,
            "tag_present": has_tag,
        })

    # Samples: last 10 EXECUTION_FAILED 10016
    cur.execute(
        f"""
        SELECT id, timestamp, symbol, action, confidence, bot_action
        FROM {table_name}
        WHERE bot_action IS NOT NULL
          AND bot_action LIKE '%EXECUTION_FAILED%'
          AND bot_action LIKE '%10016%'
        ORDER BY datetime(timestamp) DESC
        LIMIT 10;
        """
    )
    rows = cur.fetchall()
    fail_samples = []
    for r in rows:
        rid, ts, sym, act, conf, bot_action = r
        err_code = None
        action_type = None
        try:
            bj = json.loads(bot_action) if isinstance(bot_action, str) else bot_action
            if isinstance(bj, dict):
                err_code = bj.get('parameters', {}).get('retcode') or bj.get('error_code')
                action_type = bj.get('action_type')
        except Exception:
            pass
        fail_samples.append({
            "id": rid,
            "timestamp": ts,
            "symbol": sym,
            "action": act,
            "confidence": safe_float(conf),
            "action_type": action_type,
            "error_code": err_code,
        })

    result["samples"] = {
        "no_action_invalid_stops_prevented": na_samples,
        "execution_failed_10016": fail_samples,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()