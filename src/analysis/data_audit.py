import os
import json
import pandas as pd
import numpy as np
import argparse

def audit_dataset(csv_path, expected_schema_path=None):
    """Perform a comprehensive audit of a dataset CSV."""
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    print(f"Auditing dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    audit_results = {
        "filename": os.path.basename(csv_path),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "data_types": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "summary_stats": df.describe().to_dict()
    }

    # Check for anomalies in timestamps
    if "Timestamp" in df.columns:
        try:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            dt = df["Timestamp"].diff().dt.total_seconds().dropna()
            audit_results["sampling_stats"] = {
                "median_interval_ms": float(dt.median() * 1000),
                "inferred_fs_hz": float(1.0 / dt.median()) if dt.median() > 0 else 0,
                "max_gap_ms": float(dt.max() * 1000),
                "min_gap_ms": float(dt.min() * 1000)
            }
        except Exception as e:
            print(f"Warning: Could not parse timestamps: {e}")

    # Schema Validation
    if expected_schema_path and os.path.exists(expected_schema_path):
        with open(expected_schema_path) as f:
            schema = json.load(f)
            expected_cols = schema.get("columns", [])
            if expected_cols:
                missing = [c for c in expected_cols if c not in df.columns]
                extra = [c for c in df.columns if c not in expected_cols]
                audit_results["schema_validation"] = {
                    "valid": len(missing) == 0,
                    "missing_columns": missing,
                    "extra_columns": extra
                }

    print("\n--- Audit Summary ---")
    print(f"Rows: {audit_results['row_count']}")
    print(f"Columns: {', '.join(audit_results['columns'])}")
    if "sampling_stats" in audit_results:
        print(f"Inferred Frequency: {audit_results['sampling_stats']['inferred_fs_hz']:.2f} Hz")
    
    return audit_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit a dataset for TinyML Anomaly Detection.")
    parser.add_argument("csv_path", type=str, help="Path to the dataset CSV file")
    parser.add_argument("--schema", type=str, help="Path to expected schema JSON")
    parser.add_argument("--out", type=str, help="Path to save audit results JSON")
    
    args = parser.parse_args()
    
    results = audit_dataset(args.csv_path, args.schema)
    
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nAudit results saved to {args.out}")