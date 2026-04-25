import os
import pandas as pd
import numpy as np

def convert():
    best_dir = r"C:\Users\lukep\Documents\MVS\backend\ml\anomaly_detection\results\week2\window_sweep_results\window_configs\ws512_st16_thr0p25"
    out_dir = os.path.join(best_dir, "tinyml")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading windowed_train.csv...")
    df_train = pd.read_csv(os.path.join(best_dir, "windowed_train.csv"))
    print("Loading windowed_val.csv...")
    df_val = pd.read_csv(os.path.join(best_dir, "windowed_val.csv"))
    print("Loading windowed_test.csv...")
    df_test = pd.read_csv(os.path.join(best_dir, "windowed_test.csv"))
    
    label_col = "window_label"
    
    x_train = df_train.iloc[:, 7:].values.astype(np.float32)
    y_train = df_train[label_col].values.astype(np.int32)
    
    x_val = df_val.iloc[:, 7:].values.astype(np.float32)
    y_val = df_val[label_col].values.astype(np.int32)
    
    x_test = df_test.iloc[:, 7:].values.astype(np.float32)
    y_test = df_test[label_col].values.astype(np.int32)
    
    out_file = os.path.join(out_dir, "dataset.npz")
    print(f"Saving to {out_file}...")
    np.savez(
        out_file,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test
    )
    print("Done!")

if __name__ == "__main__":
    convert()
