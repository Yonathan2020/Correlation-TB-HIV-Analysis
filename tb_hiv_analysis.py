# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import chardet
import argparse
import os
import sys
from tkinter import Tk, filedialog

def detect_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)  # Read first 10KB to guess encoding
    return chardet.detect(rawdata)['encoding']

def load_data(file_path):
    """Load data with proper encoding detection."""
    try:
        # First try UTF-8 (most common)
        try:
            df = pd.read_csv(file_path, header=0, encoding='utf-8')
            print("Loaded with UTF-8 encoding")
        except UnicodeDecodeError:
            # If UTF-8 fails, detect encoding
            encoding = detect_encoding(file_path)
            print(f"Using detected encoding: {encoding}")
            df = pd.read_csv(file_path, header=0, encoding=encoding)
    except Exception as e:
        print(f"Failed to load file: {e}")
        print("\nTrying common alternative encodings...")
        for enc in ['latin1', 'iso-8859-1', 'cp1252']:  # Common alternatives
            try:
                df = pd.read_csv(file_path, header=0, encoding=enc)
                print(f"Success with {enc} encoding")
                break
            except:
                continue
        else:
            raise Exception("Could not determine proper encoding")
    return df

def analyze_data(df, output_dir='.'):
    """Perform data analysis on the loaded dataframe."""
    # 4. Initial data inspection
    print("\n=== Dataset Info ===")
    print(f"Shape: {df.shape}\n")
    print("Data Types:")
    print(df.dtypes)
    print("\nFirst 3 rows:")
    print(df.head(3))  # Using print instead of display for VS Code compatibility

    # 5. Identify numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print("\nNumeric columns:", numeric_cols)
    print("Non-numeric columns:", non_numeric_cols)

    # 6. Handle missing values ONLY in numeric columns
    print("\n=== Missing Values in Numeric Columns ===")
    print("Original missing values:")
    print(df[numeric_cols].isnull().sum())

    # Create versions for comparison
    df_nan = df.copy()  # Original with all NaNs preserved
    df_zero_num = df.copy()  # Only numeric NaNs will be filled with 0

    # Replace only numeric missing values with 0
    df_zero_num[numeric_cols] = df_zero_num[numeric_cols].fillna(0)

    # 7. Correlation analysis (numeric columns only)
    plt.figure(figsize=(16, 6))

    # Before imputation
    plt.subplot(1, 2, 1)
    if len(numeric_cols) > 1:
        sns.heatmap(df_nan[numeric_cols].corr(),
                    annot=True,
                    cmap='coolwarm',
                    vmin=-1,
                    vmax=1,
                    fmt=".2f")
        plt.title('Correlation (With Original Missing Values)', pad=20)
    else:
        plt.text(0.5, 0.5, 'Not enough numeric columns', ha='center')

    # After numeric imputation
    plt.subplot(1, 2, 2)
    if len(numeric_cols) > 1:
        sns.heatmap(df_zero_num[numeric_cols].corr(),
                    annot=True,
                    cmap='coolwarm',
                    vmin=-1,
                    vmax=1,
                    fmt=".2f")
        plt.title('Correlation (Numeric Missing Values → 0)', pad=20)
    else:
        plt.text(0.5, 0.5, 'Not enough numeric columns', ha='center')

    plt.tight_layout()
    
    # Save the heatmap
    heatmap_path = os.path.join(output_dir, 'heatmap_comparison.png')
    plt.savefig(heatmap_path)
    print(f"Heatmap saved to: {heatmap_path}")
    
    # Show the plot (optional in VS Code)
    plt.show()

    # 8. Missing value report
    print("\n=== Missing Value Report ===")
    missing_report = pd.DataFrame({
        'Missing Count': df[numeric_cols].isnull().sum(),
        'Percentage (%)': (df[numeric_cols].isnull().mean() * 100).round(2)
    })
    print(missing_report[missing_report['Missing Count'] > 0])
    
    # Save the missing value report
    report_path = os.path.join(output_dir, 'missing_values_report.csv')
    missing_report.to_csv(report_path)
    print(f"Missing values report saved to: {report_path}")

    if len(numeric_cols) > 1:
        print("\n=== Top Correlations After Imputation ===")
        corr_matrix = df_zero_num[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        top_corr = upper_tri.unstack().sort_values(ascending=False).dropna()
        print(top_corr.head(5))

    # Show non-numeric missing values (for reference)
    if len(non_numeric_cols) > 0:
        print("\n=== Non-Numeric Missing Values (Not Imputed) ===")
        print(df[non_numeric_cols].isnull().sum())
    
    return df_zero_num

def select_file_dialog():
    """Open a file dialog to select a CSV file."""
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring the dialog to the front
    file_path = filedialog.askopenfilename(
        title="Select TB-HIV CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def main():
    """Main function to run the TB-HIV analysis."""
    parser = argparse.ArgumentParser(description='TB-HIV Data Analysis Tool')
    parser.add_argument('--file', '-f', type=str, help='Path to the CSV file')
    parser.add_argument('--output', '-o', type=str, default='.', help='Output directory for results')
    args = parser.parse_args()
    
    file_path = args.file
    
    # If no file path is provided, open a file dialog
    if not file_path:
        print("No file path provided. Opening file selection dialog...")
        file_path = select_file_dialog()
        
    # If still no file path (user canceled dialog), exit
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading data from: {file_path}")
    df = load_data(file_path)
    
    # Analyze the data
    analyze_data(df, args.output)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()