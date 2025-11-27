import pandas as pd
import yaml
import os
import sys

def validate_data(input_path, schema_path, output_path):
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        sys.exit(1)

    print(f"Loading schema from {schema_path}...")
    with open(schema_path, 'r') as f:
        schema = yaml.safe_load(f)

    print("Starting validation...")
    errors = []

    # Column existence check
    for col in schema['columns']:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    if errors:
        for e in errors: print(e)
        sys.exit(1)

    # Type and Range checks
    for col, rules in schema['columns'].items():
        # Type conversion/check
        try:
            if rules['type'] == 'int':
                # Check if safe to convert
                if not pd.api.types.is_integer_dtype(df[col]):
                     # Allow float if all are integers
                    if pd.api.types.is_float_dtype(df[col]) and df[col].dropna().apply(lambda x: x.is_integer()).all():
                         df[col] = df[col].astype(int)
                    else:
                         errors.append(f"Column {col} expected int, found {df[col].dtype}")
            elif rules['type'] == 'float':
                df[col] = df[col].astype(float)
        except Exception as e:
            errors.append(f"Column {col} type conversion error: {e}")

        # Value checks
        if 'checks' in rules:
            checks = rules['checks']
            if 'min' in checks:
                if df[col].min() < checks['min']:
                    errors.append(f"Column {col} has values < {checks['min']}")
            if 'max' in checks:
                if df[col].max() > checks['max']:
                    errors.append(f"Column {col} has values > {checks['max']}")
            if 'isin' in checks:
                invalid = df[~df[col].isin(checks['isin'])]
                if not invalid.empty:
                    errors.append(f"Column {col} has invalid values not in {checks['isin']}")

    if errors:
        print("Validation FAILED with the following errors:")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)
    else:
        print("Validation PASSED.")
        
    # Save validated data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Validated data saved to {output_path}")

if __name__ == "__main__":
    validate_data(
        input_path="combined_students_final.csv",
        schema_path="schema.yml",
        output_path="data/raw/validated_students.csv"
    )
