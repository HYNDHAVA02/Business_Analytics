import os
import yaml
import shutil

def setup_directories():
    """Creates the project directory structure."""
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "reports",
        "src",
        "visuals"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

def create_schema():
    """Generates the schema.yml file for Great Expectations/Pandera."""
    schema = {
        "columns": {
            "Gender": {"type": "int", "checks": {"isin": [0, 1]}},
            "ParentalEducation": {"type": "float", "checks": {"min": 0, "max": 4}},
            "StudyTimeWeekly": {"type": "float", "checks": {"min": 0}},
            "Absences": {"type": "int", "checks": {"min": 0}},
            "Tutoring": {"type": "int", "checks": {"isin": [0, 1]}},
            "Extracurricular": {"type": "int", "checks": {"isin": [0, 1]}},
            "InternetAccess": {"type": "int", "checks": {"isin": [0, 1]}},
            "GPA": {"type": "float", "checks": {"min": 0.0, "max": 4.0}},
            "GradeClass": {"type": "int", "checks": {"min": 0, "max": 4}},
        },
        "checks": {
            "no_nulls": True
        }
    }
    
    with open("schema.yml", "w") as f:
        yaml.dump(schema, f, default_flow_style=False)
    print("Created schema.yml")

if __name__ == "__main__":
    setup_directories()
    create_schema()
    print("Setup complete.")
