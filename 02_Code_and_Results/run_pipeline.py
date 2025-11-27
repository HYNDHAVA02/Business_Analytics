import subprocess
import sys
import time

scripts = [
    "1_ingest_validate.py",
    "2_process_engineer.py",
    "3_eda_hypothesis.py",
    "4_modeling_tuning.py",
    "5_evaluation_audit.py"
]

def run_script(script_name):
    print(f"==================================================")
    print(f"Running {script_name}...")
    print(f"==================================================")
    start_time = time.time()
    try:
        # Run script using the current python interpreter
        result = subprocess.run(
            [sys.executable, script_name], 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(result.stdout)
        print(f"✅ {script_name} completed successfully in {time.time() - start_time:.2f}s.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} FAILED.")
        print("Error Output:")
        print(e.stderr)
        print("Standard Output:")
        print(e.stdout)
        sys.exit(1)

if __name__ == "__main__":
    total_start = time.time()
    for script in scripts:
        run_script(script)
    
    print(f"\n🎉 Full pipeline completed in {time.time() - total_start:.2f}s.")
