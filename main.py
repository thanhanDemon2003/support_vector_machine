import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*80}")
    print(f"RUNNING SCRIPT: {script_name}")
    print(f"{'='*80}\n", flush=True)
    subprocess.run([sys.executable, script_name])

def main():
    scripts = [
        "test_binary_custom.py",
        "svm_binary.py",
        "test_all_models.py"
    ]
    for script in scripts:
        run_script(script)

if __name__ == "__main__":
    main()

