import os
import sys
import numpy as np
import time
import subprocess

# Add current directory to path
sys.path.append(os.getcwd())

from eval_baseline import solve_lkh_windy, get_lkh_executable

def test_single_run():
    print("=== LKH DIAGNOSTIC TOOL (V2) ===")
    
    # 1. Setup paths
    debug_dir = "results/debug_deep"
    if os.path.exists(debug_dir):
        import shutil
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)
    
    name = "debug_instance"
    executable = get_lkh_executable()
    
    print(f"1. Checking Executable path: {executable}")
    if not os.path.isfile(executable):
        print(f"❌ FATAL: LKH executable not found at '{executable}'")
        return
    # SKIPPING INTERACTIVE CHECK TO AVOID HANGING
    
    # 2. Create Dummy Data
    print("2. Generating Dummy Data...")
    loc = np.random.rand(20, 2)
    wind = np.array([1.0, 0.0])
    alpha = 3.0
    
    # 3. Run Solver with TIMER
    print("3. Running solve_lkh_windy()...")
    start_time = time.time()
    
    # Passing runs=1 and disable_cache=True
    result = solve_lkh_windy(executable, debug_dir, name, loc, wind, alpha, runs=1, disable_cache=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"4. Execution finished in {duration:.4f} seconds.")
    
    # 4. Analyze Result
    if result is None:
        print("\n❌ FAILURE: Function returned None.")
        
        # Check if logs exist
        log_file = os.path.join(debug_dir, f"{name}.log")
        if os.path.exists(log_file):
            print(f"\n--- CONTENT OF {log_file} ---")
            with open(log_file, 'r') as f:
                print(f.read())
            print("-----------------------------")
        else:
            print("❌ Log file was NOT created. Permission or path issue?")
            
    else:
        cost, tour, lkh_duration = result
        print(f"\n✅ SUCCESS!")
        print(f"   Calculated Cost: {cost}")
        print(f"   Tour found: {tour}")
        print(f"   LKH Duration: {lkh_duration:.4f}s")

if __name__ == "__main__":
    test_single_run()