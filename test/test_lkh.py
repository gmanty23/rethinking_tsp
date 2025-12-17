import os
import numpy as np
from eval_baseline import solve_lkh_windy, get_lkh_executable

# 1. Create a dummy instance
loc = np.random.rand(20, 2)
wind = np.array([1.0, 0.0]) # Strong wind to right
alpha = 10.0
name = "test_debug"
directory = "results/debug_lkh"
os.makedirs(directory, exist_ok=True)

print(f"Testing LKH on a random 20-node instance in '{directory}'...")
executable = get_lkh_executable()
print(f"Using executable: {executable}")

# 2. Call the solver function directly
result = solve_lkh_windy(executable, directory, name, loc, wind, alpha, runs=1)

# 3. Check output
if result:
    cost, tour, duration = result
    print("\n✅ SUCCESS!")
    print(f"Optimal Cost: {cost:.4f}")
    print(f"Duration:     {duration:.4f}s")
    print(f"Tour:         {tour}")
else:
    print("\n❌ FAILURE: LKH returned None. Check the logs in 'results/debug_lkh/test_debug.log'")