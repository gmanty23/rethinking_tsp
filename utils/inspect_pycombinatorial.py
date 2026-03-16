import inspect
from pyCombinatorial import algorithm

def print_algorithm_signatures():
    print(f"{'Algorithm Name':<30} | {'Parameters (Default Values)'}")
    print("-" * 80)
    
    # Get all functions in the algorithm module
    for name, obj in inspect.getmembers(algorithm, inspect.isfunction):
        # Filter for typical algorithm names (ignoring internal utils like 'distance_calc')
        if name.startswith('_') or name in ['distance_calc', 'util']:
            continue
            
        try:
            sig = inspect.signature(obj)
            print(f"{name:<30} | {sig}")
        except ValueError:
            # Some compiled functions might not reveal signatures
            print(f"{name:<30} | [Signature not accessible]")

if __name__ == "__main__":
    print_algorithm_signatures()