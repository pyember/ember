"""
Simple check to verify Ember is installed correctly.
"""

import ember

def main():
    print(f"Ember version: {ember.__version__}")
    print("Ember is installed correctly.")
    
    # Check if we can import key modules
    from ember.core.registry.operator.base import Operator
    from ember.xcs.tracer.tracer_decorator import jit
    
    print("Core modules imported successfully.")
    print("Basic functionality works!")
    
if __name__ == "__main__":
    main()