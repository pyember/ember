import sys
import os

print("Python path:", sys.path)
print("Current directory:", os.getcwd())

try:
    import ember
    print("Imported ember successfully")
    print("Ember path:", ember.__file__)
except ImportError as e:
    print("Failed to import ember:", str(e))

try:
    from ember import core
    print("Imported ember.core successfully")
except ImportError as e:
    print("Failed to import ember.core:", str(e))

# Try alternative paths
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
print("After adding src to path:", sys.path)

try:
    import ember
    print("After path update, imported ember successfully")
    print("Ember path:", ember.__file__)
except ImportError as e:
    print("After path update, failed to import ember:", str(e))

try:
    from ember import core
    print("After path update, imported ember.core successfully")
except ImportError as e:
    print("After path update, failed to import ember.core:", str(e))