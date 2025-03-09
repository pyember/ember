"""Test basic functionality of XCS API facade."""

import unittest
import sys
import types
from typing import List, Dict, Any

class TestXCSBasic(unittest.TestCase):
    """Test basic functionality of the XCS API facade."""
    
    def test_module_imports(self):
        """Test that the xcs module can be imported and has expected attributes."""
        # Import the module directly
        import sys
        import importlib.util
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.absolute()
        xcs_path = project_root / "src" / "ember" / "xcs.py"
        
        spec = importlib.util.spec_from_file_location("ember.xcs", xcs_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Test that the expected attributes are present
        self.assertTrue(hasattr(module, 'jit'))
        self.assertTrue(hasattr(module, 'vmap'))
        self.assertTrue(hasattr(module, 'pmap'))
        self.assertTrue(hasattr(module, 'autograph'))
        self.assertTrue(hasattr(module, 'execute'))
        self.assertTrue(hasattr(module, 'mesh_sharded'))
        self.assertTrue(hasattr(module, 'XCSGraph'))
        self.assertTrue(hasattr(module, 'ExecutionOptions'))
        
        # Test function types
        self.assertTrue(callable(module.jit))
        self.assertTrue(callable(module.vmap))
        self.assertTrue(callable(module.pmap))
        self.assertTrue(callable(module.execute))
        
        # Test __all__ is defined correctly
        expected_exports = [
            'autograph', 'jit', 'vmap', 'pmap', 'mesh_sharded',
            'execute', 'XCSGraph', 'ExecutionOptions'
        ]
        for export in expected_exports:
            self.assertIn(export, module.__all__)
            
        # Store the module for other tests
        self.xcs = module
    
    def test_jit_decorator(self):
        """Test that the jit decorator works."""
        self.test_module_imports()  # Ensure module is loaded
        
        @self.xcs.jit
        def add(a: int, b: int) -> int:
            return a + b
        
        result = add(1, 2)
        self.assertEqual(result, 3)
    
    def test_vmap_function(self):
        """Test that the vmap function works."""
        self.test_module_imports()  # Ensure module is loaded
        
        def square(x: int) -> int:
            return x * x
        
        squared = self.xcs.vmap(square)
        results = squared([1, 2, 3, 4])
        self.assertEqual(results, [1, 4, 9, 16])
    
    def test_autograph_context(self):
        """Test that the autograph context manager works."""
        self.test_module_imports()  # Ensure module is loaded
        
        def add_one(x: int) -> int:
            return x + 1
        
        with self.xcs.autograph() as graph:
            result = add_one(5)
            # In a real implementation, this would record the operation
            # instead of executing it immediately
        
        # In the stub implementation, execute just returns an empty dict
        results = self.xcs.execute(graph)
        self.assertEqual(results, {})
        
if __name__ == "__main__":
    unittest.main()