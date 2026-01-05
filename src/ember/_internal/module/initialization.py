"""Helpers that enable natural ``__init__`` patterns for Equinox modules."""

import equinox as eqx
import jax


def patch_module_class() -> None:
    """Patch :class:`ember._internal.module.Module` for natural ``__init__`` use."""
    from ember._internal import module as ember_module

    # Save original Module class
    OriginalModule = ember_module.Module

    class PatchedModule(OriginalModule):
        """Patched Module class that allows natural ``__init__`` patterns."""

        def __init_subclass__(cls, **kwargs):
            """Hook into subclass creation to enable proper initialization."""
            super().__init_subclass__(**kwargs)

            # Wrap the __init__ method if it exists
            if hasattr(cls, "__init__"):
                original_init = cls.__init__

                def wrapped_init(self, *args, **kwargs):
                    # Temporarily enable attribute setting
                    temp_annotations = {}

                    # Create a temporary setattr that collects attributes
                    original_setattr = self.__class__.__setattr__
                    pending_attrs = {}

                    def collecting_setattr(obj, name, value):
                        if name.startswith("_"):
                            # Private attributes bypass collection
                            object.__setattr__(obj, name, value)
                        else:
                            pending_attrs[name] = value
                            # Infer type annotation
                            if isinstance(value, (jax.Array, jax.numpy.ndarray)):
                                temp_annotations[name] = jax.Array
                            else:
                                temp_annotations[name] = type(value)
                            # Ensure attribute is available during __init__ execution
                            object.__setattr__(obj, name, value)

                    # Temporarily replace setattr
                    self.__class__.__setattr__ = collecting_setattr

                    try:
                        # Call original init to collect attributes
                        original_init(self, *args, **kwargs)

                        # Add collected annotations to class
                        if temp_annotations:
                            if not hasattr(self.__class__, "__annotations__"):
                                self.__class__.__annotations__ = {}
                            self.__class__.__annotations__.update(temp_annotations)

                        # Restore setattr
                        self.__class__.__setattr__ = original_setattr

                        # Now set all collected attributes properly
                        for name, value in pending_attrs.items():
                            # Static fields should be registered on the class once so the
                            # Module metaclass can mark them static. Dynamic fields remain
                            # direct attributes.
                            if isinstance(value, (jax.Array, jax.numpy.ndarray)):
                                object.__setattr__(self, name, value)
                            else:
                                if name not in self.__class__.__dict__:
                                    setattr(
                                        self.__class__, name, eqx.field(static=True, default=value)
                                    )
                                object.__setattr__(self, name, value)

                    finally:
                        # Ensure setattr is restored
                        self.__class__.__setattr__ = original_setattr

                cls.__init__ = wrapped_init

    # Replace the Module class
    ember_module.Module = PatchedModule

    # Also update it in the module's __all__
    if hasattr(ember_module, "__all__") and "Module" in ember_module.__all__:
        # Force reload of the symbol
        ember_module.__dict__["Module"] = PatchedModule

    return PatchedModule


# Auto-patching disabled - explicit field annotations work better with equinox
# Users should declare fields at class level with type annotations.
# patch_module_class()
