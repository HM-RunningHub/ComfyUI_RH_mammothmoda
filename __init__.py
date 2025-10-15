import os, sys

package_root = os.path.abspath(os.path.dirname(__file__))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

mammothmoda2_dir = os.path.join(package_root, "mammothmoda2")
if mammothmoda2_dir not in sys.path:
    sys.path.insert(0, mammothmoda2_dir)

from .rh_mammothmoda import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']