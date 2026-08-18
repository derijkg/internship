import os
import sys
import glob
import ctypes

def setup_cuda_libraries():
    """
    Preloads all PyPI NVIDIA CUDA shared libraries (.so) globally into process memory.
    Eliminates the need to ever set LD_LIBRARY_PATH in the terminal.
    """
    site_packages = [p for p in sys.path if "site-packages" in p]
    for sp in site_packages:
        nvidia_dir = os.path.join(sp, "nvidia")
        if os.path.exists(nvidia_dir):
            for pkg in os.listdir(nvidia_dir):
                lib_dir = os.path.join(nvidia_dir, pkg, "lib")
                if os.path.isdir(lib_dir):
                    # 1. Update environment LD_LIBRARY_PATH
                    os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                    # 2. Preload all .so files globally into process memory
                    for libfile in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
                        try:
                            ctypes.CDLL(libfile, mode=ctypes.RTLD_GLOBAL)
                        except Exception:
                            pass

# MUST execute before importing llama_cpp
setup_cuda_libraries()

# Now import llama_cpp safely
from llama_cpp import Llama

print("Successfully loaded llama_cpp with CUDA support!")