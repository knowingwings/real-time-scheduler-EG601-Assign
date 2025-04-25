"""
Platform Utilities

This module provides common platform detection and system information
gathering functions used across the project.
"""

import platform
import psutil
import os
import re
from typing import Dict, Any, Optional

def get_platform_info() -> Dict[str, Any]:
    """
    Get detailed information about the current platform
    
    Returns:
        Dictionary containing system information including:
        - system: Operating system name
        - node: Network hostname
        - release: Operating system release
        - version: Operating system version
        - machine: Hardware architecture
        - processor: Processor information
        - cpu_count: Number of physical CPUs
        - cpu_count_logical: Number of logical CPUs (including hyperthreading)
        - memory_total: Total physical memory in bytes
        - memory_available: Available physical memory in bytes
        - type: Categorized platform type (raspberry_pi_3, windows_desktop, etc.)
    """
    system_info = {
        'system': platform.system(),
        'node': platform.node(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available
    }
    
    # Determine platform type using logical rules
    if 'raspberry' in system_info['node'].lower():
        system_info['type'] = 'raspberry_pi_3'
    elif system_info['system'] == 'Darwin':
        system_info['type'] = 'macbook' if 'MacBook' in platform.node() else 'mac_desktop'
    elif system_info['system'] == 'Windows':
        system_info['type'] = 'windows_laptop' if hasattr(psutil, 'sensors_battery') and psutil.sensors_battery() else 'windows_desktop'
    elif system_info['system'] == 'Linux':
        system_info['type'] = 'linux_desktop'  # Default for Linux
    else:
        system_info['type'] = 'unknown'
    
    return system_info

def extract_platform_from_dir(data_dir: str) -> Optional[str]:
    """
    Extract platform type from a directory name
    
    Args:
        data_dir: Path to the data directory
    
    Returns:
        Extracted platform type or None if not found
        
    Example:
        extract_platform_from_dir("results/data/20230405_123456_raspberry_pi_3")
        returns "raspberry_pi_3"
    """
    # Extract the basename of the directory
    basename = os.path.basename(data_dir)
    
    # Try to extract platform using regex
    # Pattern: timestamp_platform_type
    match = re.search(r'\d{8}_\d{6}_([\w_]+)', basename)
    if match:
        return match.group(1)  # Return the platform type
    
    return None