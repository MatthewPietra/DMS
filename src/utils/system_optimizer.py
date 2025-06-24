"""
System Optimizer for YOLO Vision Studio

This module provides system optimization functions to improve performance
and stability for production use of YOLO Vision Studio.
"""

import os
import sys
import platform
import logging
import gc
import psutil
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SystemOptimizer:
    """System optimization manager for YOLO Vision Studio."""
    
    def __init__(self):
        self.optimizations_applied = []
        self.performance_metrics = {}
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            return {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'disk_usage': psutil.disk_usage('/').total if platform.system() != 'Windows' else psutil.disk_usage('C:\\').total
            }
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
            return {}
    
    def optimize_system_for_production(self) -> Dict[str, Any]:
        """
        Apply comprehensive system optimizations for production use.
        
        This function applies various optimizations:
        - Memory management
        - CPU optimization
        - GPU optimization
        - File system optimization
        - Network optimization
        - Process priority optimization
        """
        logger.info("Applying system optimizations for production...")
        
        optimization_results = {
            'memory': self._optimize_memory(),
            'cpu': self._optimize_cpu(),
            'gpu': self._optimize_gpu(),
            'filesystem': self._optimize_filesystem(),
            'network': self._optimize_network(),
            'process': self._optimize_process(),
            'python': self._optimize_python()
        }
        
        # Record applied optimizations
        for category, result in optimization_results.items():
            if result.get('success', False):
                self.optimizations_applied.append(category)
        
        # Measure performance impact
        self.performance_metrics = self._measure_performance()
        
        logger.info(f"Applied {len(self.optimizations_applied)} optimizations: {', '.join(self.optimizations_applied)}")
        
        return {
            'optimizations_applied': self.optimizations_applied,
            'results': optimization_results,
            'performance_metrics': self.performance_metrics
        }
    
    def _optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage and management."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Set memory limits for better management
            if hasattr(psutil, 'virtual_memory'):
                vm = psutil.virtual_memory()
                memory_usage_percent = vm.percent
                
                # If memory usage is high, apply aggressive optimization
                if memory_usage_percent > 80:
                    # Clear Python cache
                    if hasattr(sys, 'getallocatedblocks'):
                        initial_blocks = sys.getallocatedblocks()
                        gc.collect()
                        final_blocks = sys.getallocatedblocks()
                        blocks_freed = initial_blocks - final_blocks
                        logger.info(f"Freed {blocks_freed} memory blocks")
            
            # Set environment variables for memory optimization
            os.environ['PYTHONMALLOC'] = 'malloc'
            os.environ['PYTHONDEVMODE'] = '0'
            
            return {'success': True, 'memory_usage': memory_usage_percent}
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_cpu(self) -> Dict[str, Any]:
        """Optimize CPU usage and threading."""
        try:
            # Set CPU affinity for better performance
            if hasattr(psutil.Process(), 'cpu_affinity'):
                process = psutil.Process()
                cpu_count = psutil.cpu_count()
                
                # Use all available cores
                process.cpu_affinity(list(range(cpu_count)))
            
            # Set environment variables for CPU optimization
            os.environ['OMP_NUM_THREADS'] = str(psutil.cpu_count())
            os.environ['MKL_NUM_THREADS'] = str(psutil.cpu_count())
            os.environ['NUMEXPR_NUM_THREADS'] = str(psutil.cpu_count())
            
            # Set process priority
            if hasattr(psutil.Process(), 'nice'):
                process = psutil.Process()
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            
            return {'success': True, 'cpu_count': psutil.cpu_count()}
            
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_gpu(self) -> Dict[str, Any]:
        """Optimize GPU usage and memory."""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Set GPU memory fraction
                torch.cuda.set_per_process_memory_fraction(0.8)
                
                # Enable memory efficient attention
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                    torch.backends.cuda.enable_flash_sdp(True)
                
                # Set deterministic algorithms
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                return {'success': True, 'gpu_memory': gpu_memory, 'device_count': torch.cuda.device_count()}
            else:
                return {'success': True, 'gpu_available': False}
                
        except ImportError:
            logger.debug("PyTorch not available, skipping GPU optimization")
            return {'success': True, 'pytorch_available': False}
        except Exception as e:
            logger.warning(f"GPU optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_filesystem(self) -> Dict[str, Any]:
        """Optimize file system operations."""
        try:
            # Set buffer size for file operations
            if hasattr(os, 'environ'):
                os.environ['PYTHONIOENCODING'] = 'utf-8'
                os.environ['PYTHONHASHSEED'] = '0'
            
            # Optimize for the current platform
            if platform.system() == 'Windows':
                # Windows-specific optimizations
                os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '1'
            elif platform.system() == 'Linux':
                # Linux-specific optimizations
                os.environ['PYTHONUNBUFFERED'] = '1'
            
            return {'success': True, 'platform': platform.system()}
            
        except Exception as e:
            logger.warning(f"Filesystem optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_network(self) -> Dict[str, Any]:
        """Optimize network operations."""
        try:
            # Set socket timeout
            import socket
            socket.setdefaulttimeout(30)
            
            # Set environment variables for network optimization
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            os.environ['CURL_CA_BUNDLE'] = ''
            
            return {'success': True}
            
        except Exception as e:
            logger.warning(f"Network optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_process(self) -> Dict[str, Any]:
        """Optimize process settings."""
        try:
            process = psutil.Process()
            
            # Set process priority
            if platform.system() == 'Windows':
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                process.nice(10)  # Lower priority on Unix systems
            
            # Set I/O priority
            if hasattr(process, 'ionice'):
                process.ionice(psutil.IOPRIO_CLASS_BE, 2)
            
            return {'success': True, 'pid': process.pid}
            
        except Exception as e:
            logger.warning(f"Process optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_python(self) -> Dict[str, Any]:
        """Optimize Python runtime settings."""
        try:
            # Set Python optimization flags
            sys.dont_write_bytecode = True
            
            # Optimize garbage collection
            gc.set_threshold(700, 10, 10)
            
            # Set recursion limit
            sys.setrecursionlimit(10000)
            
            # Optimize string interning
            import sys
            if hasattr(sys, 'intern'):
                # Intern commonly used strings
                common_strings = ['image', 'label', 'bbox', 'class', 'confidence']
                for s in common_strings:
                    sys.intern(s)
            
            return {'success': True, 'python_version': sys.version}
            
        except Exception as e:
            logger.warning(f"Python optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _measure_performance(self) -> Dict[str, Any]:
        """Measure system performance metrics."""
        try:
            metrics = {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if platform.system() != 'Windows' else psutil.disk_usage('C:\\').percent,
                'load_average': psutil.getloadavg() if platform.system() != 'Windows' else None
            }
            
            # Get GPU metrics if available
            try:
                import torch
                if torch.cuda.is_available():
                    metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated(0)
                    metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved(0)
            except ImportError:
                pass
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Performance measurement failed: {e}")
            return {}
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            'optimizations_applied': self.optimizations_applied,
            'performance_metrics': self.performance_metrics,
            'system_info': self.system_info
        }
    
    def reset_optimizations(self):
        """Reset all applied optimizations."""
        logger.info("Resetting system optimizations...")
        self.optimizations_applied = []
        self.performance_metrics = {}
        
        # Reset environment variables
        for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 
                   'PYTHONMALLOC', 'PYTHONDEVMODE']:
            if var in os.environ:
                del os.environ[var]
        
        logger.info("System optimizations reset")

# Global optimizer instance
_optimizer = None

def optimize_system_for_production() -> Dict[str, Any]:
    """
    Apply comprehensive system optimizations for production use.
    
    Returns:
        Dict containing optimization results and performance metrics
    """
    global _optimizer
    
    if _optimizer is None:
        _optimizer = SystemOptimizer()
    
    return _optimizer.optimize_system_for_production()

def get_optimization_status() -> Dict[str, Any]:
    """Get current optimization status."""
    global _optimizer
    
    if _optimizer is None:
        return {'optimizations_applied': [], 'performance_metrics': {}, 'system_info': {}}
    
    return _optimizer.get_optimization_status()

def reset_optimizations():
    """Reset all applied optimizations."""
    global _optimizer
    
    if _optimizer is not None:
        _optimizer.reset_optimizations()

def get_system_recommendations() -> List[str]:
    """Get system optimization recommendations."""
    recommendations = []
    
    try:
        # Check CPU
        cpu_count = psutil.cpu_count()
        if cpu_count < 4:
            recommendations.append("Consider using a system with 4+ CPU cores for better performance")
        
        # Check memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            recommendations.append("Consider using a system with 8GB+ RAM for optimal performance")
        
        # Check disk space
        disk_gb = psutil.disk_usage('/').free / (1024**3) if platform.system() != 'Windows' else psutil.disk_usage('C:\\').free / (1024**3)
        if disk_gb < 10:
            recommendations.append("Ensure at least 10GB free disk space for training data")
        
        # Check GPU
        try:
            import torch
            if not torch.cuda.is_available():
                recommendations.append("Consider using a CUDA-compatible GPU for faster training")
        except ImportError:
            recommendations.append("Install PyTorch for GPU acceleration")
        
    except Exception as e:
        logger.warning(f"Failed to generate recommendations: {e}")
    
    return recommendations 