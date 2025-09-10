#!/usr/bin/env python3
"""
GPU and Performance Optimization Module
Handles NVIDIA GPU detection, optimization, and performance tuning for ML workloads.
"""

import os
import sys
import subprocess
import logging
import psutil
import platform
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import GPU-related libraries
try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUOptimizer:
    """
    GPU optimization and performance tuning system for machine learning workloads.
    """
    
    def __init__(self):
        """Initialize the GPU optimizer."""
        self.gpu_info = {}
        self.optimization_settings = {}
        self.system_info = {}
        self.performance_mode = False
        
    def detect_system_configuration(self) -> Dict:
        """Detect system configuration and capabilities."""
        logger.info("🔍 Detecting system configuration...")
        
        system_info = {
            'platform': platform.system(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        }
        
        self.system_info = system_info
        return system_info
    
    def detect_gpu_hardware(self) -> Dict:
        """Detect NVIDIA GPU hardware and capabilities."""
        logger.info("🎮 Detecting GPU hardware...")
        
        gpu_info = {
            'nvidia_driver_available': False,
            'gpus': [],
            'cuda_available': False,
            'cuda_version': None,
            'total_gpu_memory_mb': 0
        }
        
        # Check NVIDIA driver
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_info['nvidia_driver_available'] = True
                logger.info("✅ NVIDIA driver detected")
            else:
                logger.warning("⚠️ NVIDIA driver not found")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("⚠️ nvidia-smi command not available")
        
        # Check NVIDIA ML library
        if NVML_AVAILABLE and gpu_info['nvidia_driver_available']:
            try:
                nvml.nvmlInit()
                device_count = nvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu = {
                        'index': i,
                        'name': name,
                        'memory_total_mb': memory_info.total // (1024 * 1024),
                        'memory_free_mb': memory_info.free // (1024 * 1024),
                        'memory_used_mb': memory_info.used // (1024 * 1024)
                    }
                    
                    gpu_info['gpus'].append(gpu)
                    gpu_info['total_gpu_memory_mb'] += gpu['memory_total_mb']
                
                logger.info(f"✅ Found {device_count} NVIDIA GPU(s)")
                
            except Exception as e:
                logger.warning(f"⚠️ Error accessing GPU info: {e}")
        
        # Check CUDA availability
        if TORCH_AVAILABLE:
            gpu_info['cuda_available'] = torch.cuda.is_available()
            if gpu_info['cuda_available']:
                gpu_info['cuda_version'] = torch.version.cuda
                logger.info(f"✅ CUDA {gpu_info['cuda_version']} available with PyTorch")
        
        if TF_AVAILABLE:
            tf_gpus = tf.config.list_physical_devices('GPU')
            if tf_gpus:
                logger.info(f"✅ TensorFlow detected {len(tf_gpus)} GPU(s)")
        
        self.gpu_info = gpu_info
        return gpu_info
    
    def optimize_pytorch_settings(self) -> Dict:
        """Optimize PyTorch settings for performance."""
        if not TORCH_AVAILABLE:
            return {'status': 'PyTorch not available'}
        
        logger.info("⚡ Optimizing PyTorch settings...")
        
        settings = {}
        
        # Enable CUDA if available
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
            
            settings['cuda_enabled'] = True
            settings['cudnn_benchmark'] = True
            settings['device_count'] = torch.cuda.device_count()
            
            # Set memory management
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            logger.info(f"✅ PyTorch CUDA optimization enabled for {settings['device_count']} devices")
        else:
            settings['cuda_enabled'] = False
            logger.info("ℹ️ PyTorch running on CPU")
        
        # Set number of threads for CPU operations
        cpu_threads = min(psutil.cpu_count(), 8)  # Cap at 8 for stability
        torch.set_num_threads(cpu_threads)
        settings['cpu_threads'] = cpu_threads
        
        return settings
    
    def optimize_tensorflow_settings(self) -> Dict:
        """Optimize TensorFlow settings for performance."""
        if not TF_AVAILABLE:
            return {'status': 'TensorFlow not available'}
        
        logger.info("⚡ Optimizing TensorFlow settings...")
        
        settings = {}
        
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                settings['gpu_memory_growth'] = True
                settings['gpu_count'] = len(gpus)
                logger.info(f"✅ TensorFlow GPU memory growth enabled for {len(gpus)} devices")
                
            except RuntimeError as e:
                logger.warning(f"⚠️ Error configuring TensorFlow GPU: {e}")
                settings['gpu_memory_growth'] = False
        else:
            settings['gpu_count'] = 0
            logger.info("ℹ️ TensorFlow running on CPU")
        
        # Set CPU parallelism
        tf.config.threading.set_inter_op_parallelism_threads(psutil.cpu_count())
        tf.config.threading.set_intra_op_parallelism_threads(psutil.cpu_count())
        
        settings['cpu_parallelism'] = psutil.cpu_count()
        
        return settings
    
    def install_nvidia_drivers(self) -> Dict:
        """Attempt to install or update NVIDIA drivers (Linux only)."""
        logger.info("🔧 Checking NVIDIA driver installation...")
        
        result = {
            'attempted': False,
            'success': False,
            'message': '',
            'commands_run': []
        }
        
        if platform.system() != 'Linux':
            result['message'] = "NVIDIA driver auto-installation only supported on Linux"
            return result
        
        try:
            # Check if running in container/restricted environment
            if os.path.exists('/.dockerenv') or os.getenv('CONTAINER'):
                result['message'] = "Running in container - driver installation not available"
                return result
            
            # Check current driver status
            try:
                subprocess.run(['nvidia-smi'], check=True, capture_output=True)
                result['message'] = "NVIDIA drivers already installed and working"
                result['success'] = True
                return result
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
            
            # Attempt driver installation
            result['attempted'] = True
            
            # Update package list
            cmd = ['sudo', 'apt', 'update']
            result['commands_run'].append(' '.join(cmd))
            subprocess.run(cmd, check=True, timeout=300)
            
            # Install NVIDIA driver
            cmd = ['sudo', 'apt', 'install', '-y', 'nvidia-driver-470', 'nvidia-cuda-toolkit']
            result['commands_run'].append(' '.join(cmd))
            subprocess.run(cmd, check=True, timeout=600)
            
            result['success'] = True
            result['message'] = "NVIDIA drivers installed successfully - reboot required"
            
        except subprocess.CalledProcessError as e:
            result['message'] = f"Driver installation failed: {e}"
        except subprocess.TimeoutExpired:
            result['message'] = "Driver installation timed out"
        except PermissionError:
            result['message'] = "Insufficient permissions for driver installation"
        except Exception as e:
            result['message'] = f"Unexpected error during driver installation: {e}"
        
        return result
    
    def optimize_system_performance(self) -> Dict:
        """Apply system-level performance optimizations."""
        logger.info("⚡ Applying system performance optimizations...")
        
        optimizations = {
            'applied': [],
            'failed': [],
            'warnings': []
        }
        
        # Set environment variables for performance
        performance_env = {
            'OMP_NUM_THREADS': str(psutil.cpu_count()),
            'MKL_NUM_THREADS': str(psutil.cpu_count()),
            'OPENBLAS_NUM_THREADS': str(psutil.cpu_count()),
            'CUDA_VISIBLE_DEVICES': '0',  # Use first GPU by default
            'TF_CPP_MIN_LOG_LEVEL': '2',  # Reduce TensorFlow logging
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128'  # Optimize PyTorch memory
        }
        
        for key, value in performance_env.items():
            os.environ[key] = value
            optimizations['applied'].append(f"Set {key}={value}")
        
        # Memory optimizations
        try:
            import gc
            gc.set_threshold(700, 10, 10)  # More aggressive garbage collection
            optimizations['applied'].append("Optimized garbage collection")
        except Exception as e:
            optimizations['failed'].append(f"GC optimization: {e}")
        
        # CPU affinity optimization (if available)
        try:
            current_process = psutil.Process()
            cpu_count = psutil.cpu_count(logical=False)
            if cpu_count > 1:
                # Use all physical cores
                current_process.cpu_affinity(list(range(cpu_count)))
                optimizations['applied'].append(f"Set CPU affinity to {cpu_count} cores")
        except Exception as e:
            optimizations['warnings'].append(f"CPU affinity: {e}")
        
        return optimizations
    
    def get_recommended_batch_sizes(self) -> Dict:
        """Get recommended batch sizes based on available GPU memory."""
        recommendations = {
            'small_model_batch_size': 32,
            'medium_model_batch_size': 16,
            'large_model_batch_size': 8,
            'max_sequence_length': 512
        }
        
        if self.gpu_info.get('total_gpu_memory_mb', 0) > 0:
            total_memory_gb = self.gpu_info['total_gpu_memory_mb'] / 1024
            
            if total_memory_gb >= 16:  # High-end GPU
                recommendations.update({
                    'small_model_batch_size': 128,
                    'medium_model_batch_size': 64,
                    'large_model_batch_size': 32,
                    'max_sequence_length': 1024
                })
            elif total_memory_gb >= 8:  # Mid-range GPU
                recommendations.update({
                    'small_model_batch_size': 64,
                    'medium_model_batch_size': 32,
                    'large_model_batch_size': 16,
                    'max_sequence_length': 768
                })
            elif total_memory_gb >= 4:  # Entry-level GPU
                recommendations.update({
                    'small_model_batch_size': 32,
                    'medium_model_batch_size': 16,
                    'large_model_batch_size': 8,
                    'max_sequence_length': 512
                })
        
        return recommendations
    
    def run_full_optimization(self) -> Dict:
        """Run complete GPU and performance optimization."""
        logger.info("🚀 Running full GPU and performance optimization...")
        
        results = {
            'timestamp': str(psutil.boot_time()),
            'system_info': self.detect_system_configuration(),
            'gpu_info': self.detect_gpu_hardware(),
            'pytorch_optimization': self.optimize_pytorch_settings(),
            'tensorflow_optimization': self.optimize_tensorflow_settings(),
            'system_optimization': self.optimize_system_performance(),
            'nvidia_driver_check': self.install_nvidia_drivers(),
            'recommended_batch_sizes': self.get_recommended_batch_sizes(),
            'optimization_complete': True
        }
        
        self.performance_mode = True
        
        # Print summary
        logger.info("📊 Optimization Summary:")
        logger.info(f"   💻 System: {results['system_info']['platform']} ({results['system_info']['cpu_count']} cores, {results['system_info']['memory_total_gb']:.1f}GB RAM)")
        logger.info(f"   🎮 GPUs: {len(results['gpu_info']['gpus'])} NVIDIA GPU(s), {results['gpu_info']['total_gpu_memory_mb']}MB total")
        logger.info(f"   ⚡ CUDA: {results['gpu_info']['cuda_available']}")
        logger.info(f"   🔧 PyTorch optimized: {results['pytorch_optimization'].get('cuda_enabled', False)}")
        logger.info(f"   🔧 TensorFlow optimized: {results['tensorflow_optimization'].get('gpu_count', 0) > 0}")
        
        return results
    
    def get_performance_monitor(self) -> Dict:
        """Get current system performance metrics."""
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        }
        
        # Add GPU metrics if available
        if NVML_AVAILABLE and self.gpu_info.get('nvidia_driver_available'):
            try:
                gpu_metrics = []
                for i, gpu in enumerate(self.gpu_info.get('gpus', [])):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    gpu_metrics.append({
                        'gpu_id': i,
                        'memory_used_percent': (memory_info.used / memory_info.total) * 100,
                        'gpu_utilization_percent': utilization.gpu,
                        'memory_utilization_percent': utilization.memory
                    })
                
                metrics['gpu_metrics'] = gpu_metrics
            except Exception as e:
                metrics['gpu_error'] = str(e)
        
        return metrics


def setup_gpu_optimization() -> Dict:
    """Main function to set up GPU optimization."""
    optimizer = GPUOptimizer()
    return optimizer.run_full_optimization()


if __name__ == "__main__":
    print("🚀 GPU Optimizer - Setting up performance optimization...")
    
    optimizer = GPUOptimizer()
    results = optimizer.run_full_optimization()
    
    print("\n📊 Final Results:")
    print(f"✅ System detected: {results['system_info']['platform']}")
    print(f"✅ GPUs found: {len(results['gpu_info']['gpus'])}")
    print(f"✅ CUDA available: {results['gpu_info']['cuda_available']}")
    print(f"✅ Optimization complete: {results['optimization_complete']}")
    
    if results['gpu_info']['cuda_available']:
        batch_sizes = results['recommended_batch_sizes']
        print(f"💡 Recommended batch sizes:")
        print(f"   Small models: {batch_sizes['small_model_batch_size']}")
        print(f"   Medium models: {batch_sizes['medium_model_batch_size']}")
        print(f"   Large models: {batch_sizes['large_model_batch_size']}")
    
    print("🎉 GPU optimization setup complete!")