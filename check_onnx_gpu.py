import onnxruntime as ort

providers = ort.get_available_providers()
print("Available ONNX Runtime providers:", providers)

# Check if CUDA (GPU) provider is available
if 'CUDAExecutionProvider' in providers:
    print("GPU is available for ONNX Runtime inference.")
else:
    print("GPU provider NOT available. Check CUDA/cuDNN and onnxruntime-gpu installation.")
