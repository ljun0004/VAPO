import nvidia_smi

_GPU = False
_NUMBER_OF_GPU = 0

def _check_gpu():
    global _GPU
    global _NUMBER_OF_GPU
    nvidia_smi.nvmlInit()
    _NUMBER_OF_GPU = nvidia_smi.nvmlDeviceGetCount()
    if _NUMBER_OF_GPU > 0:
        _GPU = True

def _get_gpu_usage():
    _check_gpu()
    if _GPU:
        INFO = ""
        for i in range(_NUMBER_OF_GPU):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            INFO += f"GPU{i}-Memory: {_bytes_to_megabytes(info.used)}/{_bytes_to_megabytes(info.total)} MB. "
    else:
        INFO = "No GPU found."
    return INFO

def _bytes_to_megabytes(bytes):
    return round((bytes/1024)/1024,2)

# if __name__ == '__main__':
#     print('Checking for Nvidia GPU\n')
#     _check_gpu()
#     if _GPU:
#         _print_gpu_usage()
#     else:
#         print("No GPU found.")