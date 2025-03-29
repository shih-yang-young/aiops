import torch

def get_device_info(print_info=True):
    """
    檢查是否有 CUDA GPU 可用，並印出設備與記憶體狀態資訊。
    回傳 torch.device 物件以供模型使用。
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if print_info:
        print(f"Using device: {device}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"Current GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Current GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    return device