import torch

if torch.cuda.is_available():
    device = torch.device('cuda')          # CUDA 사용 가능한 경우
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"현재 GPU: {torch.cuda.current_device()}")
    print(f"GPU 개수: {torch.cuda.device_count()}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}") # 0번 GPU 이름 출력
else:
    device = torch.device('cpu')           # CUDA 사용 불가능한 경우
    print("CUDA 사용 불가능")