import torch
print(torch.__version__)                   # versión
print("GPU disponible?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)