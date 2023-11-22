import cv2

def check_cuda_support():
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()

    if cuda_devices > 0:
        print(f"CUDA is supported on {cuda_devices} device(s).")
        for i in range(cuda_devices):
            print(f"Device {i}:")
            cv2.cuda.printCudaDeviceInfo(i)
    else:
        print("CUDA is not supported on this device.")

if __name__ == "__main__":
    check_cuda_support()
