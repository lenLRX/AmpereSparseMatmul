import numpy as np

if __name__ == '__main__':
    golden = np.fromfile('d.bin', dtype='float16')
    gpu_result = np.fromfile('d_gpu.bin', dtype='float16')
    print(golden.reshape(16, 8))
    print(gpu_result.reshape(16, 8))
    diff = np.abs(golden - gpu_result).mean()
    print('diff: {}'.format(diff))