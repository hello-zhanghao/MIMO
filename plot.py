import numpy as np
import matplotlib.pyplot as plt 
params = {
        'dataset_dir': 'H64Nt4K4.mat',  # 使用固定数据集
        'dataset_dir': None,  # 程序运行时生成数据集
        'mod_name': 'QAM4',  # 支持QAM4，QAM16
        'rho': 0,
        'Nt': 4,  # Number of transmit antennas，网络训练/测试
        'Nr': 16,  # Number of receive antennas，网络训练/测试
        'User': 2,  # Number of Users，网络训练/测试
        'batch_size': 5000,  # 样本大小， 网络训练/测试
        'SNR_dB_train': 10,  # 训练信噪比，网络训练
        'maxEpoch': 2000,  # 每轮仿真次数，网络训练
        'nRounds': 10,  # 仿真轮数，网络训练
        'SNR_dB_min_test': 3,  # Minimum SNR value in dB for simulation，网络测试
        'SNR_dB_max_test': 19,  # Maximum SNR value in dB for simulation，网络测试
        'SNR_step_test': 3,  # 仿真SNR间隔，网络测试
        'test_iterations': 20,  # 测试检测算法的迭代次数，一般保证最低误符号率的错误符号在1e2以上，网络测试
    }
[0.225705, 0.14783, 0.082085, 0.036595, 0.012390002, 0.0025200022]  
plt.semilogy(np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test']), ser1, '*-', label='noPrecode')
plt.semilogy(np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test']), ser2, '*-', label='Precode')
plt.grid(True, which='minor', linestyle='--')
plt.legend()
plt.savefig('x.png')