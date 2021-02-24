import numpy as np 
import CommonCom as Comm 
import tensorflow as tf 
import matplotlib.pyplot as plt 


class Transmitter(object):
    def __init__(self, params):
        self.Nr = params['Nr']          # 接收天线
        self.Nt = params['Nt']          # 发送天线
        self.User = params['User']      # 用户数目，（暂只支持每个用户的天线数目相同场景）
        self.constellation = params['constellation']   # 复数星座点
        self.nRounds = params['nRounds']               # Epoch
        self.maxEpoch = params['maxEpoch']             # 每个Epoch仿真的batch数
        self.SNR_dBs = np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])  # 测试信噪比
        self.batch_size = params['batch_size']

        self.H = tf.keras.Input(shape=[2 * self.Nr, 2 * self.Nt * self.User], dtype=tf.float32, name='channel-matrix')  # 多用户信道矩阵
        self.x = tf.keras.Input(shape=[2 * self.Nt * self.User], dtype=tf.float32, name='transmit-signal')  # 多个用户的发送信号
        self.noise_sigma2 = tf.keras.Input(shape=[self.User], dtype=tf.float32, name='noise')   # 多个用户的噪声方差
        self.nodes = None
        
    def _random_stream(self):
        """
        生成随机数据流，shape=[B*Nt, Nt, 1]
        """
        M = len(self.constellation)  # 调制阶数
        k = int(np.log2(M))          # 每个调制符号对应的bit位数
        s = np.random.randint(low=0, high=2, size=[self.batch_size*k, self.Nt, 1])  
        return s
    
    def _bit_encoded(self, s):
        return s 
    
    def _bit_mapping(self, s):
        """
        编码后数据进行比特映射
        """
        M = len(self.constellation)  # 调制阶数
        k = int(np.log2(M))          # 每个调制符号对应的bit位数
        data = np.ones([self.batch_size, self.Nt, 1])
        for i in range(self.batch_size):
            for j in range(self.Nt):
                bits = s[i*k:(i+1)*k, j, 0]
                data[i][j][0] = sum([bits[i]*pow(2, len(bits)-1-i) for i in range(len(bits))])
        return data 
    
    def Num2Bit(Num, B):
        """
        """
        pass 
    
    def _modulation(self, data):
        """
        调制
        data为bit映射后数据， shape=[B, Nt, 1]
        """
        data.astype(np.int32)
        data_shape = data.shape
        signal = np.ones(data_shape, dtype=np.complex64)
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                for k in range(data_shape[2]):
                    signal[i, j, k] = self.constellation[int(data[i, j, k])]
        return signal 
    
    def _precode(self, signal, H, sigma2):
        """
        发送信号预编码
        signal: 调制后信号， shape=[B, Nt, 1], dtype=np.complex64
        """
        H_tilde = np.concatenate(H, axis=2)
        u, s, vh = np.linalg.svd(H_tilde)
        Lambda_f = np.zeros([self.batch_size, self.Nt*self.User, self.Nt*self.User])
        
        for i in range(self.batch_size):
            lambda_h = s[i]  # shape=[self.Nt*self.User, ]
            index = list(np.arange(0, self.Nt*self.User))
            x = np.zeros(self.Nt*self.User)
            p = 0
            while p<self.Nt*self.User:
                sqrtsum = np.sum(1/lambda_h[index])
                rsum = np.sum(sigma2/np.square(lambda_h[index])) + self.Nt*self.User
                sqrtmu = sqrtsum / rsum 
                x[index] = 1/sqrtmu/lambda_h[index] - sigma2/np.square(lambda_h[index])
                if min(x) < 0:
                    index_min = np.where(x == min(x))[0][0]
                    index_0 = index.index(index_min)
                    index = index[:index_0] + index[index_0+1:]
                    p = p+1
                    x[:] = 0
                else:
                    p = self.Nt*self.User
            for j, k in enumerate(index):
                Lambda_f[i, k, k] = x[j] 
                    
        """           
        # for i in range(self.batch_size):
        #     flag = False  # False不存在小于0的功率分配
        #     sqrtsum = np.sum(1/s[i])
        #     rsum = np.sum(sigma2/np.square(s[i])) + self.Nt*self.User
        #     sqrtmu = sqrtsum / rsum 
        #     for j in range(self.Nt*self.User):
        #         Lambda_f[i, j, j] = 1/sqrtmu/s[i][j] - sigma2/np.power(s[i][j], 2)
        #         if Lambda_f[i, j, j] < 0:
        #             flag = True 
        #     ignore = 0
        #     while flag:
        #         flag = False 
        #         sqrtsum = sqrtsum - 1/s[i][-1-ignore]
        #         rsum = rsum - 1 - sigma2/np.square(s[i][-1-ignore])
        #         sqrtmu = sqrtsum/ rsum
        #         for j in range(self.Nt*self.User):
        #             Lambda_f[i, j, j] = 1/sqrtmu/s[i][j] - sigma2/np.power(s[i][j], 2)
        #             if Lambda_f[i, j, j] < 0:
        #                 Lambda_f[i, j, j] = 0
        #                 if j < self.User*self.Nt - ignore - 1:
        #                     flag = True 
        #         ignore = ignore + 1
        """
        
        print(vh.shape, Lambda_f.shape, vh.conj().shape)
        F_tidle_star = np.matmul(np.transpose(vh.conj(), axes=[0, 2, 1]), Lambda_f) @ vh 
        print('松弛预编码矩阵功率', np.mean(np.trace(F_tidle_star, axis1=1, axis2=2)), np.mean(np.trace(Lambda_f, axis1=1, axis2=2)))
        
        # 计算Fi, A = BB^H , 怎么从A得到B
        FFh = []  
        for i in range(self.User):
            FFh.append(F_tidle_star[:, i*self.Nt:(i+1)*self.Nt, i*self.Nt:(i+1)*self.Nt])
            
        F = []
        for i, val in enumerate(FFh):
            F.append(np.linalg.cholesky(val))
            
        if not isPrecode: 
            F = [np.tile(np.expand_dims(np.eye(self.Nt), 0), reps=[self.batch_size, 1, 1]) for _ in range(self.User)]  # 不做预编码
        tx = [F@x for F, x in zip(F, signal)]
        return tx, F
        
        
    def tx(self, H, snr_dB):
        sigma2 = self.Nt / (np.power(10, snr_dB / 10) * self.Nr) * self.User # 噪声功率
        bit_uncoded = [self._random_stream()    for _ in range(self.User)]  # 生成随机数据流, list，list[i]为第i个用户的数据流，shape=[B*k, Nt, 1]
        bit_coded   = [self._bit_encoded(item)  for item in bit_uncoded]    # 二进制编码
        data        = [self._bit_mapping(item)  for item in bit_coded]      # bit映射
        mod_signal  = [self._modulation(item)   for item in data]           # 星座调制
        tx, F = self._precode(mod_signal, H, sigma2)                        # 发送信号
        # 计算发送信号功率
        power_signal = [np.mean(np.square(np.abs(signal))) for i, signal in enumerate(mod_signal)]
        print('每个用户调制后信号功率', power_signal)
        power_tx = [np.mean(np.square(np.abs(signal))) for i, signal in enumerate(tx)]
        print('每个用户发送信号功率', power_tx)
        return tx, F, data 

class Channel():
    def __init__(self, params):
        self.Nr = params['Nr']          # 接收天线
        self.Nt = params['Nt']          # 发送天线
        self.User = params['User']      # 用户数目，（暂只支持每个用户的天线数目相同场景）
        self.SNR_dBs = np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])  # 测试信噪比
        self.batch_size = params['batch_size']
    
    def channel_data(self):
        H = [(np.random.randn(self.batch_size, self.Nr, self.Nt)+1j*np.random.randn(self.batch_size, self.Nr, self.Nt))*np.sqrt(0.5/self.Nr)
                for _ in range(self.User)]
        return H 
    
    def channel(self, H, tx, snr_db):
        rx = 0
        for H_, tx_ in zip(H, tx):
            sigma2 = self.Nt / (np.power(10, snr_db / 10) * self.Nr)  # 噪声功率
            # sigma2 = 10 / self.User 
            noise = np.sqrt(sigma2 / 2) * (np.random.randn(self.batch_size, self.Nr, 1)+1j*np.random.randn(self.batch_size, self.Nr, 1))
            rx = rx + H_ @ tx_ + noise 
        return rx, sigma2
    
    def data_train(self):
        H = [(np.random.randn(self.batch_size, self.Nr, self.Nt)+1j*np.random.randn(self.batch_size, self.Nr, self.Nt))*np.sqrt(0.5/self.Nr)
                for _ in range(self.User)]
        return H 
            
class Receiver():
    def __init__(self, params):
        self.Nr = params['Nr']          # 接收天线
        self.Nt = params['Nt']          # 发送天线
        self.User = params['User']      # 用户数目，（暂只支持每个用户的天线数目相同场景）
        self.constellation = params['constellation']   # 复数星座点
        self.nRounds = params['nRounds']               # Epoch
        self.maxEpoch = params['maxEpoch']             # 每个Epoch仿真的batch数
        self.SNR_dBs = np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])  # 测试信噪比
        self.batch_size = params['batch_size']

        self.H = tf.keras.Input(shape=[2 * self.Nr, 2 * self.Nt * self.User], dtype=tf.float32, name='channel-matrix')  # 多用户信道矩阵
        self.x = tf.keras.Input(shape=[2 * self.Nt * self.User], dtype=tf.float32, name='transmit-signal')  # 多个用户的发送信号
        self.noise_sigma2 = tf.keras.Input(shape=[self.User], dtype=tf.float32, name='noise')   # 多个用户的噪声方差
        self.nodes = None
    
    def _MMSE(self, rx, H, F, sigma2):
        H_eff = [H_ @ F_ for H_, F_ in zip(H, F)]
        H_eff_tilde = np.concatenate(H_eff, axis=2)  # 多用户等效信道矩阵
        HhHinv = np.linalg.inv((np.transpose(H_eff_tilde.conj(), [0,2,1]) @ H_eff_tilde) + np.expand_dims(self.User*sigma2*np.eye(self.User*self.Nt), axis=0))  
        Hhy = np.transpose(H_eff_tilde.conj(), axes=[0, 2, 1]) @ rx 
        signal_est = HhHinv @ Hhy
        x_est = [signal_est[:, i*self.Nt:(i+1)*self.Nt, :] for i in range(self.User)] 
        return x_est 
    
    def estimate(self, rx, H, F, sigma2):
        return self._MMSE(rx, H, F, sigma2)
        
        
def cal_ser(x_est, data, constellation):
    ser = []
    for x_est_, data_ in zip(x_est, data):
        data_est = Comm.demodulate_np(x_est_, constellation)
        ser.append(np.mean(data_est != data_))
    print(ser, np.mean(ser)) 
    return np.mean(ser)
        
if __name__ == '__main__':
    
    # np.random.seed(173)
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
        'SNR_dB_min_test': 0,  # Minimum SNR value in dB for simulation，网络测试
        'SNR_dB_max_test': 11,  # Maximum SNR value in dB for simulation，网络测试
        'SNR_step_test': 2,  # 仿真SNR间隔，网络测试
        'test_iterations': 20,  # 测试检测算法的迭代次数，一般保证最低误符号率的错误符号在1e2以上，网络测试
    }
    params['constellation'] = Comm.load_constellation(params['mod_name'])
    isPrecode = False
    ser1 = []
    np.random.seed(173)
    for snr_dB in range(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test']):
        myTransmitter = Transmitter(params)             # 发送机
        myChannel = Channel(params)                     # 信道对象
        myReceiver = Receiver(params)                   # 接收机
        H = myChannel.channel_data()                    # 信道矩阵
        tx, F, data = myTransmitter.tx(H, snr_dB)           # 多用户发送信号
        rx, sigma2 = myChannel.channel(H, tx, snr_dB)       # 过信道
        x_est = myReceiver.estimate(rx, H, F, sigma2)   # 估计信号
        SER = cal_ser(x_est, data, params['constellation'])
        ser1.append(SER)
        print(ser1)
    isPrecode = True 
    ser2 = []
    np.random.seed(173)
    for snr_dB in range(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test']):
        myTransmitter = Transmitter(params)             # 发送机
        myChannel = Channel(params)                     # 信道对象
        myReceiver = Receiver(params)                   # 接收机
        H = myChannel.channel_data()                    # 信道矩阵
        tx, F, data = myTransmitter.tx(H, snr_dB)           # 多用户发送信号
        rx, sigma2 = myChannel.channel(H, tx, snr_dB)       # 过信道
        x_est = myReceiver.estimate(rx, H, F, sigma2)   # 估计信号
        SER = cal_ser(x_est, data, params['constellation'])
        ser2.append(SER)
        print(ser2)
    ser3 = [0.225705, 0.14783, 0.082085, 0.036595, 0.012390002, 0.0025200022]  
    ser4 = [0.22366, 0.1482, 0.081635, 0.035435, 0.011764999, 0.0027800011]
    plt.semilogy(np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test']), ser1, '*-', label='noPrecode')
    plt.semilogy(np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test']), ser2, '*-', label='Precode')
    plt.semilogy(np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test']), ser3, '*-', label='NetPrecode_all')
    plt.semilogy(np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test']), ser4, '*-', label='NetPrecode_alone')
    plt.grid(True, which='minor', linestyle='--')
    plt.xlabel('SNR')
    plt.ylabel('SER')
    plt.title('NR{}NT{}K{}{}'.format(params['Nr'], params['Nt'], params['User'], params['mod_name']))
    plt.legend()
    plt.savefig('x.png')
    
    
        