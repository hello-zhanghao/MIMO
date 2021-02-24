import numpy as np
import scipy.io as scio
from Dataset import genData
import time
dt = time.localtime()
ft = '%Y%m%d%H%M%S'
nt = time.strftime(ft, dt)

class genData_Mu(genData):

    def __init__(self,params):
        super(genData_Mu,self).__init__(params)
        self.User = params['User']

    def dataTest(self, number,snr,H):               #更改为复数方法
        if self.dataset_dir:
            data_train = scio.loadmat(self.dataset_dir+'\\test_data%d' % number)
            x = data_train['x']
            y = data_train['y']
            H = data_train['H']
            sigma2 = self.Nt / (np.power(10, snr/10) * self.Nr)
            noise = np.sqrt(sigma2/2)*(np.random.randn(x.shape[0], self.Nr)
                                       + 1j*np.random.randn(x.shape[0], self.Nr))
            y_noise = y + noise
            x_real = self.complex2real(x, matrix=False)
            H_real = self.complex2real(H, matrix=True)
            y_noise_real = self.complex2real(y_noise, matrix=False)
            sigma2 = np.tile(np.expand_dims(np.expand_dims(sigma2, axis=0), axis=1), [x.shape[0], 1])
        else:
            # s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size, 2 * self.Nt])
            # x_real = self.constellation[s]
            s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size, self.liushu])
            x_complex = self.constellation[s]
            x_real = np.concatenate([np.real(x_complex), np.imag(x_complex)], axis=1)
            if self.liushu == 2 and self.Nt == 4:
                P1 = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
            elif self.liushu == self.Nt:
                P1 = np.eye(self.liushu)
            else:
                print(ValueError('非满流数据仅支持流数为2， 天线为4'))
            P1 = P1[np.newaxis, :, :]
            P1 = np.repeat(P1, self.batch_size, axis=0)
            # print(P1.shape)
            # print(x_complex.shape)
            # P1x_complex = self.batch_matvec_mul(P1, x_complex)

            # H_real = np.random.randn(self.batch_size, 2 * self.Nr, 2 * self.Nt) * np.sqrt(0.5 / self.Nr)

            H_complex = H[:self.batch_size, :, :]
            HP1_complex = np.matmul(H, P1)
            HP1r = np.real(HP1_complex)
            HP1i = np.imag(HP1_complex)
            # Hr = np.random.randn(self.batch_size, self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
            # Hi = np.random.randn(self.batch_size, self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
            HP1_real = np.concatenate([np.concatenate([HP1r, -HP1i], axis=2), np.concatenate([HP1i, HP1r], axis=2)], axis=1)
            y_real = self.batch_matvec_mul(HP1_real, x_real)
            # power_rx = np.mean(np.sum(np.square(y_real), axis=1), axis=0)
            sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr)
            noise = np.sqrt(sigma2 / 2) * np.random.randn(self.batch_size, 2 * self.Nr)
            y_noise_real = y_real + noise
            # y_noise_real = y_real
            sigma2 = np.tile(np.expand_dims(np.expand_dims(sigma2, axis=0), axis=1), [self.batch_size, 1])
        return x_real, HP1_real, y_noise_real, sigma2

    def dataTrain(self, number, snr): # 更改为复数方法
        if self.dataset_dir:
            data_train = scio.loadmat(self.dataset_dir+'\\test_data%d' % number)
            x = data_train['x']
            y = data_train['y']
            H = data_train['H']
            sigma2 = self.Nt / (np.power(10, snr/10) * self.Nr)
            noise = np.sqrt(sigma2/2)*(np.random.randn(x.shape[0], self.Nr)
                                       + 1j*np.random.randn(x.shape[0], self.Nr))
            y_noise = y + noise
            x_real = self.complex2real(x, matrix=False)
            H_real = self.complex2real(H, matrix=True)
            y_noise_real = self.complex2real(y_noise, matrix=False)
            sigma2 = np.tile(np.expand_dims(np.expand_dims(sigma2, axis=0), axis=1), [x.shape[0], 1])
        else:
            # s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size, 2 * self.Nt])
            # x_real = self.constellation[s]
            s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size, self.liushu])
            x_complex = self.constellation[s]
            x_real = np.concatenate([np.real(x_complex), np.imag(x_complex)], axis=1)

            # H_real = np.random.randn(self.batch_size, 2 * self.Nr, 2 * self.Nt) * np.sqrt(0.5 / self.Nr)
            Hr = np.random.randn(self.batch_size, self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
            Hi = np.random.randn(self.batch_size, self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
            H_real = np.concatenate([np.concatenate([Hr, -Hi], axis=2), np.concatenate([Hi, Hr], axis=2)], axis=1)
            y_real = self.batch_matvec_mul(H_real, x_real)
            # power_rx = np.mean(np.sum(np.square(y_real), axis=1), axis=0)
            sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr)
            noise = np.sqrt(sigma2 / 2) * np.random.randn(self.batch_size, 2 * self.Nr)
            y_noise_real = y_real + noise
            sigma2 = np.tile(np.expand_dims(np.expand_dims(sigma2, axis=0), axis=1), [self.batch_size, 1])
        return x_real, H_real, y_noise_real, sigma2

    def dataTrain_mu(self, number, snr):  # 多用户上行MIMO数据集
        x_real_l = np.zeros((self.batch_size, 2 * self.liushu * self.User))
        H_real_l = np.zeros((self.batch_size, 2 * self.Nr, 2 * self.liushu * self.User))
        y_noise_real_l = np.zeros((self.batch_size, 2 * self.Nr))
        sigma2_l = np.zeros((self.batch_size, 1 * self.User))
        for i in range(self.User):
            x_real, H_real, y_noise_real, sigma2 = self.dataTrain(number, snr)
            x_real_l[:, i * 2 * self.liushu:(i + 1) * 2 * self.liushu] = x_real
            H_real_l[:, :, i * 2 * self.liushu:(i + 1) * 2 * self.liushu] = H_real
            y_noise_real_l += y_noise_real
            sigma2_l[:, i:i + 1] = sigma2[:, :]
            # print("MMSE_sigma2",sigma2_l)

        return x_real_l, H_real_l, y_noise_real_l, sigma2_l

    def dataTest_mu(self,number, snr, H):  # 暂时认为所有用户的信噪比一样

        x_real_l = np.zeros((self.batch_size,2*self.liushu*self.User))
        H_real_l = np.zeros((self.batch_size,2*self.Nr,2*self.liushu*self.User))
        y_noise_real_l = np.zeros((self.batch_size,2*self.Nr))
        sigma2_l = np.zeros((self.batch_size,1*self.User))
        for i in range(self.User):
            x_real, H_real, y_noise_real, sigma2 = self.dataTest(number, snr, H)
            x_real_l[:,i*2*self.liushu:(i+1)*2*self.liushu] = x_real
            H_real_l[:,:,i*2*self.liushu:(i+1)*2*self.liushu] = H_real
            y_noise_real_l += y_noise_real
            sigma2_l[:,i:i+1] = sigma2[:,:]
        # print("MMSE_sigma2",sigma2_l)

        return x_real_l, H_real_l, y_noise_real_l, sigma2_l


if __name__ == "__main__":
    params = {
        # 二选一
        # 'dataset_dir': r'D:\Nr8Nt8batch_size500mod_nameQAM_4',  # 使用固定数据集
        'dataset_dir': None,  # 程序运行时生成数据集

        # ************************程序运行之前先检查下面的参数*****************
        'isTest': True,
        'isTrain': True,
        'savetype': [0],
        # 仿真算法
        'simulation_algorithms': [
            'ZF-QRD',
            'ZF',
            'MMSE'
        ],
        # 仿真参数
        'mod_name': 'QAM_4',
          'constellation': np.array([0.7071+0.7071j, -0.7071+0.7071j,0.7071-0.7071j,-0.7071-0.7071j]),
        'Nt': 2,  # Number of transmit antennas
        'Nr': 32,  # Number of receive antennas
        'User': 8,  # Number of Users
        'batch_size': 500,
        # 不同网络的层数

    }
    print("参数设置：", params)
    # 数据生成对象
    gen_data = genData_Mu(params)

    temp = gen_data.dataTest_mu(0,3)

    pass
    print("end")

