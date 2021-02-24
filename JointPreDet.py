# -*-coding:UTF-8-*- ----  ## 使用utf8编码
# ------------------------------------ 程序info ---------------------------------
# MIMO预编码及接收联合优化
# 不使用onehot编码， 输入信息为k*Nt
# ------------------------------------------------------------------------------

# -------------------- 倒入标准包以及自定义函数 ------------------------------------
# print('python版本：', platform.python_version())  # 3.6.9
# print('tensorflow版本：', tf.__version__)         # 2.5.0-dev20210107
# print('numpy版本：', np.__version__)              # 1.19.2
# ------------------------------------------------------------------------------
import platform
import time 
import datetime
import os
import numpy as np
import tensorflow as tf
import scipy.io as scio

from Dataset_Mu import genData_Mu
import CommonCom as Comm 

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, V):
        super(MyCallback, self).__init__()
        self.V = V        
    def on_epoch_end(self, epoch, logs=None):
        V1 = tf.matmul(self.V[0], self.V[0], transpose_a = True)
        # print(np.mean(V1))
        

# ----------------------------- 联合预编码检测系统类 ------------------------------
# 接口参数：params
# ------------------------------------------------------------------------------
class JointPreDet(object):
    def __init__(self, params):
        """
        构造函数
        """
        # ------------------------------ 场景设置 -------------------------------
        self.Nr = params['Nr']          # 接收天线
        self.Nt = params['Nt']          # 发送天线
        self.User = params['User']      # 用户数目,（暂只支持每个用户的天线数目相同场景）
        self.constellation = params['constellation']   # 复数星座点
        self.constellation_real = Comm.extr_real(self.constellation)
        
        # ---------------------------- 训练设置 ---------------------------------
        self.nRounds = params['nRounds']                # Epoch, 循环次数
        self.maxEpoch = params['maxEpoch']              # 每个Epoch仿真的batch数 
        self.SNR_dB_train = params['SNR_dB_train']      # 训练信噪比
        self.learning_rate = 0.001                      # 学习率
        self.L = 20                                     # 网络层数
        M = len(self.constellation)                     # 调制阶数
        str_ = '/Nr{}Nt{}M{}K{}SNR{}'.format(self.Nr, self.Nt, self.User, M,
                                         self.SNR_dB_train)
        self.checkpoint_dir = params['checkpoint_dir'] + str_  # 模型保存路径
        
        # ---------------------------- 测试设置 ---------------------------------
        self.SNR_dBs = np.arange(params['SNR_dB_min_test'], 
                                 params['SNR_dB_max_test'], 
                                 params['SNR_step_test'])    # 测试信噪比
        self.test_iterations = params['test_iterations']
        
        # ---------------------------- 数据接口 ---------------------------------
        # 多用户信道矩阵
        self.H = tf.keras.Input(shape=[2 * self.Nr, 2 * self.Nt * self.User], 
                                dtype=tf.float32, 
                                name='channel-matrix')  
        # 基站接收信号
        self.y = tf.keras.Input(shape=[2 * self.Nr, 1], 
                                dtype=tf.float32, 
                                name='received-signal')  
        # 多个用户的发送信号
        self.x = tf.keras.Input(shape=[2 * self.Nt * self.User, 1],
                                dtype=tf.float32, 
                                name='transmit-signal')  
        # 多个用户的噪声方差
        self.noise_sigma2 = tf.keras.Input(shape=[self.User], 
                                           dtype=tf.float32, 
                                           name='noise')
        self.nodes = None  
        
    def _precoder(self, H):
        """
        预编码网络，对每一个信道矩阵用网络构造一个相应的预编码矩阵
        Input:
            H: shape=(batch_size, d0, d1), dtype=float32
        Output: 
            return: shape=(batch_size, d1, d1), dtype=float32
        """
        # ---------------------------- DNN结构 ----------------------------------
        layer = tf.keras.layers.Flatten()(H)  # 输入数据打平[B,d0*d1]
        for _ in range(self.L):
            layer = tf.keras.layers.Dense(units=128, activation='relu')(layer)
        V = tf.keras.layers.Dense(units=2*self.Nt*2*self.Nt)(layer)
        V = tf.keras.layers.Reshape([2*self.Nt, 2*self.Nt])(V)  # 预编码矩阵
        
        # 对预编码矩阵进行归一化
        norm2_V = tf.linalg.trace(tf.matmul(V, V, transpose_a = True))
        norm2_V = tf.expand_dims(tf.expand_dims(norm2_V, axis=1), axis=2) 
        norm2_V = tf.reduce_mean(norm2_V) 
        V = V * tf.sqrt(2*self.Nt/norm2_V)
        return V

        # ---------------------------- CNN结构 ----------------------------------
        # H = tf.transpose(H, [0, 2, 1]) 
        # layer = tf.expand_dims(H, axis=3)
        # for _ in range(self.L):
        #     layer = tf.keras.layers.Conv2D(
        #         16, 5, padding='same', activation='relu')(layer)
        # layer = tf.keras.layers.Conv2D(
        #     1, 5, padding='same', activation='relu')(layer)
        # layer = tf.keras.layers.Reshape([2*self.Nt, -1])(layer)
        # V = tf.keras.layers.Dense(units=2*self.Nt, activation=None)(layer)
        # # 对预编码矩阵进行归一化
        # norm2_V = tf.linalg.trace(tf.matmul(V, V, transpose_a = True))
        # norm2_V = tf.expand_dims(tf.expand_dims(norm2_V, axis=1), axis=2)  
        # V = V * tf.sqrt(2*self.Nt/norm2_V)
        # return V
        
    def _myModel(self):
        """
        建立端到端模型
        """
        # ----------------------------- 预编码矩阵 -------------------------------
        # 每个用户的预编码矩阵均通过一个输入为总信道矩阵的网络获得
        # V = [self._precoder(self.H[:, :, k*2*self.Nt:(k+1)*2*self.Nt]) 
        #      for k in range(self.User)] 
        V = [self._precoder(self.H) for k in range(self.User)] 
        self.V = V
        # V = [tf.eye(2*self.Nt) for k in range(self.User)]     # 不进行预编码
        # 预编码矩阵总功率受限
        # norm2_V_all = 0
        # for v in V:
        #     norm2_V_all += tf.reduce_mean(tf.linalg.trace(tf.matmul(V, V, transpose_a = True)))
        # V = [V[i]*tf.sqrt(2*self.Nt/norm2_V_all*self.User) for i in range(self.User)]
            
        # ----------------------------- 等效信道 --------------------------------
        H_eff = tf.concat([self.H[:, :, k*2*self.Nt:(k+1)*2*self.Nt]@V[k] 
                           for k in range(self.User)], axis=2)  
        y = H_eff @ self.x 
        noise_sigma2 = self.noise_sigma2[0, 0]/2 *self.User     # n=n1+n2+...
        noise = tf.sqrt(noise_sigma2) * tf.random.normal(shape=tf.shape(y))
        y_noise = y + noise 
        
        # ----------------------------- MMSE检测 --------------------------------
        HtH = tf.matmul(H_eff, H_eff, transpose_a=True)
        Hty = tf.matmul(H_eff, y_noise, transpose_a=True)
        sigma2_ = self.User*tf.reshape(self.noise_sigma2[:, 0]/2, [-1, 1, 1])
        I_ = tf.eye(2*self.Nt*self.User, batch_shape=[tf.shape(Hty)[0]])
        HtHinv = tf.linalg.inv(HtH + sigma2_*I_)
        xhat = tf.matmul(HtHinv, Hty)
        
        # ----------------------------- 网络检测 --------------------------------
        # decoder_input = tf.concat([tf.squeeze(y_noise, axis=-1), 
        # tf.reshape(H_eff, [-1, 2*self.Nr*self.Nt*self.User*2])], axis=1)
        # xhat = tf.keras.layers.Dense(128, 'relu')(decoder_input)
        # for _ in range(10):
        #     xhat  = tf.keras.layers.Dense(128, 'relu')(xhat)
        # xhat  = tf.keras.layers.Dense(2*self.Nt*self.User)(xhat)
        # xhat = tf.expand_dims(xhat, axis=2)
        
        # ------------------------ 建立模型 & 添加观测指标-------------------------
        model = tf.keras.models.Model(inputs=[self.H, self.noise_sigma2, self.x], 
                                      outputs=xhat)
        model.summary()
        ser = self._ser_tf(xhat, self.x)
        v1 = tf.matmul(V[0], V[0], transpose_a = True)
        y_norm2 = tf.linalg.trace(tf.matmul(y, y, transpose_a=True))
        noise_norm2 = tf.linalg.trace(tf.matmul(noise, noise, transpose_a=True))
        model.add_metric(ser, name='ser')
        model.add_metric(tf.reduce_mean(v1), name='precoder_norm')
        model.add_metric(tf.reduce_mean(y_norm2), name='receive_energy')
        model.add_metric(tf.reduce_mean(noise_norm2), name='noise_energy')
        model.add_metric(10*tf.math.log(
            tf.reduce_mean(y_norm2) / tf.reduce_mean(noise_norm2))
                         / tf.math.log(10.), name='SNR' )
        model.compile(optimizer=
                      tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.mean_squared_error,
                      metrics=[])
        return model 

    def build(self):
        self.model = self._myModel()
        return self 

    def _dataGenerator(self, Data):
        while True:
            x_Feed, H_Feed, y_Feed, noise_sigma2 = Data.dataTrain_mu(
                1, self.SNR_dB_train)
            x_Feed = np.expand_dims(x_Feed, axis=2)
            yield [H_Feed, noise_sigma2, x_Feed], x_Feed

    def _ser_tf(self, xhat, x):
        """
        计算估计信号xhat和x的误符号率, tf操作
        Input:
            xhat: 估计信号，shape = [B, 2*Nt*User, 1]
            x:    真实信号，shape = [B, 2*Nt*User, 1]
        Output:
            ser: float32
        """
        # 去掉最后一个维度（如果最后一个维度为1）[B, 2*Nt*User]
        xhat = tf.squeeze(xhat, -1) 
        # 去掉最后一个维度（如果最后一个维度为1）[B, 2*Nt*User]  
        x = tf.squeeze(x, -1)        
        # 实数信号等效转换为复数信号， [B, Nt*User]
        xhat_complex = Comm.x_convert_to_complex(xhat, self.User, self.Nt) 
        # 实数信号等效转换为复数信号，[B, Nt*User]
        x_complex = Comm.x_convert_to_complex(x, self.User, self.Nt)     
        # 解调为对应的数据，[B, Nt*User]   
        xhat_idx = Comm.demodulate2(xhat_complex, self.constellation)  
        # 解调为对应的数据，[B, Nt*User]     
        x_idx = Comm.demodulate2(x_complex, self.constellation)         
        # 计算误符号率    
        ser = 1 - tf.reduce_mean(tf.cast(tf.equal(xhat_idx, x_idx), tf.float32))  
        return ser
    
    def _callbacks(self):
        """
        定义需要的回调函数
        """
        callbacks = [
            # 提前终止
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=0.0,
                patience=2,
                verbose=1,
            ),
            # 模型保存
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_dir+'/epoch={epoch:.0f}loss={loss:.6f}',
            ),
            # 可视化记录
            tf.keras.callbacks.TensorBoard(
                log_dir=self.checkpoint_dir,
                update_freq=10,
            )
        ]
        return callbacks

    def train(self, Data):
        """
        训练模型
        """
        callback_list = self._callbacks()  # 回调函数，保存模型，提前终止
        callback_list = [MyCallback(self.V)]
        self.model.fit(self._dataGenerator(Data), 
                       epochs=self.nRounds, 
                       steps_per_epoch=self.maxEpoch, 
                       callbacks=callback_list)

    def _eval_spec_snr(self, Data, snr):
        """
        测试特定信噪比下的误符号率
        """
        ser = []
        for _ in range(self.test_iterations):
            x_Feed, H_Feed, y_Feed, noise_sigma2 = Data.dataTrain_mu(
                1, snr)
            x_Feed = np.expand_dims(x_Feed, axis=2)
            xhat = self.model.predict([H_Feed, noise_sigma2, x_Feed])
            xhat = tf.constant(xhat, dtype=tf.float32)
            x_Feed = tf.constant(x_Feed, dtype=tf.float32)
            ser.append(self._ser_tf(xhat, x_Feed).numpy())
        return np.mean(ser) 
        
    def eval(self, Data):
        """
        评估模型
        """
        ser_all = [self._eval_spec_snr(Data, snr) for snr in self.SNR_dBs]
        print(ser_all)
        return ser_all

    def validate(self, Data):
        self.model.evaluate(self._dataGenerator(Data), max_queue_size=1)
    


# ------------------------------------- main -----------------------------------
# 程序入口
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    try:  # 命令行输入
        args = Comm.parse()
        params = {
            'dataset_dir': args.data_dir,  # 数据集目录
            'mod_name': args.modulation,  # 支持QAM4，QAM16
            'rho': args.rho,
            'Nt': args.Nt,  # Number of transmit antennas，网络训练/测试
            'Nr': args.Nr,  # Number of receive antennas，网络训练/测试
            'User': args.K,  # Number of Users，网络训练/测试
            'batch_size': args.batch_size,  # 样本大小， 网络训练/测试
            'SNR_dB_train': args.SNR_dB_train,  # 训练信噪比，网络训练
            'maxEpoch': args.maxEpoch,  # 每轮仿真次数，网络训练
            'nRounds': args.nRounds,  # 仿真轮数，网络训练
            'SNR_dB_min_test': args.SNR_dB_min_test,  # Minimum SNR value in dB for simulation，网络测试
            'SNR_dB_max_test': args.SNR_dB_max_test,  # Maximum SNR value in dB for simulation，网络测试
            'SNR_step_test': args.SNR_step_test,  # 仿真SNR间隔，网络测试
            'test_iterations': args.iterations,  # 测试检测算法的迭代次数，一般保证最低误符号率的错误符号在1e2以上，网络测试
            'constellation':Comm.load_constellation(args.modulation),
            'checkpoint_dir': 'DeepSIC_model',
            'rho': args.rho,
        }
    except:  # 直接运行
        params = {
            # ---------------------- 场景设置参数 --------------------------------
            'dataset_dir': None,  # 数据集目录
            'mod_name': 'QAM4',  # 支持QAM4，QAM16
            'rho': 0,
            'Nt': 4,  # Number of transmit antennas，网络训练/测试
            'Nr': 16,  # Number of receive antennas，网络训练/测试
            'User': 2,  # Number of Users，网络训练/测试
            
            # ---------------------- 训练及测试设置参数 ---------------------------
            'batch_size': 500,      # 样本大小， 网络训练/测试
            'SNR_dB_train': 10,     # 训练信噪比，网络训练
            'maxEpoch': 100,       # 每轮仿真次数，网络训练
            'nRounds': 20,         # 仿真轮数，网络训练
            'SNR_dB_min_test': 0,   # 最小测试信噪比
            'SNR_dB_max_test': 11,  # 最大测试信噪比
            'SNR_step_test': 2,     # 信噪比测试间隔
            'test_iterations': 50,  # 迭代次数，一般保证最低误符号率的错误符号在1e2以上
            'checkpoint_dir': 'test',
        }
        # --------------------------- 获取星座点 --------------------------------
        params['constellation'] = Comm.load_constellation(params['mod_name'])
        Data = genData_Mu(params)           # 数据集生成对象
        
    # ----------------------- 模型建立，训练及测试 --------------------------------
    Comm.log2txt('text.log')
    print(params)
    Comm.gpu_seting(0)                  # GPU索引号， 选择对应GPU
    model = JointPreDet(params).build() # 建立网络图模型
    model.train(Data)                   # 训练网络
    np.random.seed(173)               # 固定测试集种子
    model.eval(Data)                    # 测试网络
    # model.validate(Data)                # 验证集
    print(params)



