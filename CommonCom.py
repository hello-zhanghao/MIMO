import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
import sys 
import logging


# ********************************** 系统设置 ***************************************************
def gpu_seting(i):
    # 设置使用的GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[i], True)
    print(gpus)
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[i], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[i], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'logical GPU')
        except RuntimeError as e:
            print(e)

def batch_matvec_mul_tf(A, b, transpose_a=False):
    """ 通用函数，与类无关
    矩阵A与矩阵b相乘，其中A.shape=(batch_size, Nr, Nt)
    b.shape = (batch_size, Nt)
    输出矩阵C，C.shape = (batch_size, Nr)
    """
    C = tf.matmul(A, tf.expand_dims(b, axis=2), transpose_a=transpose_a)
    return tf.squeeze(C, -1)

    
def batch_matvec_mul2(A, b, transpose_a=False):
    """ 通用函数，与类无关
    矩阵A与矩阵b相乘，其中A.shape=(batch_size, Nr, Nt)
    b.shape = (batch_size, Nt)
    输出矩阵C，C.shape = (batch_size, Nr)
    """
    C = tf.matmul(A, tf.expand_dims(b, axis=2), transpose_a=transpose_a)
    return tf.squeeze(C, -1)


def demodulate2(x, constellation):
    """ 星座解调，与类无关
    将x中的每一个复数信号映射到constellation中欧式距最小的信号，并返回该信号在constellation中的索引
    :param x: 待解调信号，shape=[d1, d2], dtype=tf.complex64
    :param constellation:  星座点，shape=[-1], dtype=tf.complex64
    :return: x信号解调后索引，shape=x.shape, dtype=tf.int32
    """
    x_complex = tf.reshape(x, shape=[-1, 1])
    constellation_complex = tf.reshape(constellation, [1, -1])
    d_square = tf.square(tf.abs(x_complex - constellation_complex))
    indices = tf.argmin(d_square, axis=1, output_type=tf.int32)
    ans = tf.reshape(indices, shape=tf.shape(x))
    return ans


def x_convert_to_complex(x, K, Nt):
    """ K个用户实数信号转换为复数信号
    :param x: shape=[None, K*2*Nt], dtype = tf.float32, 其中x=[x1_r,x1_i,x2_r,x2_i,...]
    :param K: shape=[], 用户数目
    :param Nt: shape=[], 每个用户天线数目
    :return:
    """
    x_complex = tf.complex(x[:, :Nt], x[:, Nt:2*Nt])  # 第1个用户的复数信号
    for k in range(1, K):
        xk = tf.complex(x[:, (k*2*Nt):(k*2*Nt+Nt)], x[:, (k*2*Nt+Nt):(k*2*Nt+2*Nt)])  # 第k个用户的复数信号
        x_complex = tf.concat([x_complex, xk], axis=1)
    return x_complex


def x_convert_to_real(x, K, Nt):
    """
    K个用户复数信号转换为实数信号
    :param x: k个用户的复数信号，shape=[None, K*Nt], dtype=tf.float64，其中x=[x1,x2,...]
    :param K: 用户数目，int
    :param Nt: 每个用户天线数，int
    :return:k个用户转换后实数信号，shape=[None, 2*K*Nt]
    """
    x_real = tf.concat([tf.real(x[:, :Nt]), tf.imag(x[:, :Nt])], axis=1)    # 第一个用户复数信号转换后的实数信号
    for k in range(1, K):
        xk = tf.concat([tf.real(x[:, k*Nt:(k+1)*Nt]), tf.imag(x[:, k*Nt:(k+1)*Nt])], axis=1)  # 第k个用户的实数信号
        x_real = tf.concat([x_real, xk], axis=1)
    return x_real


def network_init():
    init = tf.compat.v1.global_variables_initializer()   # 初始化操作
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session(config=config)
    sess.run(init)
    return sess


def DetNet(xhatt, H, y, L, Nt):
    xhat = []
    for i in range(L):
        W = tf.compat.v1.Variable(tf.random.normal(shape=[1, 2*Nt, 6*Nt], mean=0.0, stddev=0.01))
        b = tf.compat.v1.Variable(tf.constant(0.001, shape=[1, 2*Nt]), dtype=tf.float32)
        W_t = tf.tile(W, [tf.shape(xhatt)[0], 1, 1])
        b_t = tf.tile(b, [tf.shape(xhatt)[0], 1])
        HTH = tf.matmul(H, H, transpose_a=True)
        HTy = batch_matvec_mul2(H, y, transpose_a=True)
        HTHxhatt = batch_matvec_mul2(HTH, xhatt)
        concatenation = tf.concat([HTy, xhatt, HTHxhatt], axis=1)
        cal_t = batch_matvec_mul2(W_t, concatenation) + b_t
        # 非线性
        t = tf.compat.v1.Variable(0.5, dtype=tf.float32)
        xhatt = -1.0 + tf.nn.relu(cal_t + t)/tf.abs(t) - tf.nn.relu(cal_t - t)/tf.abs(t)
        xhat.append(xhatt)
    return xhat


def accuracy(x, y):
    """
    Computes the fraction of elements for which x and y are equal
    """
    return tf.reduce_mean(tf.cast(tf.equal(x, y), tf.float32))


def readData(path):
    """
    :param path: txt文件路径
    :return:
    """
    with open(path, 'r', encoding='utf-8') as f:
        js = f.read()
        data = json.loads(js)
    return data


def writeData(path, dict_data):
    js = json.dumps(dict_data, ensure_ascii=False)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(js)


def convert_to_complex(x, y, H, Nr, Nt, User):
    # Ns = self.Nt * self.User  # 所用用户的总切片数
    H_comp = tf.complex(H[:, :Nr, :Nt], H[:, Nr:, :Nt])
    for i in range(1, User):
        H_real = H[:, :Nr, i * 2 * Nt:i * 2 * Nt + Nt]

        H_imag = H[:, Nr:, i * 2 * Nt:i * 2 * Nt + Nt]

        H_complex_k = tf.complex(H_real, H_imag)

        H_comp = tf.concat((H_comp, H_complex_k), axis=2)

    x_comp = tf.complex(x[:, 0:Nt], x[:, Nt:2*Nt])
    for i in range(1, User):
        x_real = x[:, 2*i*Nt: 2*i*Nt+Nt]
        x_imag = x[:, 2*i*Nt+Nt: 2*(i+1)*Nt]
        x_complex_k = tf.complex(x_real, x_imag)
        x_comp = tf.concat((x_comp, x_complex_k), axis=1)
    y_comp = tf.complex(y[:, :Nr], y[:, Nr:])
    return x_comp, y_comp, H_comp


def loss_fun(xhat, x):
    """
    损失函数1，每一层输出的x的估计xhatk与x的均方差之和
    Input:
    xhat: 每一层输出的x的估计，是一个包含L个元素的列表，每个元素是Tensor(shape=(batch_size, 2*Nt), dtype=tf.float32)
    x: 发送调制符号x Tensor(shape=(batch_size,2*Nt), dtype=float32)
    Output:
    loss: 损失值 Tensor(shape=(), dtype=float32)
    """
    loss = 0.0
    for i in range(len(xhat)):
        lk = tf.compat.v1.losses.mean_squared_error(labels=x, predictions=xhat[i])
        loss += lk * tf.math.log(i+1)
    return loss


# ************************************** 基于numpy的常用方法 *******************************************
def extr_real(constellation):
    """
    仅支持实部虚部取值范围相同的星座提取实数可能取值范围
    """
    constellation = np.reshape(constellation, [-1, ])
    constellation_real = np.real(constellation)
    constellation_imag = np.imag(constellation)
    constellation_all = np.concatenate([constellation_real, constellation_imag], axis=0)
    ans = []
    for x in constellation_all:
        if x in ans:
            pass
        else:
            ans.append(x)
    return np.asarray(ans)

def num2bit(num, B): # 暂不使用
    """十进制到二进制转换
    0-(M-1)转换为bit表示
    num: dtype=np.int32, min=0, max=2^B-1，num.shape=[d1,d2]
    B: num中的每个数转换后的bit数, 0<B<=8
    """
    data = np.unpackbits(np.array(num, dtype=np.uint8))
    bit = np.reshape(data, [-1, num.shape[1], 8])[:, :, (8-B):]
    return bit


def demodulate_np(x, constellation):
    """
    基于numpy的复数信号解调
    :param x:shape=[d1,d2, ...], dtype=np.complex64, 待解调信号
    :param constellation: shape=[d1,],
    """
    x_complex = np.reshape(x, [-1, 1])
    constellation_complex = np.reshape(constellation, [1, -1])
    d_square = np.square(np.abs(x_complex - constellation_complex))
    indices = np.argmin(d_square, axis=1)
    ans = np.reshape(indices, x.shape)
    return ans


def real2complex_np(x: np, K: int, Nt: int):
    x_complex = x[:, :Nt] + 1j*x[:, Nt:2 * Nt]  # 第1个用户的复数信号
    for k in range(1, K):
        xk = x[:, (k * 2 * Nt):(k * 2 * Nt + Nt)] + 1j*x[:, (k * 2 * Nt + Nt):(k * 2 * Nt + 2 * Nt)]  # 第k个用户的复数信号
        x_complex = np.concatenate([x_complex, xk], axis=1)
    return x_complex


def plot_result(params, results):
    """
    绘制结果图
    """
    for key, value in results.items():
        if key == 'SNR_dBs':
            pass
        else:
            print(key, value)
            plt.semilogy(results['SNR_dBs'], value, '*-', linewidth=1.75, label=key)
    plt.grid(True, which='minor', linestyle='--')
    plt.yscale('log')
    plt.xlabel('SNR')
    plt.ylabel('SER')
    plt.legend()
    plt.title('Nr%dNt%dUser%d%s' % (params['Nr'], params['Nt'], params['User'], params['mod_name']))
    plt.legend()
    # plt.savefig('/home/xliangseu/Users/zfzdr/MIMO系统/dasjkdgs.png', dpi=1000)  # 保存图片
    # plt.show()
    return plt


def save_result(params, results):
    """
    结果保存
    """
    # 保存结果
    nt = datetime.datetime()
    if 1 in params['savetype']:
        np.save(params['results_savedir'] + '\\Nr%d_Nt%d_mod%s_SNR%d_%d_train_batchsize%d_maxEpoch%d_Layer%d_%s.npy' % (
            params['Nr'], params['Nt'], params['modulation'], params['SNR_dB_min'], params['SNR_dB_max'],
            params['train_batchsize'], params['maxEpoch'], params['L'], nt), results)

    # 保存图片
    if 2 in params['savetype']:
        plt.savefig(params['figures_savedir'] + '\\Nr%d_Nt%d_mod%s_SNR%d_%d_train_batchsize%d_maxEpoch%d_Layer%d_%s.jpg' % (
            params['Nr'], params['Nt'], params['modulation'], params['SNR_dB_min'], params['SNR_dB_max'],
            params['train_batchsize'], params['maxEpoch'], params['L'], nt), format='jpg', dpi=1000)
        plt.savefig(params['figures_savedir'] + '\\Nr%d_Nt%d_mod%s_SNR%d_%d_train_batchsize%d_maxEpoch%d_Layer%d_%s.eps' % (
            params['Nr'], params['Nt'], params['modulation'], params['SNR_dB_min'], params['SNR_dB_max'],
            params['train_batchsize'], params['maxEpoch'], params['L'], nt), format='eps', dpi=1000)
    return


def read_result(path):
    """
    读取results下的仿真结果，存入字典results中
    """
    results = {}
    with open(path, mode='r') as f:
        data = f.read()
        data = data.replace('\t', ' ')
        data = data.split('\n')
        for item in data:
            # print(item)
            if item != '':
                s = [i for i in item.split(' ') if i != '']
            if s[0] == '#':
                pass
            elif len(s) == 2:
                results[s[0]] = int(s[1])
            else:
                results[s[0]] = [float(i) for i in s[1:]]
    # for key, value in results.items():
    #     print(key, value)
    return results


def write_result(path, results):
    """
    未验证****暂不使用
    """
    with open(path, 'w+') as f:
        for key, value in results.items():
            f.write(key + '\t')
            for item in value:
                f.write(str(item) + '\t')
            f.write('\n')


def load_constellation(mod_name):
    """
    读取保存在constellation文件夹下txt中的星座点
    """
    path = 'constellation/' + mod_name + '.txt'
    with open(path, 'r') as f:
        data = f.read()
        try:
            data = data.replace('i', 'j')
        except:
            pass
        s = data.split('\n')
        constellation_complex = np.array([complex(item) for item in s], dtype=np.complex64)
        norm2 = np.mean(np.square(np.abs(constellation_complex)))
        constellation_complex = constellation_complex / np.sqrt(norm2)
    return constellation_complex


# ****************************** 命令行解析模块 *************************
def parse():
    parser = argparse.ArgumentParser(description='MIMO signal detection benchmark')

    parser.add_argument('--maxEpoch',
                        type=int,
                        required=False,
                        default=1000,
                        help='maxEpoch')
    parser.add_argument('--metric',
                        type=str,
                        required=False,
                        default='SER',
                        help='metric, SER or BER')
    parser.add_argument('--rho',
                       type=float,
                       required=False,
                       default= 0.0,
                       help='correlation factor')
    parser.add_argument('--SNR_dB_train',
                        type=int,
                        default=0,
                        required=False,
                        help='train SNR')
    parser.add_argument('--nRounds',
                        type=int,
                        default=10,
                        required=False,
                        help='train SNR')
    parser.add_argument('--K',
                        type=int,
                        required=True,
                        help='Number of Users'
                        )
    parser.add_argument('--Nt',
                        type=int,
                        required=True,
                        help='Number of senders')
    parser.add_argument('--Nr',
                        type=int,
                        required=True,
                        help='Number of receivers')
    parser.add_argument('--SNR_dB_min_test',
                        type=int,
                        required=True,
                        help='Minimum SNR in dB')
    parser.add_argument('--SNR_dB_max_test',
                        type=int,
                        required=True,
                        help='Maximum SNR in dB')
    parser.add_argument('--SNR_step_test',
                        type=int,
                        required=True,
                        help='Step SNR in dB')
    parser.add_argument('--iterations',
                        type=int,
                        default=50,
                        required=False,
                        help='iterations of test')
    parser.add_argument('--batch-size',
                        type=int,
                        default=500,
                        required=False,
                        help='Batch size')
    parser.add_argument('--modulation', '-mod',
                        type=str,
                        required=False,
                        default='QAM4',
                        help='Modulation type which can be BPSK, 4PAM, or MIXED')
    parser.add_argument('--ML',
                        action='store_true',
                        help='Include Maximum Likielihood')
    parser.add_argument('--OAMP',
                        action='store_true',
                        help='Include Orthogonal Approximate Message Passing')
    parser.add_argument('--AMP',
                        action='store_true',
                        help='Include Approximate Message Passing')
    parser.add_argument('--SDR',
                        action='store_true',
                        help='Include SDR detection algorithm')
    parser.add_argument('--BLAST',
                        action='store_true',
                        help='Include BLAST detection algorithm')
    parser.add_argument('--MMSE',
                        action='store_true',
                        help='Include MMSE')
    parser.add_argument('--ZF',
                        action='store_true',
                        help='Include Zero Forcing'
                        )
    parser.add_argument('--data-dir',
                        type=str,
                        required=False,
                        default=None,
                        help='Channel data directory')
    # parser.add_argument('--rho',
    #                    type=float,
    #                    required=False,
    #                    default=0.,
    #                    help='correlation factor of H')
    args = parser.parse_args()
    return args

# *********************************控制台输出重定向 ************************************************   
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a+')  # 不重写文件，以添加的形式加入文件末尾  'w'则重写文件
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
    
def log2txt(filename):
    """
    将控制台输出保存到文件 filename中
    """
    sys.stdout = Logger(filename, sys.stdout)
    # sys.stderr = Logger('a.log_file', sys.stderr)
    return 

def loggerset(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger 
    
    
    
    

if __name__ == "__main__":
    pass
    import numpy as np
    # import CommonCom as Comm 
    Nr = 32
    Nt = 4
    K = 4
    modName = 'QAM4'
    params = {'Nr': Nr, 'Nt':Nt, 'User': K, 'mod_name':modName}

    results = {
    'SNR_dBs': np.array([ 0,  2,  4,  6,  8, 10]), 
    'ZF': [0.28605, 0.190575, 0.10916250000000002, 0.0454, 0.014424999999999999, 0.0023625000000000005], 
    'MMSE': [0.2306875, 0.14795, 0.08103750000000001, 0.03435, 0.010849999999999999, 0.0017375],
    'ML': [0.21865625,	0.11765, 0.0415875,	0.00745, 0.000825, 3.75E-05]}
    plot_result(params, results)

