"""
这份代码利用numpy进行基本的矩阵运算，通过归一化，计算协方差矩阵，求解特征值和特征向量的方法来
Author: Wang_Bingyi 
"""
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
import logging
import math


class PCA2(object):
    def __init__(self, features, window_width=10, window_step=10, components=2, plot_save_name='pca2.png'):
        """
        初始化参数，加载数据，并提取出感兴趣的6个特征
        窗宽与步长：如果窗宽=10，步长=10，则每个窗取得的数据为[0,10),[10,20)...
        窗宽与步长：如果窗宽=10，步长=1，则每个窗取得的数据为[0,10),[1,11)，[2,12)...
        :param features:特征数据
        :param window_width:窗口宽度
        :param window_step:窗口移动步长
        :param components:需要保留的pc的数量
        :param plot_save_name:保存图片的文件名
        """
        self.logger = logging.getLogger(__name__)
        self.init_logging()
        self.logger.info('Initializing data')
        self.window_width = window_width
        self.window_step = window_step
        self.components = components
        self.plot_save_name = plot_save_name
        self.features = features
        self.features_size = len(features)
        self.ratio = 1
        self.sample_num = 0
        # 判断输入的参数的合法性，如果components比数据量还大，无法进行运算
        if (window_width < components or window_width > self.features_size or window_step < 0 or
                window_step > self.features_size or components < 0 or components > 6 or plot_save_name == ''):
            raise Exception("参数输入出错")

    def process_data(self):
        """
        调用sklearn的PCA库函数，计算主成分
        :return:
        """
        self.logger.debug('Boston feature size is %d' % self.features_size)
        # 根据窗宽，步长，数据集大小计算出一共有多少个窗口
        self.sample_num = math.floor((self.features_size - self.window_width) / self.window_step) + 1
        # 根据components初始化ratio的维度
        self.ratio = np.empty(self.components)
        self.logger.info('窗口数量为 %d', self.sample_num)
        for i in range(0, self.sample_num):
            self.logger.debug("%d -- %d" % (i*self.window_step, i*self.window_step+self.window_width))
            # 把当前窗口的数据提取出来
            sub_feature = self.features[i * self.window_step:i * self.window_step + self.window_width]
            # 归一化数据，下面的两个矩阵维度不相等，但是numpy有广播的功能，所以可以进行加减
            std_sub_feature = sub_feature-np.mean(sub_feature, axis=0)  # 减去均值的数据
            # 数据除以标准差
            std_sub_feature = std_sub_feature/(np.std(std_sub_feature, axis=0)+0.000001)
            # 计算协方差矩阵，用矩阵相乘代替for循环求和，运算速度应该是加快的，而且代码还少
            covariance_sub_feature = np.matmul(std_sub_feature.T, std_sub_feature)/(std_sub_feature.shape[0]-1)
            # 调用函数，求出特征方程和特征向量
            eigenvalues, eigenvectors = np.linalg.eig(covariance_sub_feature)
            # 将特征值按由大到小排列
            eigenvalues = np.sort(np.real(eigenvalues))[::-1]
            # 计算不同pc的百分比
            eigenvalues = eigenvalues/eigenvalues.sum(axis=0)
            self.ratio = np.c_[self.ratio, eigenvalues[0:self.components]]
            #print(ratio)
        # 因为在初始化ratio矩阵的时候会生成一组0数据，在这里剔除掉
        self.ratio = self.ratio[:, 1:]

    def draw_stack_plot(self):
        # 个人感觉用堆积图来表示不同成分占比更好一些
        # 设置标题，标签，x轴，y轴
        stack_label = []
        for i in range(0, self.components):
            stack_label.append('pc%d' % (i + 1))
        plt.stackplot(np.arange(0, self.sample_num, 1), self.ratio, labels=stack_label)
        plt.legend()
        plt.title('Principal component on Boston house price dataset use numpy and calculate')
        plt.xlabel('Time')
        plt.ylabel('Explained variance ratio')

    def show_plt(self):
        """
        显示图片，如果要保存图片的话，该函数要放在保存图片之后。
        为什么要分开写，是因为我想调用这个类给别的GUI的时候，不要让他自己显示
        :return:
        """
        self.logger.debug('This is not a static function!')
        plt.show()

    def save_plt(self):
        """
        保存图片。但是保存图片一定要放在plt.show()之前，否则输出的是一张空白。
        :return:
        """
        plt.savefig(self.plot_save_name)
        plt.clf()

    def init_logging(self, is_file_handler=True, is_command_handler=True, level=logging.ERROR):
        """
        设置日志的输出与级别
        然而我实在不知道这个logger初始化代码应该放在哪
        :param is_file_handler:是否开启向文件中保存日志信息
        :param is_command_handler:是否开启向命令行窗口输出日志信息
        :param level: 设置日志输出的级别，由低到高分别为 DEBUG INFO WARNING ERROR CRITICAL，比方说设置的是WARNING，
                      则DEBUG和INFO级别的日志将不会被输出
        :return:
        """
        self.logger.setLevel(level)
        # 设置向文件中输出日志的Handler
        if is_file_handler:
            fh = logging.FileHandler('logger.log', mode='w', encoding=None, delay=False)
            fh.setFormatter(logging.
                            Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
            self.logger.addHandler(fh)
        # 设置向命令行中输出日志的Handler
        if is_command_handler:
            ch = logging.StreamHandler(stream=None)
            ch.setFormatter(logging.
                            Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
            self.logger.addHandler(ch)


# 该位置的if语句能够使得本代码在被当做模块调用的时候，以下代码不会被执行
# 这份代码也可以单独执行，单独执行本代码文件，以下语句能被执行。


if __name__ == '__main__':
    #加载数据
    boston = load_boston()
    #提取后6个特征
    boston_features = boston['data'][:, -6:]
    # ww, ws, cp = input("input window_width window_step and components").split()
    pca1 = PCA2(boston_features, 10, 10, 2)
    pca1.process_data()
    pca1.draw_stack_plot()
    plt.savefig(pca1.plot_save_name)
    pca1.show_plt()
