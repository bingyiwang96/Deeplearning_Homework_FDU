"""
这份代码调用sklearn中的PCA方法实现对波士顿房价，最后6个属性的主成分分析
Author: Wang_Bingyi
"""
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import logging
import math


class PCA1(object):
    def __init__(self, features, window_width=10, window_step=10, components=2, plot_save_name='pca1.png'):
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
        # 根据需要保留的pc数量初始化ratio的维度
        self.ratio = np.empty(self.components)
        self.logger.info('窗口数量为 %d', self.sample_num)
        for i in range(0, self.sample_num):
            self.logger.debug("%d -- %d" % (i*self.window_step, i*self.window_step+self.window_width))
            # 设置主成分数量
            pca_inst = PCA(n_components=self.components)
            # 输入数据
            sub_feature = self.features[i * self.window_step:i * self.window_step + self.window_width]
            # 归一化数据，下面的两个矩阵维度不相等，但是numpy有广播的功能，所以可以进行加减
            std_sub_feature = sub_feature-np.mean(sub_feature, axis=0)  # 减去均值的数据
            # 数据除以标准差
            std_sub_feature = std_sub_feature/(np.std(std_sub_feature, axis=0)+0.000001)
            # 计算协方差矩阵，用矩阵相乘代替for循环求和，运算速度应该是加快的，而且代码还少
            pca_inst.fit(std_sub_feature)
            # 将输出的结果拼接至ratio矩阵
            self.ratio = np.c_[self.ratio, pca_inst.explained_variance_ratio_.T]
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
        plt.title('Principal component on Boston house price dataset use sklearn')
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
    pca1 = PCA1(boston_features, 10, 10, 2)
    pca1.process_data()
    pca1.draw_stack_plot()
    plt.savefig(pca1.plot_save_name)
    pca1.show_plt()
