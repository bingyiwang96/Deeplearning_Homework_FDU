import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.ttk as ttk
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


class GUI(object):
    def __init__(self):
        """
        定义GUI控件和初始化变量
        """
        self.window = tk.Tk()           #下面都是和GUI相关的
        self.window.title("GUI")
        self.window.geometry('1600x550')
        self.notebook = ttk.Notebook(self.window)
        self.frame1 = tk.Frame()
        self.frame2 = tk.Frame()
        self.frame3 = tk.Frame()
        self.frame4 = tk.Frame()
        self.make_moon_ins = MAKE_MOON(self.frame1)  # make_moon实例
        self.make_circles_ins = MAKE_CIRCLES(self.frame2)  # make_circles实例
        self.make_classification_ins = MAKE_CLASSIFICATION(self.frame3)  # make_classification实例
        self.test_ins = TEST_CLF(self.frame4)  # make_classification实例
        self.widget_init()
        self.window.mainloop()

    def widget_init(self):
        """
        GUI各种控件的初始化，在这里不详细注释了
        :return:
        """
        self.notebook.add(self.frame1, text=' make_moon ')
        self.notebook.add(self.frame2, text=' make_circle ')
        self.notebook.add(self.frame3, text=' make_classification ')
        self.notebook.add(self.frame4, text=' 测试使用 ')
        self.notebook.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)


class MAKE_MOON(object):
    def __init__(self, frame):
        """
        定义GUI控件和初始化变量
        """
        # GUI控件定义
        self.frame = frame
        self.pic_lab1 = tk.Label(self.frame, text='pic1')
        self.pic_lab1.place(x=250, y=10)
        self.pic_lab2 = tk.Label(self.frame, text='pic2')
        self.pic_lab2.place(x=900, y=10)
        tmp_lab = tk.Label(self.frame, text='noise')
        tmp_lab.place(x=10, y=100)
        self.input_noise = tk.Entry(self.frame, width=10)
        self.input_noise.insert(0, '0.3')
        self.input_noise.place(x=120, y=100)
        tmp_lab = tk.Label(self.frame, text='随机种子')
        tmp_lab.place(x=10, y=200)
        self.input_rs = tk.Entry(self.frame, width=10)
        self.input_rs.insert(0, '2')
        self.input_rs.place(x=120, y=200)
        tmp_lab = tk.Label(self.frame, text='样本数量')
        tmp_lab.place(x=10, y=300)
        self.input_n_sample = tk.Entry(self.frame, width=10)
        self.input_n_sample.insert(0, '300')
        self.input_n_sample.place(x=120, y=300)
        tmp_lab = tk.Label(self.frame, text='测试集比例(0-1.0)')
        tmp_lab.place(x=10, y=250)
        self.input_test_scale = tk.Entry(self.frame, width=10)
        self.input_test_scale.insert(0, '0.4')
        self.input_test_scale.place(x=120, y=250)
        button_tmp = tk.Button(self.frame, text="开始迭代", command=lambda: self.begin_iter())
        button_tmp.place(x=20, y=400)
        # ----------------以上都是GUI控件定义和安置
        self.save_name1 = 'make_moon_1.png'
        self.save_name2 = 'make_moon_2.png'
        self.noise = 0.3        # make_moon 参数，噪声大小
        self.random_state = 2   # make_moon 参数，随机种子，是经过sklearn某个函数算出来的
        self.n_samples = 300    # make_moon 参数，样本点数目
        self.test_scale = 0.4    # make_moon 参数，测试集比例

    def begin_iter(self):
        """
        用GaussianNB拟合参数，并且保存图片
        :return:
        """
        self.clear_pic()  # 清除图像缓存
        self.get_pram()
        # 利用sklearn自带数据集产生数据
        x, y = make_moons(noise=self.noise, random_state=self.random_state, n_samples=self.n_samples)
        # 数据归一化，减均值除方差
        x = StandardScaler().fit_transform(x)
        # 将数据集分为训练集和测试集 random_state是如何划分的随机化种子
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_scale, random_state=42)
        # 绘图，meshgrid的分辨率
        h = .02  # step size in the mesh
        # 产生mesh
        x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
        y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        # 这是plt.contourf的颜色Rd是红Bu是蓝，从红变蓝
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        plt.figure(1)
        plt.title("make_moon data")
        # 训练集数据
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # 测试集数据，只是透明度不一样,浅色底点是测试集
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
        # 限制绘图大小
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.savefig(self.save_name1)
        # 拟合进先验分布为高斯函数的朴素贝叶斯
        guassnb_ins = GaussianNB().fit(x_train, y_train)
        # 得到测试集结果
        score = guassnb_ins.score(x_test, y_test)
        # 其实就是显示决策边界，那个底色的数据
        # z其实是把xx yy拼接成一对一对的数，predict_proba返回值是分为两个类的概率，[:, 1]是只取一类的概率，然后做图
        # 但现在z是串起来的
        # print(guassnb_ins.predict_proba(np.c_[xx.ravel(), yy.ravel()]))
        z = guassnb_ins.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        #print(z)
        plt.figure(2)
        # 然后再把z展开成数组
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=cm, alpha=.8)
        # 训练集数据
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # 测试集数据
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        # 在图片右下角显示score
        plt.text(xx.max() - .2, yy.min() + .2, ('score: %.2f' % score).lstrip('0'),
                 size=15, horizontalalignment='right')
        plt.savefig(self.save_name2)
        self.show_pict()

    def get_pram(self):
        """
        接受GUI输入的参数
        :return:
        """
        self.noise = float(self.input_noise.get())
        self.n_samples = int(self.input_n_sample.get())
        self.random_state = int(self.input_rs.get())
        self.test_scale = float(self.input_test_scale.get())

    def show_pict(self):
        """
        将保存的图片显示在GUI上
        :return:
        """
        img1 = tk.PhotoImage(file=self.save_name1)
        self.pic_lab1.configure(image=img1)
        self.pic_lab1.image = img1
        img2 = tk.PhotoImage(file=self.save_name2)
        self.pic_lab2.configure(image=img2)
        self.pic_lab2.image = img2

    def clear_pic(self):
        """
        清除matplotlab的图片缓存，因为show可以自动清缓存，savefig不能，所以要手动clf
        :return:
        """
        plt.figure(1)
        plt.clf()
        plt.savefig(self.save_name1)
        plt.figure(2)
        plt.clf()
        plt.savefig(self.save_name2)


class MAKE_CIRCLES(object):
    def __init__(self, frame):
        """
        定义GUI控件和初始化变量
        """
        # GUI控件定义
        self.frame = frame
        self.pic_lab1 = tk.Label(self.frame, text='pic1')
        self.pic_lab1.place(x=250, y=10)
        self.pic_lab2 = tk.Label(self.frame, text='pic2')
        self.pic_lab2.place(x=900, y=10)
        tmp_lab = tk.Label(self.frame, text='noise')
        tmp_lab.place(x=10, y=100)
        self.input_noise = tk.Entry(self.frame, width=10)
        self.input_noise.insert(0, '0.2')
        self.input_noise.place(x=120, y=100)
        tmp_lab = tk.Label(self.frame, text='随机种子')
        tmp_lab.place(x=10, y=200)
        self.input_rs = tk.Entry(self.frame, width=10)
        self.input_rs.insert(0, '2')
        self.input_rs.place(x=120, y=200)
        tmp_lab = tk.Label(self.frame, text='内外圆距离')
        tmp_lab.place(x=10, y=150)
        self.input_factor = tk.Entry(self.frame, width=10)
        self.input_factor.insert(0, '0.5')
        self.input_factor.place(x=120, y=150)
        tmp_lab = tk.Label(self.frame, text='样本数量')
        tmp_lab.place(x=10, y=300)
        self.input_n_sample = tk.Entry(self.frame, width=10)
        self.input_n_sample.insert(0, '300')
        self.input_n_sample.place(x=120, y=300)
        tmp_lab = tk.Label(self.frame, text='测试集比例(0-1.0)')
        tmp_lab.place(x=10, y=250)
        self.input_test_scale = tk.Entry(self.frame, width=10)
        self.input_test_scale.insert(0, '0.4')
        self.input_test_scale.place(x=120, y=250)
        button_tmp = tk.Button(self.frame, text="开始迭代", command=lambda: self.begin_iter())
        button_tmp.place(x=20, y=400)
        # ----------------以上都是GUI控件定义和安置
        self.save_name1 = 'make_moon_1.png'
        self.save_name2 = 'make_moon_2.png'
        self.noise = 0.2        # make_circles 参数，噪声大小
        self.random_state = 2   # make_circles 参数，随机种子，是经过sklearn某个函数算出来的
        self.n_samples = 300    # make_circles 参数，样本点数目
        self.test_scale = 0.4    # make_circles 参数，测试集比例
        self.factor = 0.5      # make_circles 参数，样本点数目

    def begin_iter(self):
        """
        用GaussianNB拟合参数，并且保存图片
        :return:
        """
        self.clear_pic()  # 清除图像缓存
        self.get_pram()
        # 利用sklearn自带数据集产生数据
        x, y = make_circles(factor=self.factor, noise=self.noise,
                            random_state=self.random_state, n_samples=self.n_samples)
        # 数据归一化，减均值除方差
        x = StandardScaler().fit_transform(x)
        # 将数据集分为训练集和测试集 random_state是如何划分的随机化种子
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_scale, random_state=42)
        # 绘图，meshgrid的分辨率
        h = .02  # step size in the mesh
        # 产生mesh
        x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
        y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        # 这是plt.contourf的颜色Rd是红Bu是蓝，从红变蓝
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        plt.figure(1)
        plt.title("make_circles data")
        # 训练集数据
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # 测试集数据，只是透明度不一样,浅色底点是测试集
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
        # 限制绘图大小
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.savefig(self.save_name1)
        # 拟合进先验分布为高斯函数的朴素贝叶斯
        guassnb_ins = GaussianNB().fit(x_train, y_train)
        # 得到测试集结果
        score = guassnb_ins.score(x_test, y_test)
        # 其实就是显示决策边界，那个底色的数据
        z = guassnb_ins.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        plt.figure(2)
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=cm, alpha=.8)
        # 训练集数据
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # 测试集数据
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        # 在图片右下角显示score
        plt.text(xx.max() - .2, yy.min() + .2, ('score: %.2f' % score).lstrip('0'),
                 size=15, horizontalalignment='right')
        plt.savefig(self.save_name2)
        self.show_pict()

    def get_pram(self):
        """
        接受GUI输入的参数
        :return:
        """
        self.noise = float(self.input_noise.get())
        self.n_samples = int(self.input_n_sample.get())
        self.random_state = int(self.input_rs.get())
        self.test_scale = float(self.input_test_scale.get())
        self.factor = float(self.input_factor.get())

    def show_pict(self):
        """
        将保存的图片显示在GUI上
        :return:
        """
        img1 = tk.PhotoImage(file=self.save_name1)
        self.pic_lab1.configure(image=img1)
        self.pic_lab1.image = img1
        img2 = tk.PhotoImage(file=self.save_name2)
        self.pic_lab2.configure(image=img2)
        self.pic_lab2.image = img2

    def clear_pic(self):
        """
        清除matplotlab的图片缓存，因为show可以自动清缓存，savefig不能，所以要手动clf
        :return:
        """
        plt.figure(1)
        plt.clf()
        plt.savefig(self.save_name1)
        plt.figure(2)
        plt.clf()
        plt.savefig(self.save_name2)


class MAKE_CLASSIFICATION(object):
    def __init__(self, frame):
        """
        定义GUI控件和初始化变量
        """
        # GUI控件定义
        self.frame = frame
        self.pic_lab1 = tk.Label(self.frame, text='pic1')
        self.pic_lab1.place(x=250, y=10)
        self.pic_lab2 = tk.Label(self.frame, text='pic2')
        self.pic_lab2.place(x=900, y=10)
        tmp_lab = tk.Label(self.frame, text='随机种子')
        tmp_lab.place(x=10, y=200)
        self.input_rs = tk.Entry(self.frame, width=10)
        self.input_rs.insert(0, '8')
        self.input_rs.place(x=120, y=200)
        tmp_lab = tk.Label(self.frame, text='样本数量')
        tmp_lab.place(x=10, y=300)
        self.input_n_sample = tk.Entry(self.frame, width=10)
        self.input_n_sample.insert(0, '300')
        self.input_n_sample.place(x=120, y=300)
        tmp_lab = tk.Label(self.frame, text='测试集比例(0-1.0)')
        tmp_lab.place(x=10, y=250)
        self.input_test_scale = tk.Entry(self.frame, width=10)
        self.input_test_scale.insert(0, '0.4')
        self.input_test_scale.place(x=120, y=250)
        button_tmp = tk.Button(self.frame, text="开始迭代", command=lambda: self.begin_iter())
        button_tmp.place(x=20, y=400)
        # ----------------以上都是GUI控件定义和安置
        self.save_name1 = 'make_moon_1.png'
        self.save_name2 = 'make_moon_2.png'
        self.noise = 0.2        # make_circles 参数，噪声大小
        self.random_state = 8   # make_circles 参数，随机种子，是经过sklearn某个函数算出来的
        self.n_samples = 300    # make_circles 参数，样本点数目
        self.test_scale = 0.4    # make_circles 参数，测试集比例
        self.factor = 0.5      # make_circles 参数，样本点数目

    def begin_iter(self):
        """
        用GaussianNB拟合参数，并且保存图片
        :return:
        """
        self.clear_pic()  # 清除图像缓存
        self.get_pram()
        # 利用sklearn自带数据集产生数据
        x, y = make_classification(n_samples=self.n_samples, n_features=2, n_redundant=0, n_informative=2,
                                   random_state=self.random_state, n_clusters_per_class=1)
        # 数据归一化，减均值除方差
        x = StandardScaler().fit_transform(x)
        # 将数据集分为训练集和测试集 random_state是如何划分的随机化种子
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_scale, random_state=42)
        # 绘图，meshgrid的分辨率
        h = .02  # step size in the mesh
        # 产生mesh
        x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
        y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        # 这是plt.contourf的颜色Rd是红Bu是蓝，从红变蓝
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        plt.figure(1)
        plt.title("make_classification data")
        # 训练集数据
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # 测试集数据，只是透明度不一样,浅色底点是测试集
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
        # 限制绘图大小
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.savefig(self.save_name1)
        # 拟合进先验分布为高斯函数的朴素贝叶斯
        guassnb_ins = GaussianNB().fit(x_train, y_train)
        # 得到测试集结果
        score = guassnb_ins.score(x_test, y_test)
        # 其实就是显示决策边界，那个底色的数据
        z = guassnb_ins.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        plt.figure(2)
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=cm, alpha=.8)
        # 训练集数据
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # 测试集数据
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        # 在图片右下角显示score
        plt.text(xx.max() - .2, yy.min() + .2, ('score: %.2f' % score).lstrip('0'),
                 size=15, horizontalalignment='right')
        plt.savefig(self.save_name2)
        self.show_pict()

    def get_pram(self):
        """
        接受GUI输入的参数
        :return:
        """
        self.n_samples = int(self.input_n_sample.get())
        self.random_state = int(self.input_rs.get())
        self.test_scale = float(self.input_test_scale.get())

    def show_pict(self):
        """
        将保存的图片显示在GUI上
        :return:
        """
        img1 = tk.PhotoImage(file=self.save_name1)
        self.pic_lab1.configure(image=img1)
        self.pic_lab1.image = img1
        img2 = tk.PhotoImage(file=self.save_name2)
        self.pic_lab2.configure(image=img2)
        self.pic_lab2.image = img2

    def clear_pic(self):
        """
        清除matplotlab的图片缓存，因为show可以自动清缓存，savefig不能，所以要手动clf
        :return:
        """
        plt.figure(1)
        plt.clf()
        plt.savefig(self.save_name1)
        plt.figure(2)
        plt.clf()
        plt.savefig(self.save_name2)


class TEST_CLF(object):
    def __init__(self, frame):
        """
        定义GUI控件和初始化变量
        """
        # GUI控件定义
        self.frame = frame
        self.pic_lab1 = tk.Label(self.frame, text='pic1')
        self.pic_lab1.place(x=250, y=10)
        self.pic_lab2 = tk.Label(self.frame, text='pic2')
        self.pic_lab2.place(x=900, y=10)
        tmp_lab = tk.Label(self.frame, text='noise')
        tmp_lab.place(x=10, y=100)
        self.input_noise = tk.Entry(self.frame, width=10)
        self.input_noise.insert(0, '0.3')
        self.input_noise.place(x=120, y=100)
        tmp_lab = tk.Label(self.frame, text='随机种子')
        tmp_lab.place(x=10, y=200)
        self.input_rs = tk.Entry(self.frame, width=10)
        self.input_rs.insert(0, '4')
        self.input_rs.place(x=120, y=200)
        tmp_lab = tk.Label(self.frame, text='样本数量')
        tmp_lab.place(x=10, y=300)
        self.input_n_sample = tk.Entry(self.frame, width=10)
        self.input_n_sample.insert(0, '300')
        self.input_n_sample.place(x=120, y=300)
        tmp_lab = tk.Label(self.frame, text='测试集比例(0-1.0)')
        tmp_lab.place(x=10, y=250)
        self.input_test_scale = tk.Entry(self.frame, width=10)
        self.input_test_scale.insert(0, '0.4')
        self.input_test_scale.place(x=120, y=250)
        button_tmp = tk.Button(self.frame, text="开始迭代", command=lambda: self.begin_iter())
        button_tmp.place(x=20, y=400)
        # ----------------以上都是GUI控件定义和安置
        self.save_name1 = 'make_moon_1.png'
        self.save_name2 = 'make_moon_2.png'
        self.noise = 0.3        # make_moon 参数，噪声大小
        self.random_state = 4   # make_moon 参数，随机种子，是经过sklearn某个函数算出来的
        self.n_samples = 300    # make_moon 参数，样本点数目
        self.test_scale = 0.4    # make_moon 参数，测试集比例

    def begin_iter(self):
        """
        用GaussianNB拟合参数，并且保存图片
        :return:
        """
        self.clear_pic()  # 清除图像缓存
        self.get_pram()
        # 利用sklearn自带数据集产生数据
        # x, y = make_moons(noise=self.noise, random_state=self.random_state, n_samples=self.n_samples)
        # x, y = make_circles(factor=0.5, noise=self.noise,
        #                     random_state=self.random_state, n_samples=self.n_samples)
        x, y = make_classification(n_samples=self.n_samples, n_features=2, n_redundant=0, n_informative=2,
                                   random_state=self.random_state, n_clusters_per_class=1)
        # 数据归一化，减均值除方差
        x = StandardScaler().fit_transform(x)
        #x = x + 5
        # 将数据集分为训练集和测试集 random_state是如何划分的随机化种子
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_scale, random_state=42)
        # 绘图，meshgrid的分辨率
        h = 0.52  # step size in the mesh
        # 产生mesh
        # x_train = x_train + 5
        # x_test = x_test + 5
        # y_train = y_train + 5
        # y_test = y_test + 5
        x_min, x_max = x[:, 0].min() - 6.5, x[:, 0].max() + 6.5
        y_min, y_max = x[:, 1].min() - 6.5, x[:, 1].max() + 6.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        # 这是plt.contourf的颜色Rd是红Bu是蓝，从红变蓝
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        plt.figure(1)
        plt.title("make_moon data")
        # 训练集数据
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # 测试集数据，只是透明度不一样,浅色底点是测试集
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
        # 限制绘图大小
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.savefig(self.save_name1)
        # 拟合进先验分布为高斯函数的朴素贝叶斯
        guassnb_ins = GaussianNB().fit(x_train, y_train)
        # 得到测试集结果
        score = guassnb_ins.score(x_test, y_test)
        # 其实就是显示决策边界，那个底色的数据
        # z其实是把xx yy拼接成一对一对的数，predict_proba返回值是分为两个类的概率，[:, 1]是只取一类的概率，然后做图
        # 但现在z是串起来的
        # print(guassnb_ins.predict_proba(np.c_[xx.ravel(), yy.ravel()]))
        z = guassnb_ins.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        #print(z)
        plt.figure(2)
        # 然后再把z展开成数组
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=cm, alpha=.8)
        # 训练集数据
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # 测试集数据
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        # 在图片右下角显示score
        plt.text(xx.max() - .2, yy.min() + .2, ('score: %.2f' % score).lstrip('0'),
                 size=15, horizontalalignment='right')
        plt.savefig(self.save_name2)
        self.show_pict()

    def get_pram(self):
        """
        接受GUI输入的参数
        :return:
        """
        self.noise = float(self.input_noise.get())
        self.n_samples = int(self.input_n_sample.get())
        self.random_state = int(self.input_rs.get())
        self.test_scale = float(self.input_test_scale.get())

    def show_pict(self):
        """
        将保存的图片显示在GUI上
        :return:
        """
        img1 = tk.PhotoImage(file=self.save_name1)
        self.pic_lab1.configure(image=img1)
        self.pic_lab1.image = img1
        img2 = tk.PhotoImage(file=self.save_name2)
        self.pic_lab2.configure(image=img2)
        self.pic_lab2.image = img2

    def clear_pic(self):
        """
        清除matplotlab的图片缓存，因为show可以自动清缓存，savefig不能，所以要手动clf
        :return:
        """
        plt.figure(1)
        plt.clf()
        plt.savefig(self.save_name1)
        plt.figure(2)
        plt.clf()
        plt.savefig(self.save_name2)


if __name__ == '__main__':
    GUI_ins = GUI()  #GUI程序
