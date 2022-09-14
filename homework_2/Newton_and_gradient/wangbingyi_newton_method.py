"""
这份代码用牛顿法法寻找函数极小值、最小值点
Author: Wang_Bingyi
"""
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from mpl_toolkits.mplot3d import Axes3D


def get_func(x, y, f_name='f'):
    """
    返回f(x,y)的各项倒数以及梯度和Hessian矩阵
    :param x:
    :param y:
    :param f_name:
    :return:
    """
    res = dict()
    res['f'] = x**3 + y**3 - 3*x*y      #原函数
    if f_name == 'f':                   #因为x y 有时候是meshgrid，所以运行到res['hessian']会出Warning
        return res[f_name]
    res['pfpx'] = 3*(x**2) - 3*y        #对x求导
    res['pfpy'] = 3*(y**2) - 3*x        #对y求导
    res['p2fpx2'] = 6*x                 #对x求二阶导
    res['p2fpxpy'] = -3               #对xy求偏导
    res['p2fpypx'] = -3              #对yx求偏导
    res['p2fpy2'] = 6*y                 #对y求二阶导
    res['grad'] = np.array([res['pfpx'], res['pfpy']]).T         #梯度
    res['hessian'] = np.array([[res['p2fpx2'], res['p2fpxpy']],  #Hessian矩阵
                               [res['p2fpypx'], res['p2fpy2']]])
    return res[f_name]


def clear_plt():
    """
    清除matplot的图像缓存
    :return:
    """
    plt.figure(1)
    plt.clf()
    plt.figure(2)
    plt.clf()
    plt.figure(3)
    plt.clf()


class NEWTON(object):
    def __init__(self):
        """
        初始化NEWTON类对象各种变量
        """
        self.X = np.array([0, 0])  #临时变量 自变量(x,y)初始点
        self.z = 0                 #临时变量 函数值z的大小
        self.X_list = []           #绘图用的自变量序列
        self.z_list = []           #绘图用的函数值序列
        self.alpha = 0.1           #学习率一样的东西
        self.alpha2 = 10           #减少抖动的参数
        self.epsilon = 0.001       #递归终止条件
        self.iter_num = 0          #迭代次数
        self.is_avoid_isolate = True         #是否开启，防止hessian逆矩阵变得很大，导致很大抖动
        self.hess = np.array([0, 0])         #hessian矩阵
        self.grad = np.array([0, 0])         #梯度
        self.save_name1 = 'NewtonOPT_1.png'  #三张图片的保存名字
        self.save_name2 = 'NewtonOPT_2.png'
        self.save_name3 = 'NewtonOPT_3.png'
        self.analyse = ''                #记录是否是极值点的字符串，用于显示
        self.need_init_plt = 1        #是否需要初始化plt，就是等高线图
        self.ax = Axes3D(plt.figure(3), auto_add_to_figure=False)
        plt.figure(3).add_axes(self.ax)

    def one_iter(self):
        """
        用梯度法进行一次迭代的计算
        代码写了两百行，算法不到十二行
        :return: 无
        """
        self.grad = get_func(self.X[0], self.X[1], 'grad')
        self.hess = get_func(self.X[0], self.X[1], 'hessian')
        if self.is_avoid_isolate:   # 判断是否开启减少抖动
            if np.linalg.norm(np.dot(np.linalg.inv(self.hess), self.grad)) > 5:
                # 如果变化量的二范数过大，则变化量除以二范数
                self.X = self.X - np.dot(np.linalg.inv(self.hess), self.grad) / \
                         (np.linalg.norm(np.dot(np.linalg.inv(self.hess), self.grad))*self.alpha2)
            else:
                self.X = self.X - np.dot(np.linalg.inv(self.hess), self.grad) * self.alpha
        else:
            # 这是正常的牛顿法
            self.X = self.X - np.dot(np.linalg.inv(self.hess), self.grad) * self.alpha
        self.z = get_func(self.X[0], self.X[1], 'f')
        self.X_list.append(self.X)
        self.z_list.append(self.z)

    def judje(self):
        """
        通过Hessian矩阵判断当前的驻点是否是极值点
        通过成员self.analyse记录结果字符串
        :return: 0代表还没收敛，1代表收敛
        """
        self.analyse = ''
        if np.linalg.norm(self.grad) < self.epsilon:
            self.draw_line()
            print(self.X, "找到驻点")
            self.analyse += "\n  %f,%f " % (self.X[0], self.X[1])
            #根据二元函数极值点判定方法
            if np.linalg.det(self.hess) > 0 > self.hess[0][0]:
                print("找到极大值点")
                self.analyse += "找到驻点 找到极大值点"
            elif np.linalg.det(self.hess) > 0 and self.hess[0][0] > 0:
                print("找到极小值点")
                self.analyse += "找到驻点 找到极小值点"
            elif np.linalg.det(self.hess) < 0:
                print("该点不是极值点")
                self.analyse += "找到驻点 该点不是极值点"
            elif np.abs(np.linalg.det(self.hess)) < self.epsilon:
                print("该点不确定是否极大值极小值点")
                self.analyse += "找到驻点 该点不确定是否极大值极小值点"
            return 1
        return 0

    def add_one_point(self, x=3, y=3, is_random=False):
        cnt = 0  #迭代次数
        if is_random:
            self.X = np.random.rand(2)*10-5  #初始化范围(-5,5)
        else:
            self.X = np.array([x, y])        #手动设定点
        self.X_list.append(self.X)           #把结果保存到list中
        self.z_list.append(get_func(self.X[0], self.X[1], 'f'))
        self.one_iter()
        while self.judje() == 0:  #judje返回0代表还没收敛，返回1代表收敛
            self.one_iter()       #进行一次迭代
            cnt += 1
            if cnt > 1000:    #超过1000次迭代梯度还比较大，则认为是不收敛
                self.draw_line()
                self.analyse = '不收敛'
                print('不收敛')
                break

    def draw_line(self):
        """
        和matplotlab相关的绘图部分代码，一共是三张图
        :return: 无
        """
        # 第一张图 平面等高线图、热力图
        plt.figure(1)
        if self.need_init_plt:    #初始化第一张图的等高线图，热力图
            xx, yy = np.meshgrid(np.linspace(-5, 5, 100),
                                 np.linspace(-5, 5, 100))
            #print(xx)
            zz = get_func(xx, yy, 'f')
            plt.figure(1)
            plt.title("Newton", fontsize=20)  # 图像标题
            plt.xlabel('x', fontsize=14)      # X、Y刻度显示的文本
            plt.ylabel('y', fontsize=14)
            plt.tick_params(labelsize=10)  # 刻度标签大小
            plt.grid(linestyle=":")  # 网格线，并且只画与y相关联的
            plt.contourf(xx, yy, zz, 100, cmap='jet')
        newx = np.array(self.X_list).T
        plt.scatter(newx[0], newx[1])
        plt.plot(newx[0], newx[1])
        plt.savefig(self.save_name1)
        # 第二张图 下降的折线图
        plt.figure(2)
        plt.plot(np.arange(0, len(self.z_list)), self.z_list)
        plt.title('z change curve')
        plt.xlabel('iterator count', fontsize=14)      # X、Y刻度显示的文本
        plt.ylabel('f(x,y)', fontsize=14)
        plt.savefig(self.save_name2)
        # 第三张图 3D的图
        plt.figure(3)
        x_r = np.arange(-5, 5, 0.5)
        y_r = np.arange(-5, 5, 0.5)
        x_m, y_m = np.meshgrid(x_r, y_r)
        z_z = get_func(x_m, y_m, 'f')
        if self.need_init_plt:
            self.ax = Axes3D(plt.figure(3), auto_add_to_figure=False)
            plt.figure(3).add_axes(self.ax)
            self.need_init_plt = False
            self.ax.plot_surface(x_m, y_m, z_z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.3)
        self.ax.plot(newx[0], newx[1], self.z_list)
        self.ax.scatter(newx[0], newx[1], self.z_list)
        #.plot_surface(X_, Y_, Z_, color='palegreen', linewidth=0, antialiased=False, alpha=0.5)
        plt.savefig(self.save_name3)
        self.X_list.clear()
        self.z_list.clear()


class OPGUI(object):
    def __init__(self):
        """
        定义GUI和牛顿法的部分变量
        """
        self.newton_ins = NEWTON()      #牛顿法的一个实例
        self.window = tk.Tk()           #下面都是和GUI相关的
        self.window.title("PCAGUI")
        self.window.geometry('1500x900')
        self.save_name1 = 'NewtonOPT_1.png'
        self.save_name2 = 'NewtonOPT_2.png'
        self.save_name3 = 'NewtonOPT_3.png'
        self.pic_lab1 = tk.Label()
        self.pic_lab2 = tk.Label()
        self.pic_lab3 = tk.Label()
        self.is_avoid_isolate = tk.IntVar()
        self.img1 = tk.PhotoImage()
        self.img2 = tk.PhotoImage()
        self.img3 = tk.PhotoImage()
        self.inputx = tk.Entry()
        self.inputy = tk.Entry()
        self.input_lr = tk.Entry()
        self.input_lr2 = tk.Entry()
        self.label_log = tk.Label()
        self.widget_init()
        self.newton_ins.alpha = 0.1
        self.window.mainloop()

    def widget_init(self):
        """
        GUI各种控件的初始化，在这里不详细注释了
        :return:
        """
        button_tmp = tk.Button(self.window, text="清空", command=lambda: self.clr_plt())
        button_tmp.place(x=20, y=100)
        button_tmp = tk.Button(self.window, text="添加一个随机点", command=lambda: self.add_one(True))
        button_tmp.place(x=0, y=200)
        button_tmp = tk.Button(self.window, text="添加一个指定点", command=lambda: self.add_one(False))
        button_tmp.place(x=0, y=300)
        label_tmp = tk.Label(self.window, text="x")
        label_tmp.place(x=150, y=300)
        label_tmp = tk.Label(self.window, text="y")
        label_tmp.place(x=300, y=300)
        label_tmp = tk.Label(self.window, text="学习率")
        label_tmp.place(x=0, y=30)
        label_tmp = tk.Label(self.window, text="减少抖动参数")
        label_tmp.place(x=150, y=30)
        ckb_tmp = tk.Checkbutton(text="开启减少抖动参数", onvalue=1, offvalue=0,
                                 variable=self.is_avoid_isolate, command=lambda: self.check_bot())
        ckb_tmp.select()
        ckb_tmp.place(x=150, y=60)
        self.label_log = tk.Label(self.window, text="log", font="Helvetic 12 bold", bg="yellow")
        self.label_log.place(x=0, y=400)
        self.pic_lab1 = tk.Label(self.window, text='pic1')
        self.pic_lab1.place(x=450, y=10)
        self.pic_lab2 = tk.Label(self.window, text='pic2')
        self.pic_lab2.place(x=450, y=500)
        self.pic_lab3 = tk.Label(self.window, text='pic3')
        self.pic_lab3.place(x=1100, y=10)
        self.inputx = tk.Entry(self.window, width=10)
        self.inputx.place(x=170, y=300)
        self.inputy = tk.Entry(self.window, width=10)
        self.inputy.place(x=320, y=300)
        self.input_lr = tk.Entry(self.window, width=10)
        self.input_lr.insert(0, '0.3')
        self.input_lr.place(x=50, y=30)
        self.input_lr2 = tk.Entry(self.window, width=10)
        self.input_lr2.insert(0, '10')
        self.input_lr2.place(x=250, y=30)

    def show_plt(self):
        """将保存到本地的png图片显示在GUI上"""
        self.img1 = tk.PhotoImage(file=self.save_name1)
        self.pic_lab1.configure(image=self.img1)
        self.pic_lab1.image = self.img1
        self.img2 = tk.PhotoImage(file=self.save_name2)
        self.pic_lab2.configure(image=self.img2)
        self.pic_lab2.image = self.img2
        self.img3 = tk.PhotoImage(file=self.save_name3)
        self.pic_lab3.configure(image=self.img3)
        self.pic_lab3.image = self.img3

    def clr_plt(self):
        """
        清除GUI上的图片，以及清空plt缓存
        :return:
        """
        plt.figure(1)
        plt.clf()
        plt.savefig(self.save_name1)
        plt.figure(2)
        plt.clf()
        plt.savefig(self.save_name2)
        plt.figure(3)
        plt.clf()
        plt.savefig(self.save_name3)
        self.show_plt()
        self.newton_ins.need_init_plt = True

    def add_one(self, is_rand):
        """
        :param is_rand:
        :return:
        """
        self.newton_ins.is_avoid_isolate = self.is_avoid_isolate.get()
        self.newton_ins.alpha = float(self.input_lr.get())
        self.newton_ins.alpha2 = float(self.input_lr2.get())
        if is_rand:
            self.newton_ins.add_one_point(is_random=True)
        else:
            self.newton_ins.add_one_point(float(self.inputx.get()), float(self.inputy.get()))
        self.label_log.config(text=self.newton_ins.analyse)
        self.show_plt()

    def check_bot(self):
        if self.is_avoid_isolate.get() == 1:
            self.input_lr2['state'] = tk.NORMAL
        else:
            self.input_lr2['state'] = tk.DISABLED


if __name__ == '__main__':
    GUI_ins = OPGUI()  # GUI程序

    # 下面再单独运行10组数据，显示plt窗口图片
    clear_plt()
    newton_ins = NEWTON()
    for i in range(10):
        newton_ins.add_one_point(is_random=True)
    plt.show()
