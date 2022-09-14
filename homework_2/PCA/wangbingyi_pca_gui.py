"""
PCA主成分分析，根据题目要求
Author: Wang_Bingyi
"""

from sklearn.datasets import load_boston
import tkinter as tk
import tkinter.messagebox
import wangbingyi_pca1
import wangbingyi_pca2


class PCAGUI(object):
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("PCAGUI")
        self.window.geometry('1500x900')

        self.slider_ww = tkinter.Scale()
        self.slider_ws = tkinter.Scale()
        self.slider_cp = tkinter.Scale()
        self.label_pca1 = tk.Label()
        self.label_pca2 = tk.Label()
        self.PCA1 = 0
        self.PCA2 = 0
        self.img = 0
        self.img2 = 0
        # 加载数据
        self.boston = load_boston()
        # 提取后6个特征
        self.boston_features = self.boston['data'][:, -6:]
        self.widget_init()
        self.window.mainloop()
        #window.bind("<Key>", xFunc1)

    def widget_init(self):
        label_tmp = tk.Label(self.window, text="窗宽")
        label_tmp.place(x=0, y=100)
        self.slider_ww = tkinter.Scale(self.window, from_=1, to=100, orient=tkinter.HORIZONTAL,
                                       width='30', tickinterval=20, length=400, command=lambda x: self.test(x))
        self.slider_ww.place(x=0, y=120)
        self.slider_ww.set(10)

        label_tmp = tk.Label(self.window, text="步长")
        label_tmp.place(x=0, y=300)
        self.slider_ws = tkinter.Scale(self.window, from_=1, to=100, orient=tkinter.HORIZONTAL,
                                       width='30', tickinterval=20, length=400, command=lambda x: self.test(x))
        self.slider_ws.place(x=0, y=320)
        self.slider_ws.set(10)

        label_tmp = tk.Label(self.window, text="成分数量")
        label_tmp.place(x=0, y=500)
        self.slider_cp = tkinter.Scale(self.window, from_=1, to=6, orient=tkinter.HORIZONTAL,
                                       width='30', tickinterval=1, length=400, command=lambda x: self.test(x))
        self.slider_cp.place(x=0, y=520)
        self.slider_cp.set(2)

        label_frame = tk.LabelFrame(self.window, text="PCA1", width=500, height=500)
        label_frame.place(x=450, y=100)
        self.label_pca1 = tk.Label(label_frame)
        self.label_pca1.pack()
        label_frame = tk.LabelFrame(self.window, text="PCA2")
        label_frame.place(x=1200, y=100)
        self.label_pca2 = tk.Label(label_frame, text='2')
        self.label_pca2.pack()

    def test(self, x):
        # 虽然pycharm认为参数x是没有用的，实际上是tkinter GUI控件，传递进来的参数，不接受参数的话会报错
        len(x)
        ww = self.slider_ww.get()
        ws = self.slider_ws.get()
        cp = self.slider_cp.get()

        self.PCA1 = wangbingyi_pca1.PCA1(features=self.boston_features, window_step=ws, window_width=ww, components=cp)
        self.PCA2 = wangbingyi_pca2.PCA2(features=self.boston_features, window_step=ws, window_width=ww, components=cp)
        self.PCA1.process_data()
        self.PCA1.draw_stack_plot()
        self.PCA1.save_plt()
        self.PCA2.process_data()
        self.PCA2.draw_stack_plot()
        self.PCA2.save_plt()

        self.img = tk.PhotoImage(file='pca1.png')
        self.label_pca1.configure(image=self.img)
        self.label_pca1.image = self.img

        self.img2 = tk.PhotoImage(file='pca2.png')
        self.label_pca2.configure(image=self.img2)
        self.label_pca2.image = self.img2


if __name__ == '__main__':
    PCAGUI_inst = PCAGUI()
