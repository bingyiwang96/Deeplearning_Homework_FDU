# 第一次作业

截止 无截止日期 得分 0 可访问时间 2021年9月27日 0:00 之后

1、中英文深度学习的教材一并发给大家。

2、按照引言部分提供网址安装软件（新入门同学建议采用keras，anaconda)，安装过程中如有问题，请先百度查找解决方案，解决不了再向助教老师请教。

3、如果是第一次接触python的同学，请到

https://www.runoob.com/python3/python3-data-type.html (链接到外部网站。)

学习到基本数据类型集合处。

# Answer

## 1.Anaconda下载安装 

Anaconda官方网站：<https://www.anaconda.com/>

众所周知因为国内网络环境原因，从官方网站上面下载Anaconda回很慢，甚至下载失败。包括安装package的时候，也会很慢甚至失败。因此可以考虑从清华大学开源软件镜像站下载Anaconda，并把安装源换为清华源。

清华大学开源软件镜像站Anaconda：<https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/>

可以通过修改用户目录下的 `.condarc` 文件。Windows 用户无法直接创建名为 `.condarc` 的文件，可先执行 `conda config --set show_channel_urls yes` 生成该文件之后再修改。

```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

即可添加 Anaconda Python 免费仓库。

运行 conda clean -i 清除索引缓存，保证用的是镜像站提供的索引。

运行 conda create -n myenv numpy 测试一下吧。

## 2.Conda的使用

```
# 1.创建一个虚拟环境
conda create --name pytorch_gpu python=3.8
#创建一个名为pytorch_gpu（可自定义）的虚拟环境，3.8指的是所安装python版本

# 2.进入一个已经存在的虚拟环境
conda activate pytorch

# 3.关闭当前的虚拟环境
conda deactivate

# 4.查看当前存在的所有虚拟环境
conda info -e

# 5.删除一个虚拟环境
conda remove --name pytorch_gpu --all  # pytorch_gpu指的是要删除的虚拟环境名称

# 6.安装package
conda install 包名
conda install -n 环境名 包名   # 如果不用-n指定环境名称，则被安装在当前活跃环境

# 7.查看当前环境下已安装的包
conda list

# 8.查看某个指定环境的已安装包
conda list -n 环境名

# 9.更新package
conda update -n 环境名 包名

# 10.删除package
conda remove -n 环境名 包名

# 11.查找包信息
conda search 包名

# 12.更新conda
conda update conda

# 13.更新anaconda
conda update anaconda
```

# 参考资料

[1] <https://blog.csdn.net/qq_46941656/article/details/119702123>