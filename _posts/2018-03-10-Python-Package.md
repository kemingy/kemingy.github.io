---
title: Python Package
date: 2018-03-10 16:28:17
tags: [Python]
categories: [笔记]
---

偶尔手痒了会撸一个 Python 的项目打包发布到 PyPI 上，方便以后安装使用。即使不是打算发布的，如果考虑把文档写在代码的注释里然后用 Sphinx 生成的，通常也是打包安装到本地，然后在一个地方集中生成文档，方便管理。

<!--more-->

今天就来记录一下 Python 项目打包的流程。（其实是怕自己又犯傻折腾半天找不到原因）[官方文档](https://packaging.python.org/tutorials/distributing-packages/) 改了几次之后明显质量好多了。

## 组成

一个简单的项目，通常由以下几部分组成。

- `setup.py` ：核心配置文件
- `setup.cfg` ：wheel 配置相关
- `README.rst` ：PyPI 不支持 **Markdown**，必须用 **reStructuredText** ，不是必须的
- `MANIFEST.in` ：是否需要包含其他非必须文件，不是必须的
- `LICENSE.txt` ：不是必须的
- `<your package>` ：不是必须的，但不可能没有这个吧 : )
- `.travis.yml` ：集成测试配置，不是必须的
- `.gitignore`

### `setup.py`

核心文件，感觉像 Sphinx 那样自动生成应该更好用。下面是一个 [官方](https://github.com/pypa/sampleproject/blob/master/setup.py) 缩略版的例子，复制粘贴大法好。

```python
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sampleproject',  # Required
    version='1.2.0',  # Required
    description='A sample Python project',  # Required
    long_description=long_description,  # Optional
    url='https://github.com/pypa/sampleproject',  # Optional
    author='The Python Packaging Authority',  # Optional
    author_email='pypa-dev@googlegroups.com',  # Optional
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    install_requires=['peppercorn'],  # Optional
)
```

看名字就知道是干嘛的了，就不多解释了。

如果需要包含模型数据之类的，可以查阅文档看相关的配置。

`version` 是比较需要注意的，一定要按照规范的格式来。

### `setup.cfg`

主要是关于项目是否支持 Python 2 和 Python 3 的，我已经弃用 Python 2 了，所以默认配置就够了。

```
[bdist_wheel]
universal=1
```

### `README.rst`

不是很习惯这个格式，而且 PyPI 支持的 rst 跟 Sphinx 又不太一样。

不过没关系，反正这个文件仅仅是在 PyPI 页面上显示的，大家习惯了从 GitHub 上看详细介绍，然后去看文档，所以没有这个文件也没关系，或者直接用一些工具转一下。

当然熟悉这个格式的就直接用这个好了。

### `MANIFEST.in` 

如果你需要把协议，数据，文档等东西也打包进去的话，可以配置这个。

```
# Include the license file
include LICENSE.txt

# Include the data files
recursive-include data *
```

### `.travis.yml` 

一般把代码发布到 GitHub 的时候习惯性会用这个，可以获得一个测试通过的标志 d(`･∀･)b

```yaml
language: python
python:
  - "2.7"
  - "3.5"
install:
  - pip install -r requirements.txt
script:
  - pytest
```

复制粘贴大法好。上次手误打错一个单词，然后测试的时候总是通不过，半天没找到原因 ( º﹃º )

## 打包

这个时候已经可以安装到本地了。

```sh
python setup.py install
```

执行之后会生成几个文件夹，暂时不用管。

这个时候有必要了解一下 Wheel 和 Egg 的区别了，详情看 [官方](https://packaging.python.org/discussions/wheel-vs-egg/) 。

先说个坑，我提交 Egg 之后发现根本没法安装，总是报错：

```
Collecting plane-0.0.3-py3.5.egg
  Could not find a version that satisfies the requirement plane-0.0.3-py3.5.egg (from versions: )
No matching distribution found for plane-0.0.3-py3.5.egg
```

搜一波反正是找不到一个合理的解释的。然后换到 Wheel 就一切正常了。

总之，用 Wheel 这个新的格式肯定没错了。

生成 wheel 文件：

```sh
python setup.py bdist_wheel --universal
```

如果你在 `setup.cfg` 中配置了，那么就可以省略后面的参数了。

## 发布

这时候要用到 `twine` 这个工具了，想起来几年前发布的时候各种奇怪的流程，现在真是精简多了。

首先你要在 [PyPI](https://pypi.org/account/register/) 上注册账号。

为了不用每次提交都手动填账号密码，可以在本地写一个文件：`$HOME/.pypirc`

```
[pypi]
username = <username>
password = <password>
```

然后提交就可以了：

```
twine upload dist/<your-wheel-package>
```

提交后记得测试一下能不能用 PIP 安装。