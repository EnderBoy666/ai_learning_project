# 模型下载
---
嗯对，这里的模型是需要手动下载的，你可以选择适合的模型，然后去主程序里选择参数。

下载的模型请放置在model（就这个文件夹里）

#### 下载方法
* 在这个文件夹中内置了一些下载链接,从[魔塔](https://modelscope.cn/)进行下载。
* 如果您身处国外，可以使用[hugging face](https://huggingface.co/)进行下载。

好的，接下来我将讲述使用方法。
* 1.首先，我建议您先生成一个虚拟环境。
```py
python -m venv .downloads
```

* 2.接下来，请安装库
```bash
pip install modelscope sentencepiece protobuf
```
如果您使用hugging face,可以直接用官方工具下载，方法大致是一样的。只要确保模型最终被放到了model\模型名称内就可以了

* 3.安装好了之后，运行model_download.py文件（务必启动虚拟环境，如下）
```bash
.\.downloads\Scripts\activate
```
```bash
python model_download.py
```
* 4.接下来按照指示进行操作即可
* 5.现在你已经安装好了，请在主程序的settings.py里修改路径（即下载完成后跳出来的那个）
