# 参与TensorFlow文档写

TensorFlow迎文档献 - 如果改了文档，等同于改TensorFlow本身。 tensorflow.org上的文档分以下几：

* *API 文档* [API 文档](https://www.tensorflow.org/api_docs/)
  由
  [TensorFlow 源代](https://github.com/tensorflow/tensorflow)中的文档字符串(docstring)生成.
* *叙述文档* 部分内容[教程](https://www.tensorflow.org/tutorials)、
  [指南](https://www.tensorflow.org/guide)以及其他不属于TensorFlow代的内容. 部分代位于GitHub的
  [tensorflow/docs](https://github.com/tensorflow/docs) (repository)中.
* *社区翻* 些是由社区翻的指南和教程。他都被存放在
  [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site) (repository)中.

一些 [TensorFlow 目](https://github.com/tensorflow) 将文档源文件保存在独的存中，通常位于`docs/`目中。 参目的`CONTRIBUTING.md`文件或系者以参与。

参与到TensorFlow文档社区的方式有:

* 注GitHub中的 [tensorflow/docs](https://github.com/tensorflow/docs) (repository).
*  [docs@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs).
* 加入 [Gitter 聊天室](https://gitter.im/tensorflow/docs).

## API 文档

如果想更新API文档，找到其的
[源文件](https://www.tensorflow.org/code/tensorflow/python/)
并相的
<a href="https://www.python.org/dev/peps/pep-0257/" class="external">文档字符串(docstring)</a>.
tensorflow.org上的多API 引用的面都包含了指向源文件定位置的接。 文档字符串支持
<a href="https://help.github.com/en/articles/about-writing-and-formatting-on-github" class="external">Markdown格式</a>
并且（大多数候）都能使用
<a href="http://tmpvar.com/markdown.html" class="external">Markdown 器</a>
行.

有参考文档量以及如何参与文档冲刺和社区，参
[TensorFlow 2.0 API文档建](https://docs.google.com/document/d/1e20k9CuaZ_-hp25-sSd8E8qldxKPKQR-SkwojYr_r-U/preview)。

### 版本(Versions) 和 分支(Branches)

本网站的 [API 文档](https://www.tensorflow.org/api_docs/python/tf)
版本默最新的定二制文件即与通`pip install tensorflow`安装的版本所匹配.

默的TensorFlow 包是根据<a href="https://github.com/tensorflow/tensorflow" class="external">tensorflow/tensorflow</a>(repository)中的定分支`rX.x`所建的。文档是由
<a href="https://www.tensorflow.org/code/tensorflow/python/" class="external">Python</a>、
<a href="https://www.tensorflow.org/code/tensorflow/cc/" class="external">C++</a>与
<a href="https://www.tensorflow.org/code/tensorflow/java/" class="external">Java</a>代中的注与文档字符串所生成。

以前版本的TensorFlow文档在TensorFlow Docs (repository)中以[rX.x 分支](https://github.com/tensorflow/docs/branches) 的形式提供。在布新版本会添加些分支。

### 建API文档

注意：或API文档字符串不需要此，只需生成tensorflow.org上使用的HTML。

#### Python 文档

`tensorflow_docs`包中包含[Python API 文档](https://www.tensorflow.org/api_docs/python/tf)的生成器。
安装方式：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">pip install git+https://github.com/tensorflow/docs</code>
</pre>

要生成TensorFlow 2.0文档，使用
`tensorflow/tools/docs/generate2.py` 脚本:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow tensorflow</code>
<code class="devsite-terminal">cd tensorflow/tensorflow/tools/docs</code>
<code class="devsite-terminal">pip install tensorflow</code>
<code class="devsite-terminal">python generate2.py --output_dir=/tmp/out</code>
</pre>

注意：此脚本使用*已安装*的TensorFlow包生成文档并且用于TensorFlow 2.x.

## 叙述文档

TensorFlow [指南](https://www.tensorflow.org/guide) 和
[教程](https://www.tensorflow.org/tutorials) 是通
<a href="https://guides.github.com/features/mastering-markdown/" class="external">Markdown</a>
文件和交互式的
<a href="https://jupyter.org/" class="external">Jupyter</a> 本所写。 可以使用
<a href="https://colab.research.google.com/notebooks/welcome.ipynb"
   class="external">Google Colaboratory</a>
在的器中行本。
[tensorflow.org](https://www.tensorflow.org)中的叙述文档是根据
<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>的
`master` 分支建. 旧版本存在在GitHub (repository)下的`rX.x`行版分支中。

### 更改

行文档更新和修的最方法是使用GitHub的
<a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">Web文件器</a>。
[tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site/en)
(repository) 以找与
<a href="https://www.tensorflow.org">tensorflow.org</a>
中的URL 相的Markdown或notebook文件。 在文件的右上角，
<svg version="1.1" width="14" height="16" viewBox="0 0 14 16" class="octicon octicon-pencil" aria-hidden="true"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"></path></svg>
来打文件器。 文件，然后提交新的拉取求(pull request)。

### 置本地Git(repository)

于多文件或更的更新，最好使用本地Git工作流来建拉取求(pull request)。

注意：<a href="https://git-scm.com/" class="external">Git</a> 是用于跟踪源代更改的源版本控制系（VCS）。
<a href="https://github.com" class="external">GitHub</a>是一在服，
提供与Git配合使用的作工具。参<a href="https://help.github.com" class="external">GitHub Help</a>以置的GitHub并始使用。

只有在第一次置本地目才需要以下Git。

#### 制(fork) tensorflow/docs (repository)

在
<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>
的Github中，点*Fork*按
<svg class="octicon octicon-repo-forked" viewBox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"></path></svg>
在的GitHub下建自己的副本。制(fork) 完成，需要保持的副本副本与上游TensorFlow的同。

#### 克隆的(repository)

下一 <var>username</var>/docs 的副本到本地算机。是之后行操作的工作目：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone git@github.com:<var>username</var>/docs</code>
<code class="devsite-terminal">cd ./docs</code>
</pre>

#### 添加上游(upstream repo)以保持最新（可）

要使本地存与`tensorflow/docs`保持同，需要添加一个*上游(upstream)*
来下最新的更改。

注意：保在始撰稿*之前*更新的本地。定期向上游同会降低在提交拉取求(pull request)生<a href="https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line" class="external">合并冲突(merge conflict)</a>的可能性。

添加程:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git remote add <var>upstream</var> git@github.com:tensorflow/docs.git</code>

# 程
<code class="devsite-terminal">git remote -v</code>
origin    git@github.com:<var>username</var>/docs.git (fetch)
origin    git@github.com:<var>username</var>/docs.git (push)
<var>upstream</var>  git@github.com:tensorflow/docs.git (fetch)
<var>upstream</var>  git@github.com:tensorflow/docs.git (push)
</pre>

更新:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git checkout master</code>
<code class="devsite-terminal">git pull <var>upstream</var> master</code>

<code class="devsite-terminal">git push</code>  # Push changes to your GitHub account (defaults to origin)
</pre>

### GitHub 工作流

#### 1. 建一个新分支

从`tensorflow / docs`更新的后，从本地*master*分支中建一个新的分支:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git checkout -b <var>feature-name</var></code>

<code class="devsite-terminal">git branch</code>  # 列出本地分支
  master
* <var>feature-name</var>
</pre>

#### 2. 做更改

在喜的器中文件，并遵守
[TensorFlow文档式指南](./docs_style.md)。

提交文件更改：

<pre class="prettyprint lang-bsh">
# 看更改
<code class="devsite-terminal">git status</code>  # 看些文件被修改
<code class="devsite-terminal">git diff</code>    # 看文件中的更改内容

<code class="devsite-terminal">git add <var>path/to/file.md</var></code>
<code class="devsite-terminal">git commit -m "Your meaningful commit message for the change."</code>
</pre>

根据需要添加更多提交。

#### 3. 建一个拉取求(pull request)

将的本地分支上到的程GitHub
(github.com/<var>username</var>/docs):

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git push</code>
</pre>

推送完成后，消息可能会示一个URL，以自向上游存提交拉取求。如果没有，到
<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>
或者自己的GitHub将提示建拉取求(pull request)。

#### 4. 校

者和其他献者将核的拉取求(pull request)。参与并根据要求行修改。当的求得批准后，它将合并到上游TensorFlow文档中。

成功后：的更改会被TensorFlow文档接受。

从GitHub更新
[tensorflow.org](https://www.tensorflow.org)是一个独的。通常情况下，多个更改将被一并理，并定期上至网站中。

## 交互式本（notebook）

然可以使用GitHub的<a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">web文本器</a>来本JSON文件，但不推荐使用它，因格式的JSON可能会坏文件。 保在提交拉取求(pull request)之前本。

<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a>
是一个托管本境，可以松和行本文档。 GitHub中的本通将路径Colab URL（例如，位于GitHub中的本）在Google Colab中加：
<a href="https&#58;//github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb">https&#58;//github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb</a><br/>
可以通以下URL接在Google Colab中加:
<a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb">https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb</a>

有一个
<a href="https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo" class="external">Open in Colab</a>
展程序，可以在GitHub上本行此URL替。 在制的中中打本非常有用，因部按始接到TensorFlow Docs的`master`分支。

### 在Colab

在Google Colab境中，双元格以文本和代。文本元格使用Markdown格式，遵循
[TensorFlow文档式指南](./docs_style.md).

通点 *File > Download .pynb* 可以从Colab中下本文件。 将此文件提交到的[本地Git](###置本地Git(repository))后再提交拉取求。

如需要建新本，制和
<a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow 本模板</a>.

### Colab-GitHub工作流

可以直接从Google Colab和更新制的GitHub，而不是下本文件并使用本地Git工作流：

1. 在制(fork)的 <var>username</var>/docs 中，使用GitHub Web界面
   <a href="https://help.github.com/articles/creating-and-deleting-branches-within-your-repository" class="external">建新分支</a>。
2. 航到要的本文件。
3. 在Google Colab中打本：使用URL替或*Open in Colab* Chrome展程序。
4. 在Colab中本。
5. 通点
   *File > Save a copy in GitHub...*从Colab中向GitHub提交更改。保存框中到相的与分支。并添加一条有意的提交消息。
6. 保存之后，的或者
   <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>
   ，GitHub会提示建一个pull求。
7. 者会核的拉取求(pull request)。

成功后：的更改会被TensorFlow文档接受。

## 社区翻

社区翻是TensorFlow在全世界都可以的好方法。如需更新或添加翻，在[言目](https://github.com/tensorflow/docs/tree/master/site)中按照`en/`相同的目找到或添加一个新文件。英文档是*最基*的来源，翻尽可能地遵循些指南。也就是，翻尽量保持原汁原味。如果英，短，格或气不能翻成其他言，采用合者的翻。

注意：*勿翻* tensorflow.org中的API引用.

有特定于言的文档，使翻献者可以更松地行。 如果是作者，者或只是想社区建TensorFlow.org内容，加入：

* 体中文: [docs-zh-cn@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-zh-cn)
* 日: [docs-ja@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ja)
* : [docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
* 俄文: [docs-ru@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ru)
* 土耳其: [docs-tr@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-tr)

### 校知

所有文档更新都需要核。 了更有效地与TensorFlow翻社区行作，以下是一些保持言特定活的方法：

* 加入上面列出的言，以接收任何及言<code><a
  href="https://github.com/tensorflow/docs/tree/master/site">site/<var>lang</var></a></code>目的*已建的* 拉取求。
* 将的GitHub用名添加至`site/<lang>/REVIEWERS`文件在拉取求中能被自注。在被后，GitHub会向送拉取求中所有更改和的通知。

### 在翻中代保持最新

于像TensorFlow的源目，保持文档最新是一挑。在与社区交之后，翻内容的者能容忍有点的文本，但的代会人抓狂。了更容易保持代同，翻的本使用
[nb-code-sync](https://github.com/tensorflow/docs/blob/master/tools/nb_code_sync.py)工具：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">./tools/nb_code_sync.py [--lang=en] site/<var>lang</var>/notebook.ipynb</code>
</pre>

此脚本取言本的代元格，并根据英版本行。 剥注后，它会比代并更新言本（如果它不同）。 此工具于交互式git工作流特有用，可以性地将文件添加至更改中: `git add --patch site/lang/notebook.ipynb`

## Docs sprint

参加附近的
[TensorFlow 2.0 Global Docs Sprint](https://www.google.com/maps/d/viewer?mid=1FmxIWZBXi4cvSy6gJUW9WRPfvVRbievf)
活，或程加入。 注此
[博客文章](https://medium.com/tensorflow/https-medium-com-margaretmz-tf-docs-sprint-cheatsheet-7cb1dfd3e8b5?linkId=68384164)。些事件是始TensorFlow文档做出献的好方法。
