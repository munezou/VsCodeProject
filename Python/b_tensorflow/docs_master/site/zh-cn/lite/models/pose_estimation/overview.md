# 姿

<img src="../images/pose.png" class="attempt-right" />

## 始使用

_PoseNet_ 能通像或中人体的位置行姿的。

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite">下此模</a>

Android 和 iOS 上的一一程即将面世. 与此同，如果想要在 web 器中体此模，可以
<a href="https://github.com/tensorflow/tfjs-models/tree/master/posenet">TensorFlow.js
GitHub 代</a>.

## 工作原理

姿通使用算机形技来片和中的人行和判断，如片中的人露出了肘臂。

了到清晰的目的，算法只是像中的人的身体位置所在，而不会去辨此人是。

点使用“号 部位”的格式行索引，并部位的探果伴随一个信任。信任取范在 0.0 至 1.0，1.0 最高信任。

<table style="width: 30%;">
  <thead>
    <tr>
      <th>号</th>
      <th>部位</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>鼻子</td>
    </tr>
    <tr>
      <td>1</td>
      <td>左眼</td>
    </tr>
    <tr>
      <td>2</td>
      <td>右眼</td>
    </tr>
    <tr>
      <td>3</td>
      <td>左耳</td>
    </tr>
    <tr>
      <td>4</td>
      <td>右耳</td>
    </tr>
    <tr>
      <td>5</td>
      <td>左肩</td>
    </tr>
    <tr>
      <td>6</td>
      <td>右肩</td>
    </tr>
    <tr>
      <td>7</td>
      <td>左肘</td>
    </tr>
    <tr>
      <td>8</td>
      <td>右肘</td>
    </tr>
    <tr>
      <td>9</td>
      <td>左腕</td>
    </tr>
    <tr>
      <td>10</td>
      <td>右腕</td>
    </tr>
    <tr>
      <td>11</td>
      <td>左</td>
    </tr>
    <tr>
      <td>12</td>
      <td>右</td>
    </tr>
    <tr>
      <td>13</td>
      <td>左膝</td>
    </tr>
    <tr>
      <td>14</td>
      <td>右膝</td>
    </tr>
    <tr>
      <td>15</td>
      <td>左踝</td>
    </tr>
    <tr>
      <td>16</td>
      <td>右踝</td>
    </tr>
  </tbody>
</table>

## 示例出

<img alt="Animation showing pose estimation" src="https://www.tensorflow.org/images/lite/models/pose_estimation.gif"/>

## 模性能

性能很大程度取决于的性能以及出的幅度(点和偏移向量)。PoseNet 于不同尺寸的片是不式，也就是在原始像和小后像中姿位置是一的。也意味着 PostNet 能精配置性能消耗。

出幅度决定了小后的和入的片尺寸的相程度。出幅度同影到了的尺寸和出的模型。更高的出幅度决定了更小的网和出的分辨率，和更小的可信度。

在此示例中，出幅度可以 8、16 或 32。句，当出幅度 32，会有最高性能和最差的可信度；当出幅度 8，会有用最高的可信度和最低的性能。我出的建是 16。

下展示了出幅度的程度决定放后的出和入的像的相度。更高的出幅度速度更快，但也会致更低的可信度。

<img alt="Output stride and heatmap resolution" src="../images/output_stride.png" >

## 于此模的更多内容

<ul>
  <li><a href="https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5">博客: 使用 TensorFlow.js 在器端上人体姿</a></li>
  <li><a href="https://github.com/tensorflow/tfjs-models/tree/master/posenet">TF.js 代: 器中的姿: PoseNet Model</a></li>
</ul>

### 使用案例

<ul>
  <li><a href="https://vimeo.com/128375543">‘毛球子’</a></li>
  <li><a href="https://youtu.be/I5__9hq-yas">神奇之将</a></li>
  <li><a href="https://vimeo.com/34824490">木偶列</a></li>
  <li><a href="https://vimeo.com/2892576">弥撒的声音 (性能)</a></li>
  <li><a href="https://www.instagram.com/p/BbkKLiegrTR/">添加</a></li>
  <li><a href="https://www.instagram.com/p/Bg1EgOihgyh/">互画片</a></li>
  <li><a href="https://www.runnersneed.com/expert-advice/gear-guides/gait-analysis.html">分析</a></li>
</ul>
