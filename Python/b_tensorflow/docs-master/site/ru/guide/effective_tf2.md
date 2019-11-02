# „^„†„†„u„{„„„y„r„~„„z TensorFlow 2.0

„B TensorFlow 2.0 „ƒ„t„u„|„p„~ „‚„‘„t „y„x„}„u„~„u„~„y„z „t„u„|„p„„‹„y„‡ „„€„|„„x„€„r„p„„„u„|„u„z TensorFlow „q„€„|„u„u
„„‚„€„t„…„{„„„y„r„~„„}„y. TensorFlow 2.0 „…„t„p„|„y„|
[„y„x„q„„„„€„‰„~„„u API](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md),
„„€„ƒ„|„u „‰„u„s„€ API „ƒ„„„p„|„y „q„€„|„u„u „ƒ„€„s„|„p„ƒ„€„r„p„~„~„„}„y
([Unified RNNs](https://github.com/tensorflow/community/blob/master/rfcs/20180920-unify-rnn-interface.md),
[Unified Optimizers](https://github.com/tensorflow/community/blob/master/rfcs/20181016-optimizer-unification.md)),
„y „|„…„‰„Š„u „y„~„„„u„s„‚„y„‚„€„r„p„|„ƒ„‘ „ƒ Python runtime, „ƒ
[Eager execution](https://www.tensorflow.org/guide/eager).

„M„~„€„s„y„u
[RFCs](https://github.com/tensorflow/community/pulls?utf8=%E2%9C%93&q=is%3Apr)
„€„q„Œ„‘„ƒ„~„y„|„y „y„x„}„u„~„u„~„y„‘ „{„€„„„€„‚„„u „r„€„Š„|„y „r TensorFlow 2.0. „^„„„€ „‚„…„{„€„r„€„t„ƒ„„„r„€
„„‚„u„t„ƒ„„„p„r„|„‘„u„„ „r„x„s„|„‘„t „~„p „{„„„€ „{„p„{ „t„€„|„w„~„p „r„„s„|„‘„t„u„„„ „‚„p„x„‚„p„q„€„„„{„p „r TensorFlow 2.0.
„P„‚„u„t„„€„|„p„s„p„u„„„ƒ„‘, „‰„„„€ „r„ „x„~„p„{„€„}„ „ƒ TensorFlow 1.x.

## „K„€„‚„€„„„{„p„‘ „r„„t„u„‚„w„{„p „€„ƒ„~„€„r„~„„‡ „y„x„}„u„~„u„~„y„z

### „O„‰„y„ƒ„„„{„p API

„M„~„€„s„€ API „|„y„q„€
[„…„t„p„|„u„~„ „|„y„q„€ „„u„‚„u„}„u„‹„u„~„](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md)
„r TF 2.0. „N„u„{„€„„„€„‚„„u „y„x „€„ƒ„~„€„r„~„„‡ „y„x„}„u„~„u„~„y„z „r„{„|„„‰„p„„„ „…„t„p„|„u„~„y„u `tf.app`,
`tf.flags`, „y `tf.logging` „r „„€„|„„x„…
[absl-py](https://github.com/abseil/abseil-py) „{„€„„„€„‚„„z „ƒ„u„z„‰„p„ƒ „ƒ „€„„„{„‚„„„„„}
„y„ƒ„‡„€„t„~„„} „{„€„t„€„}, „„u„‚„u„~„€„ƒ „„‚„€„u„{„„„€„r „{„€„„„€„‚„„u „~„p„‡„€„t„y„|„y„ƒ„ „r `tf.contrib`, „y „€„‰„y„ƒ„„„{„y
„€„ƒ„~„€„r„~„€„s„€ „„‚„€„ƒ„„„‚„p„~„ƒ„„„r„p „y„}„u„~ `tf.*` „„…„„„u„} „„u„‚„u„}„u„‹„u„~„y„‘ „‚„u„t„{„€ „y„ƒ„„€„|„„x„…„u„}„„‡ „†„…„~„{„ˆ„y„z
„r „„€„t„„p„{„u„„„ „~„p„„€„t„€„q„y„u `tf.math`. „N„u„€„{„„„€„‚„„u API „q„„|„y „x„p„}„u„‹„u„~„ „ƒ„r„€„y„}„y
„„{„r„y„r„p„|„u„~„„„p„}„y 2.0 - `tf.summary`, `tf.keras.metrics`, „y `tf.keras.optimizers`.
„N„p„y„q„€„|„u„u „„‚„€„ƒ„„„„} „ƒ„„€„ƒ„€„q„€„} „p„r„„„€„}„p„„„y„‰„u„ƒ„{„y „„‚„y„}„u„~„y„„„ „„„„y „„u„‚„u„y„}„u„~„€„r„p„~„y„‘ „‘„r„|„‘„u„„„ƒ„‘
„y„ƒ„„€„|„„x„€„r„p„~„y„u [„ƒ„{„‚„y„„„„p „€„q„~„€„r„|„u„~„y„‘ v2](upgrade.md).

### Eager execution

„B TensorFlow 1.X „€„„ „„€„|„„x„€„r„p„„„u„|„u„z „„„‚„u„q„€„r„p„|„€„ƒ„ „r„‚„…„‰„~„…„ „ƒ„€„q„y„‚„p„„„
[„p„q„ƒ„„„‚„p„{„„„~„€„u „ƒ„y„~„„„p„{„ƒ„y„‰„u„ƒ„{„€„u „t„u„‚„u„r„€](https://ru.wikipedia.org/wiki/%D0%90%D0%B1%D1%81%D1%82%D1%80%D0%B0%D0%BA%D1%82%D0%BD%D0%BE%D0%B5_%D1%81%D0%B8%D0%BD%D1%82%D0%B0%D0%BA%D1%81%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B5_%D0%B4%D0%B5%D1%80%D0%B5%D0%B2%D0%BE)
(„s„‚„p„†) „r„„„€„|„~„‘„‘ `tf.*` API „x„p„„‚„€„ƒ„. „H„p„„„u„} „„€„|„„x„€„r„p„„„u„|„y „t„€„|„w„~„ „r„‚„…„‰„~„…„
„ƒ„{„€„}„„y„|„y„‚„€„r„p„„„ „p„q„ƒ„„„‚„p„{„„„~„€„u „ƒ„y„~„„„p„{„ƒ„y„‰„u„ƒ„{„€„u „t„u„‚„u„r„€ „„…„„„u„} „„u„‚„u„t„p„‰„y „}„~„€„w„u„ƒ„„„r„p
„r„„‡„€„t„~„„‡ „y „r„‡„€„t„~„„‡ „„„u„~„x„€„‚„€„r „r „r„„x„€„r `session.run()`. TensorFlow 2.0 „r„„„€„|„~„‘„u„„„ƒ„‘
„ƒ„‚„p„x„… „w„u („{„p„{ „„„„€ „€„q„„‰„~„€ „t„u„|„p„u„„ Python) „y „r 2.0, „s„‚„p„†„ „y „ƒ„u„ƒ„ƒ„y„y „t„€„|„w„~„
„‚„p„ƒ„ƒ„}„p„„„‚„y„r„p„„„„ƒ„‘ „{„p„{ „t„u„„„p„|„y „‚„u„p„|„y„x„p„ˆ„y„y.

„O„t„~„y„} „x„p„}„u„„„~„„} „„€„q„€„‰„~„„} „„‚„€„t„…„{„„„€„} eager execution „‘„r„|„‘„u„„„ƒ„‘ „„„€, „‰„„„€
`tf.control_dependencies()` „q„€„|„u„u „~„u „„„‚„u„q„…„u„„„ƒ„‘, „„„p„{ „{„p„{ „r„ƒ„u „ƒ„„„‚„€„{„y „{„€„t„p
„r„„„€„|„~„‘„„„„ƒ„‘ „„€ „€„‰„u„‚„u„t„y („r „„‚„u„t„u„|„p„‡ `tf.function`, „{„€„t „ƒ „„€„q„€„‰„~„„}„y „„†„†„u„{„„„p„}„y
„r„„„€„|„~„‘„u„„„ƒ„‘ „r „„„€„} „„€„‚„‘„t„{„u „r „{„€„„„€„‚„€„} „€„~ „~„p„„y„ƒ„p„~).

### „N„u„„ „q„€„|„„Š„u „s„|„€„q„p„|„€„r

TensorFlow 1.X „x„s„p„‰„y„„„u„|„„~„€ „x„p„r„y„ƒ„u„| „€„„ „~„u„‘„r„~„„‡ „s„|„€„q„p„|„„~„„‡ „„‚„€„ƒ„„„‚„p„~„ƒ„„„r „y„}„u„~. „K„€„s„t„p
„r„ „r„„x„„r„p„|„y `tf.Variable()`, „€„~„p „„€„}„u„‹„p„|„p„ƒ„ „r „s„‚„p„† „„€ „…„}„€„|„‰„p„~„y„, „y „€„~„p
„€„ƒ„„„p„r„p„|„p„ƒ„ „„„p„} „t„p„w„u „u„ƒ„|„y „r„ „„€„„„u„‚„‘„|„y track „„u„‚„u„}„u„~„~„€„z Python „…„{„p„x„„r„p„r„Š„u„z „~„p
„~„u„s„€. „B„ „}„€„w„u„„„u „x„p„„„u„} „r„€„ƒ„ƒ„„„p„~„€„r„y„„„ „„„… `tf.Variable`, „~„€ „„„€„|„„{„€ „u„ƒ„|„y „r„ „x„~„p„|„y „y„}„‘
„ƒ „{„€„„„€„‚„„} „€„~„p „q„„|„p „ƒ„€„x„t„p„~„p. „^„„„€ „q„„|„€ „ƒ„|„€„w„~„€ „ƒ„t„u„|„p„„„ „u„ƒ„|„y „r„ „~„u „{„€„~„„„‚„€„|„y„‚„€„r„p„|„y
„ƒ„€„x„t„p„~„y„u „„u„‚„u„}„u„~„~„„‡. „B „‚„u„x„…„|„„„„p„„„u „„„„€„s„€, „‚„p„x„}„~„€„w„p„|„y„ƒ„ „r„ƒ„u „r„y„t„ „}„u„‡„p„~„y„x„}„€„r
„„„„„p„r„Š„y„u„ƒ„‘ „„€„}„€„‰„ „„€„|„„x„€„r„p„„„u„|„‘„} „ƒ„~„€„r„p „~„p„z„„„y „y„‡ „„u„‚„u„}„u„~„~„„u, „p „t„|„‘ „†„‚„u„z„}„r„€„‚„{„€„r -
„~„p„z„„„y „ƒ„€„x„t„p„~„~„„u „„€„|„„x„€„r„p„„„u„|„‘„}„y „„u„‚„u„}„u„~„~„„u: „O„q„|„p„ƒ„„„y „„u„‚„u„}„u„~„~„„‡, „s„|„€„q„p„|„„~„„u
„{„€„|„|„u„{„ˆ„y„y, „}„u„„„€„t„ „„€„}„€„‹„~„y„{„y „„„p„{„y„u „{„p„{ `tf.get_global_step()`,
`tf.global_variables_initializer()`, „€„„„„y„}„y„x„p„„„€„‚„ „~„u„‘„r„~„€ „r„„‰„y„ƒ„|„‘„„‹„y„u „s„‚„p„t„y„u„~„„„
„„€ „r„ƒ„u„} „€„q„…„‰„p„u„}„„} „„u„‚„u„}„u„~„~„„}, „y „„.„t. TensorFlow 2.0 „…„ƒ„„„‚„p„~„‘„u„„ „r„ƒ„u „„„„y „}„u„‡„p„~„y„x„}„
([Variables 2.0 RFC](https://github.com/tensorflow/community/pull/11)) „r „„€„|„„x„…
„}„u„‡„p„~„y„x„}„p „„€ „…„}„€„|„‰„p„~„y„: „O„„„ƒ„|„u„w„y„r„p„z„„„u „ƒ„r„€„y „„u„‚„u„}„u„~„~„„u! „E„ƒ„|„y „r„ „„€„„„u„‚„‘„|„y „ƒ„|„u„t
`tf.Variable`, „€„~ „q„…„t„u„„ „€„‰„y„‹„u„~ „ƒ„q„€„‚„‹„y„{„€„} „}„…„ƒ„€„‚„p.

„S„‚„u„q„€„r„p„~„y„u „€„„„ƒ„|„u„w„y„r„p„„„ „}„…„ƒ„€„‚ „ƒ„€„x„t„p„u„„ „t„€„„€„|„~„y„„„u„|„„~„…„ „‚„p„q„€„„„… „t„|„‘ „„€„|„„x„€„r„p„„„u„|„‘,
„~„€ „ƒ „€„q„Œ„u„{„„„p„}„y Keras („ƒ„}. „~„y„w„u), „~„p„s„‚„…„x„{„p „}„y„~„y„}„y„x„y„‚„€„r„p„~„p.

### „U„…„~„{„ˆ„y„y, „~„u „ƒ„u„ƒ„ƒ„y„y

„B„„x„€„r `session.run()` „„€„‰„„„y „„€„‡„€„w „~„p „r„„x„€„r „†„…„~„{„ˆ„y„y: „B„ „€„„‚„u„t„u„|„‘„u„„„u „r„r„€„t„~„„u
„t„p„~„~„„u, „†„…„~„{„ˆ„y„‘ „r„„x„„r„p„u„„„ƒ„‘ „y „r„ „„€„|„…„‰„p„u„„„u „~„p„q„€„‚ „‚„u„x„…„|„„„„p„„„€„r. „B TensorFlow 2.0,
„r„ „}„€„w„u„„„u „t„u„{„€„‚„y„‚„€„r„p„„„ „†„…„~„{„ˆ„y„ Python „y„ƒ„„€„|„„x„…„‘ `tf.function()` „‰„„„€„q„ „€„„„}„u„„„y„„„
„u„u „t„|„‘ JIT „{„€„}„„y„|„‘„ˆ„y„y „„„p„{ „‰„„„€ TensorFlow „r„„„€„|„~„‘„u„„ „u„s„€ „{„p„{ „u„t„y„~„„z „s„‚„p„†
([Functions 2.0 RFC](https://github.com/tensorflow/community/pull/20)). „^„„„€„„
„}„u„‡„p„~„y„x„} „„€„x„r„€„|„‘„u„„ TensorFlow 2.0 „„€„|„…„‰„y„„„ „r„ƒ„u „„‚„u„y„}„…„‹„u„ƒ„„„r„p „‚„u„w„y„}„p „s„‚„p„†„p:

-   „P„‚„€„y„x„r„€„t„y„„„u„|„„~„€„ƒ„„„: „†„…„~„{„ˆ„y„‘ „}„€„w„u„„ „q„„„„ „€„„„„y„}„y„x„y„‚„€„r„p„~„p (node pruning, kernel
    fusion, etc.)
-   „P„€„‚„„„p„„„y„r„~„€„ƒ„„„: „†„…„~„{„ˆ„y„‘ „}„€„w„u„„ „q„„„„ „„{„ƒ„„€„‚„„„y„‚„€„r„p„~„p / „‚„u„y„}„„€„‚„„„y„‚„€„r„p„~„p
    ([RFC SavedModel 2.0](https://github.com/tensorflow/community/pull/34), „‰„„„€
    „„€„x„r„€„|„‘„u„„ „„€„|„„x„€„r„p„„„u„|„‘„} „„€„r„„„€„‚„~„€ „y„ƒ„„€„|„„x„€„r„p„„„ „y „t„u„|„y„„„„ƒ„‘ „}„€„t„…„|„„~„„}„y
    „†„…„~„{„ˆ„y„‘„}„y TensorFlow.

```python
# TensorFlow 1.X
outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# TensorFlow 2.0
outputs = f(input)
```

„A„|„p„s„€„t„p„‚„‘ „r„€„x„}„€„w„~„€„ƒ„„„y „ƒ„r„€„q„€„t„~„€ „„u„‚„u„}„u„w„p„„„ „{„€„t Python „y TensorFlow „„€„|„„x„€„r„p„„„u„|„y
„}„€„s„…„„ „r„€„ƒ„„€„|„„x„€„r„p„„„„ƒ„‘ „„‚„u„y„}„…„‹„u„ƒ„„„r„p„}„y „r„„‚„p„x„y„„„u„|„„~„€„ƒ„„„y Python. „N„€ „„u„‚„u„~„€„ƒ„y„}„„z
TensorFlow „r„„„€„|„~„‘„u„„„ƒ„‘ „r „{„€„~„„„u„{„ƒ„„„p„‡, „„„p„{„y„‡ „{„p„{ mobile, C ++ „y JavaScript „q„u„x
„y„~„„„u„‚„„‚„u„„„p„„„€„‚„p Python. „X„„„€„q„ „„€„|„„x„€„r„p„„„u„|„‘„} „~„u „~„…„w„~„€ „q„„|„€ „„u„‚„u„„y„ƒ„„r„p„„„ „ƒ„r„€„z „{„€„t
„„‚„y „t„€„q„p„r„|„u„~„y„y `@ tf.function`, [AutoGraph](function.ipynb) „„‚„u„€„q„‚„p„x„…„u„„
„„€„t„}„~„€„w„u„ƒ„„„r„€ Python „{„€„~„ƒ„„„‚„…„y„‚„…„‘ „u„s„€ „r „„{„r„y„r„p„|„u„~„„„p„‡ TensorFlow:

*   `for`/`while` -> `tf.while_loop` (`break` and `continue` are supported)
*   `if` -> `tf.cond`
*   `for _ in dataset` -> `dataset.reduce`

AutoGraph „„€„t„t„u„‚„w„y„r„p„u„„ „„‚„€„y„x„r„€„|„„~„„u „r„|„€„w„u„~„y„‘ control flow, „‰„„„€ „t„u„|„p„u„„ „r„€„x„}„€„w„~„„}
„„†„†„u„{„„„y„r„~„€ „y „{„‚„p„„„{„€ „‚„u„p„|„y„x„€„r„p„„„ „}„~„€„s„y„u „ƒ„|„€„w„~„„u „„‚„€„s„‚„p„}„}„ „}„p„Š„y„~„~„€„s„€ „€„q„…„‰„u„~„y„‘,
„„„p„{„y„u „{„p„{ „‚„u„{„{„…„‚„u„~„„„~„„u „}„€„t„u„|„y, „€„q„…„‰„u„~„y„u „ƒ „„€„t„{„‚„u„„|„u„~„y„u„}, „„€„|„„x„€„r„p„„„u„|„„ƒ„{„y„u „ˆ„y„{„|„
„€„q„…„‰„u„~„y„‘ „y „}„~„€„s„€„u „t„‚„…„s„€„u.

## „Q„u„{„€„}„u„~„t„p„ˆ„y„y „‡„p„‚„p„{„„„u„‚„~„„u „t„|„‘ TensorFlow 2.0

### „Q„u„†„p„{„„„€„‚„„„„u „r„p„Š „{„€„t „r „}„u„~„„Š„y„u „†„…„~„{„ˆ„y„y

„O„q„„‰„~„„z „„€„|„„x„€„r„p„„„u„|„„ƒ„{„y„z „„p„„„„„u„‚„~ „r TensorFlow 1.X „q„„|„p „ƒ„„„‚„p„„„u„s„y„‘ "kitchen
sink"(„{„…„‡„€„~„~„p„‘ „}„€„z„{„p), „s„t„u „„‚„u„t„r„p„‚„y„„„u„|„„~„€ „r„„{„|„p„t„„r„p„|„€„ƒ„ „€„q„Œ„u„t„y„~„u„~„y„u „r„ƒ„u„‡
„r„€„x„}„€„w„~„„‡ „r„„‰„y„ƒ„|„u„~„y„z, „p „„€„„„€„} „r„„q„‚„p„~„~„„u „„„u„~„x„€„‚„ „€„ˆ„u„~„y„r„p„|„y„ƒ„ „ƒ `session.run()`. „B
TensorFlow 2.0, „„€„|„„x„€„r„p„„„u„|„‘„} „~„u„€„q„‡„€„t„y„}„€ „€„„„‚„u„†„p„{„„„€„‚„y„„„ „ƒ„r„€„z „{„€„t „r „}„u„~„„Š„y„u
„†„…„~„{„ˆ„y„y „{„€„„„€„‚„„u „r„„x„„r„p„„„„ƒ„‘ „„€ „}„u„‚„u „~„u„€„q„‡„€„t„y„}„€„ƒ„„„y. „B „€„q„‹„u„}, „~„u „€„q„‘„x„p„„„u„|„„~„€
„t„u„{„€„‚„y„‚„€„r„p„„„ „{„p„w„t„…„ „y„x „„„„y„‡ „†„…„~„{„ˆ„y„z „ƒ `tf.function`; „y„ƒ„„€„|„„x„…„z„„„u `tf.function`
„„„€„|„„{„€ „t„|„‘ „t„u„{„€„‚„y„‚„€„r„p„~„y„‘ „r„„ƒ„€„{„€„…„‚„€„r„~„u„r„„‡ „r„„‰„y„ƒ„|„u„~„y„z - „~„p„„‚„y„}„u„‚, „€„t„y„~ „Š„p„s
„€„q„…„‰„u„~„y„‘ „y„|„y „„‚„€„‡„€„t „r„„u„‚„u„t „r „r„p„Š„u„z „}„€„t„u„|„y.

### „I„ƒ„„€„|„„x„…„z„„„u „ƒ„|„€„y „y „}„€„t„u„|„y Keras „t„|„‘ „…„„‚„p„r„|„u„~„y„‘ „„u„‚„u„}„u„~„~„„}„y


Keras models and layers offer the convenient `variables` and
`trainable_variables` properties, which recursively gather up all dependent
variables. This makes it easy to manage variables locally to where they are
being used.

„M„€„t„u„|„y „y „ƒ„|„€„y Keras „„‚„u„t„|„p„s„p„„„ „…„t„€„q„~„„u „ƒ„r„€„z„ƒ„„„r„p `variables` „y
`trainable_variables`, „{„€„„„€„‚„„u „‚„u„{„…„‚„ƒ„y„r„~„€ „ƒ„€„q„y„‚„p„„„ „r„ƒ„u „x„p„r„y„ƒ„y„}„„u „„u„‚„u„}„u„~„~„„u. „^„„„€
„€„q„|„u„s„‰„p„u„„ „|„€„{„p„|„„~„€„u „…„„‚„p„r„|„u„~„y„u „„u„‚„u„}„u„~„~„„}„y „r „„„€„} „}„u„ƒ„„„u, „s„t„u „€„~„y „y„ƒ„„€„|„„x„€„r„p„|„y„ƒ„.

„R„‚„p„r„~„y„„„u:

```python
def dense(x, W, b):
  return tf.nn.sigmoid(tf.matmul(x, W) + b)

@tf.function
def multilayer_perceptron(x, w0, b0, w1, b1, w2, b2 ...):
  x = dense(x, w0, b0)
  x = dense(x, w1, b1)
  x = dense(x, w2, b2)
  ...

# „B„p„} „~„u„€„q„‡„€„t„y„}„€ „…„„‚„p„r„|„‘„„„ w_i and b_i, „p „y„‡ „‚„p„x„}„u„‚„~„€„ƒ„„„y „€„„‚„u„t„u„|„u„~„ „t„p„|„u„{„€ „€„„ „{„€„t„p.
```

„ƒ „r„u„‚„ƒ„y„u„z Keras:

```python
# „K„p„w„t„„z „ƒ„|„€„z „}„€„w„u„„ „q„„„„ „r„„x„r„p„~ „ƒ „ƒ„y„s„~„p„„„…„‚„€„z „„{„r„y„r„p„|„u„~„„„~„€„z linear(x)
layers = [tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for _ in range(n)]
perceptron = tf.keras.Sequential(layers)

# layers[3].trainable_variables => returns [w3, b3]
# perceptron.trainable_variables => returns [w0, b0, ...]
```

„R„|„€„y/„}„€„t„u„|„y Keras „~„p„ƒ„|„u„t„…„„„„ƒ„‘ „€„„ `tf.train.Checkpointable` „y „y„~„„„u„s„‚„y„‚„€„r„p„~„
„ƒ `@ tf.function`, „‰„„„€ „„€„x„r„€„|„‘„u„„ „~„p„„‚„‘„}„…„ „„‚„€„r„u„‚„‘„„„ „y„|„y „„{„ƒ„„€„‚„„„y„‚„€„r„p„„„
SavedModels „y„x „€„q„Œ„u„{„„„€„r Keras. „B„p„} „~„u „€„q„‘„x„p„„„u„|„„~„€ „y„ƒ„„€„|„„x„€„r„p„„„ Keras
`.fit ()` API „‰„„„€„q„ „r„€„ƒ„„€„|„„x„€„r„p„„„„ƒ„‘ „„„„y„}„y „y„~„„„u„s„‚„p„ˆ„y„‘„}„y.

„B„€„„ „„‚„y„}„u„‚ transfer learning, „{„€„„„€„‚„„z „t„u„}„€„~„ƒ„„„‚„y„‚„…„u„„, „{„p„{ Keras „€„q„|„u„s„‰„p„u„„
„ƒ„q„€„‚ „„€„t„}„~„€„w„u„ƒ„„„r„p „‚„u„|„u„r„p„~„„„~„„‡ „„u„‚„u„}„u„~„~„„‡. „D„€„„…„ƒ„„„y„}, „r„ „€„q„…„‰„p„u„„„u multi-headed
model with a shared trunk:

```python
trunk = tf.keras.Sequential([...])
head1 = tf.keras.Sequential([...])
head2 = tf.keras.Sequential([...])

path1 = tf.keras.Sequential([trunk, head1])
path2 = tf.keras.Sequential([trunk, head2])

# „O„q„…„‰„u„~„y„u „~„p „„u„‚„r„y„‰„~„„‡ „t„p„~„~„„‡
for x, y in main_dataset:
  with tf.GradientTape() as tape:
    prediction = path1(x)
    loss = loss_fn_head1(prediction, y)
  # „O„t„~„€„r„‚„u„}„u„~„~„p„‘ „€„„„„y„}„y„x„p„ˆ„y„‘ „r„u„ƒ„€„r trunk „y head1.
  gradients = tape.gradient(loss, path1.trainable_variables)
  optimizer.apply_gradients(zip(gradients, path1.trainable_variables))

# „S„€„~„{„p„‘ „~„p„ƒ„„„‚„€„z„{„p „r„„„€„‚„€„z head, „„u„‚„u„y„ƒ„„€„|„„x„€„r„p„~„y„u trunk
for x, y in small_dataset:
  with tf.GradientTape() as tape:
    prediction = path2(x)
    loss = loss_fn_head2(prediction, y)
  # „O„„„„y„}„y„x„y„‚„…„„„„ƒ„‘ „„„€„|„„{„€ „r„u„ƒ„p head2, „~„u „r„u„ƒ„p trunk
  gradients = tape.gradient(loss, head2.trainable_variables)
  optimizer.apply_gradients(zip(gradients, head2.trainable_variables))

# „B„ „}„€„w„u„„„u „€„„…„q„|„y„{„€„r„p„„„ „„„€„|„„{„€ „r„„‰„y„ƒ„|„u„~„y„‘ trunk „‰„„„€„q„ „t„‚„…„s„y„u „|„„t„y „}„€„s„|„y „y„}„y „r„€„ƒ„„€„|„„x„€„r„p„„„„ƒ„‘.
tf.saved_model.save(trunk, output_path)
```

### „K„€„}„q„y„~„y„‚„…„z„„„u tf.data.Datasets „y @tf.function

„P„‚„y „y„„„u„‚„p„ˆ„y„y „„€ „„„‚„u„~„y„‚„€„r„€„‰„~„„} „t„p„~„~„„}, „{„€„„„€„‚„„u „„€„}„u„‹„p„„„„ƒ„‘ „r „„p„}„‘„„„, „ƒ„r„€„q„€„t„~„€
„y„ƒ„„€„|„„x„…„z„„„u „‚„u„s„…„|„‘„‚„~„…„ „y„„„u„‚„p„ˆ„y„ Python. „I„~„p„‰„u, `tf.data.Dataset` - „|„…„‰„Š„y„z „ƒ„„€„ƒ„€„q
„t„|„‘ „„u„‚„u„t„p„‰„y „„„‚„u„~„y„‚„€„r„€„‰„~„„‡ „t„p„~„~„„‡ „ƒ „t„y„ƒ„{„p. „D„p„~„~„„u „‘„r„|„‘„„„„ƒ„‘
[iterables („~„u iterators)](https://docs.python.org/3/glossary.html#term-iterable),
„y „‚„p„q„€„„„p„„„ „„„p„{ „w„u, „{„p„{ „y „t„‚„…„s„y„u Python iterables „r „‚„u„w„y„}„u Eager. „B„ „}„€„w„u„„„u
„„€„|„~„€„ƒ„„„„ „y„ƒ„„€„|„„x„€„r„p„„„ „ƒ„r„€„z„ƒ„„„r„p dataset async prefetching/streaming „…„„p„{„€„r„p„r
„ƒ„r„€„z „{„€„t „r `tf.function ()`, „{„€„„„€„‚„p„‘ „x„p„}„u„~„‘„u„„ „y„„„u„‚„p„ˆ„y„ Python „„{„r„y„r„p„|„u„~„„„~„„}
„s„‚„p„†„€„} „€„„u„‚„p„ˆ„y„y „y„ƒ„„€„|„„x„…„„‹„y„} AutoGraph.

```python
@tf.function
def train(model, dataset, optimizer):
  for x, y in dataset:
    with tf.GradientTape() as tape:
      prediction = model(x)
      loss = loss_fn(prediction, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

If you use the Keras `.fit()` API, you won't have to worry about dataset
iteration.

```python
model.compile(optimizer=optimizer, loss=loss_fn)
model.fit(dataset)
```

### „B„€„ƒ„„€„|„„x„…„z„„„u„ƒ„ „„‚„u„y„}„…„‹„u„ƒ„„„r„p„}„y AutoGraph „ƒ Python control flow

AutoGraph „„‚„u„t„€„ƒ„„„p„r„|„‘„u„„ „ƒ„„€„ƒ„€„q „„‚„u„€„q„‚„p„x„€„r„p„~„y„‘ „x„p„r„y„ƒ„‘„‹„u„s„€ „€„„ „t„p„~„~„„‡ control flow
„r „„{„r„y„r„p„|„u„~„„„„z „‚„u„w„y„} „s„‚„p„†„p, „~„p„„‚„y„}„u„‚ `tf.cond` „y `tf.while_loop`.

„O„t„~„€ „€„q„„‰„~„€„u „}„u„ƒ„„„€, „s„t„u „„€„‘„r„|„‘„u„„„ƒ„‘ „x„p„r„y„ƒ„‘„‹„y„z „€„„ „t„p„~„~„„‡ control flow „~„p„‡„€„t„y„„„ƒ„‘
sequence models. `tf.keras.layers.RNN` „€„q„€„‚„p„‰„y„r„p„u„„ „‘„‰„u„z„{„… RNN, „„€„x„r„€„|„‘„‘ „r„p„}
„ƒ„„„p„„„y„‰„u„ƒ„{„y „y„|„y „t„y„~„p„}„y„‰„u„ƒ„{„y „‚„p„x„r„u„‚„~„…„„„ recurrence. „N„p„„‚„y„}„u„‚, „r„ „}„€„w„u„„
„„u„‚„u„€„„‚„u„t„u„|„y„„„ „t„y„~„p„}„y„‰„u„ƒ„{„…„ „‚„p„x„r„u„‚„„„{„… „ƒ„|„u„t„…„„‹„y„} „€„q„‚„p„x„€„}:

```python
class DynamicRNN(tf.keras.Model):

  def __init__(self, rnn_cell):
    super(DynamicRNN, self).__init__(self)
    self.cell = rnn_cell

  def call(self, input_data):
    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])
    outputs = tf.TensorArray(tf.float32, input_data.shape[0])
    state = self.cell.zero_state(input_data.shape[1], dtype=tf.float32)
    for i in tf.range(input_data.shape[0]):
      output, state = self.cell(input_data[i], state)
      outputs = outputs.write(i, output)
    return tf.transpose(outputs.stack(), [1, 0, 2]), state
```

„D„|„‘ „q„€„|„u„u „„€„t„‚„€„q„~„€„s„€ „€„q„x„€„‚„p „ƒ„r„€„z„ƒ„„„r AutoGraph, „ƒ„}„€„„„‚„y
[„‚„…„{„€„r„€„t„ƒ„„„r„€](./function.ipynb).

### tf.metrics „p„s„s„‚„u„s„y„‚„…„u„„ „t„p„~„~„„u and tf.summary „r„u„t„u„„ „y„‡ „|„€„s

„D„|„‘ „|„€„s„p summaries „y„ƒ„„€„|„„x„…„z„„„u `tf.summary. (Scalar | histogram | ...)` „y
„„u„‚„u„~„p„„‚„p„r„„„„u „u„s„€ „~„p writer „y„ƒ„„€„|„„x„…„‘ context manager. („E„ƒ„|„y „r„ „€„„…„ƒ„„„y„„„u context
manager, „~„y„‰„u„s„€ „ƒ„|„…„‰„y„„„ƒ„‘.) „B „€„„„|„y„‰„y„u „€„„ TF 1.x, summaries „€„„„„‚„p„r„|„‘„„„„ƒ„‘
„~„u„„€„ƒ„‚„u„t„ƒ„„„r„u„~„~„€ writer; „„„p„} „~„u„„ „€„„„t„u„|„„~„€„z „€„„u„‚„p„ˆ„y„y "merge" „y „€„„„t„u„|„„~„€„s„€ „r„„x„€„r„p
`add_summary()`, „‰„„„€ „€„x„~„p„‰„p„u„„, „‰„„„€ „x„~„p„‰„u„~„y„u `step` „t„€„|„w„~„€ „q„„„„ „…„{„p„x„p„~„€ „~„p „}„u„ƒ„„„u
„r„„x„€„r„p.

```python
summary_writer = tf.summary.create_file_writer('/tmp/summaries')
with summary_writer.as_default():
  tf.summary.scalar('loss', 0.1, step=42)
```

„X„„„€„q„ „€„q„Œ„u„t„y„~„y„„„ „t„p„~„~„„u „„u„‚„u„t „y„‡ „x„p„„y„ƒ„„ „r „r„y„t„u summaries, „y„ƒ„„€„|„„x„…„z„„„u
`tf.metrics`. „M„u„„„‚„y„{„p „‘„r„|„‘„„„„ƒ„‘ stateful: „€„~„y „~„p„{„p„„|„y„r„p„„„ „x„~„p„‰„u„~„y„‘ „y „r„€„x„r„‚„p„‹„p„„„
„ƒ„€„r„€„{„…„„~„„z „‚„u„x„…„|„„„„p„„, „{„€„s„t„p „r„ „r„„x„€„r„y„„„u `.result()`. „O„‰„y„ƒ„„„y„„„u „~„p„{„€„„|„u„~„~„„u
„x„~„p„‰„u„~„y„‘ „ƒ „„€„}„€„‹„„ `.reset_states ()`.

```python
def train(model, optimizer, dataset, log_freq=10):
  avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  for images, labels in dataset:
    loss = train_step(model, optimizer, images, labels)
    avg_loss.update_state(loss)
    if tf.equal(optimizer.iterations % log_freq, 0):
      tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
      avg_loss.reset_states()

def test(model, test_x, test_y, step_num):
  loss = loss_fn(model(test_x), test_y)
  tf.summary.scalar('loss', loss, step=step_num)

train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test')

with train_summary_writer.as_default():
  train(model, optimizer, dataset)

with test_summary_writer.as_default():
  test(model, test_x, test_y, optimizer.iterations)
```

„B„y„x„…„p„|„y„x„y„‚„…„z„„„u „ƒ„s„u„~„u„‚„y„‚„€„r„p„~„~„„u „‚„u„x„…„|„„„„p„„„ „~„p„„‚„p„r„y„r TensorBoard „r „t„y„‚„u„{„„„€„‚„y„z „ƒ
summary log:

```
tensorboard --logdir /tmp/summaries
```
