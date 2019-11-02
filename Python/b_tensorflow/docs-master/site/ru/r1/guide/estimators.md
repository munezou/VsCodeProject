# Estimators

Note: „B„ƒ„‘ „y„~„†„€„‚„}„p„ˆ„y„‘ „r „„„„€„} „‚„p„x„t„u„|„u „„u„‚„u„r„u„t„u„~„p „ƒ „„€„}„€„‹„Ž„ „‚„…„ƒ„ƒ„{„€„s„€„r„€„‚„‘„‹„u„s„€
Tensorflow „ƒ„€„€„q„‹„u„ƒ„„„r„p „~„p „€„q„‹„u„ƒ„„„r„u„~„~„„‡ „~„p„‰„p„|„p„‡. „P„€„ƒ„{„€„|„Ž„{„… „„„„€„„ „„u„‚„u„r„€„t „~„u
„‘„r„|„‘„u„„„ƒ„‘ „€„†„y„ˆ„y„p„|„Ž„~„„}, „}„ „~„u „s„p„‚„p„~„„„y„‚„…„u„} „‰„„„€ „€„~ „~„p 100% „p„{„{„…„‚„p„„„u„~ „y „ƒ„€„€„„„r„u„„„ƒ„„„r„…„u„„
[„€„†„y„ˆ„y„p„|„Ž„~„€„z „t„€„{„…„}„u„~„„„p„ˆ„y„y „~„p „p„~„s„|„y„z„ƒ„{„€„} „‘„x„„{„u](https://www.tensorflow.org/?hl=en).
„E„ƒ„|„y „… „r„p„ƒ „u„ƒ„„„Ž „„‚„u„t„|„€„w„u„~„y„u „{„p„{ „y„ƒ„„‚„p„r„y„„„Ž „„„„€„„ „„u„‚„u„r„€„t, „}„ „q„…„t„u„} „€„‰„u„~„Ž „‚„p„t„
„…„r„y„t„u„„„Ž pull request „r [tensorflow/docs](https://github.com/tensorflow/docs)
„‚„u„„€„x„y„„„€„‚„y„z GitHub. „E„ƒ„|„y „r„ „‡„€„„„y„„„u „„€„}„€„‰„Ž „ƒ„t„u„|„p„„„Ž „t„€„{„…„}„u„~„„„p„ˆ„y„ „„€ Tensorflow
„|„…„‰„Š„u („ƒ„t„u„|„p„„„Ž „ƒ„p„} „„u„‚„u„r„€„t „y„|„y „„‚„€„r„u„‚„y„„„Ž „„u„‚„u„r„€„t „„€„t„s„€„„„€„r„|„u„~„~„„z „{„u„}-„„„€ „t„‚„…„s„y„}),
„~„p„„y„Š„y„„„u „~„p„} „~„p
[docs-ru@tensorflow.org list](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ru).

„B „„„„€„} „t„€„{„…„}„u„~„„„u „}„ „„€„x„~„p„{„€„}„y„}„ƒ„‘ `tf.estimator`, „r„„ƒ„€„{„€„…„‚„€„r„~„u„r„„} API TensorFlow,
„{„€„„„€„‚„„z „x„~„p„‰„y„„„u„|„Ž„~„€ „…„„‚„€„‹„p„u„„ „„‚„€„ˆ„u„ƒ„ƒ „ƒ„€„x„t„p„~„y„‘ „}„€„t„u„|„u„z „}„p„Š„y„~„~„€„s„€ „€„q„…„‰„u„~„y„‘.
Estimators „r„{„|„„‰„p„u„„ „r „ƒ„u„q„‘ „ƒ„|„u„t„…„„‹„y„u „€„„u„‚„p„ˆ„y„y:

*   „€„q„…„‰„u„~„y„u
*   „€„ˆ„u„~„{„…
*   „„‚„u„t„ƒ„{„p„x„p„~„y„u
*   „„{„ƒ„„€„‚„„ „}„€„t„u„|„y „~„p „ƒ„u„‚„r„u„‚

„S„ „}„€„w„u„Š„Ž „y„ƒ„„€„|„Ž„x„€„r„p„„„Ž „|„y„q„€ „…„w„u „s„€„„„€„r„„u Estimators „y„|„y „~„p„„y„ƒ„p„„„Ž „ƒ„r„€„y
„ƒ„€„q„ƒ„„„r„u„~„~„„u „t„|„‘ „€„ˆ„u„~„{„y. „B„ƒ„u Estimators „€„ƒ„~„€„r„p„~„ „~„p „{„|„p„ƒ„ƒ„u `tf.estimator.Estimator`.

„D„|„‘ „q„„ƒ„„„‚„€„s„€ „€„x„~„p„{„€„}„|„u„~„y„‘ „„€„„‚„€„q„…„z „x„p„„…„ƒ„„„y„„„Ž [„y„~„„„u„‚„p„{„„„y„r„~„„u „…„‚„€„{„y „„€ Estimator](../tutorials/estimators/linear.ipynb)
„r Google Colab. „X„„„€„q„ „…„x„~„p„„„Ž „€ „{„p„w„t„€„z „†„…„~„{„ˆ„y„y „„€„t„‚„€„q„~„u„u „ƒ„}„€„„„‚„y „ƒ„„„p„„„Ž„ [„s„€„„„€„r„„u Estimators](premade_estimators.md).
„D„|„‘ „€„x„~„p„{„€„}„|„u„~„y„‘ „ƒ „t„y„x„p„z„~„€„} „„„„€„s„€ API „ƒ„}„€„„„‚„y „~„p„Š [„t„€„{„|„p„t „~„p arxiv.org](https://arxiv.org/abs/1708.02637).

„O„q„‚„p„„„y „r„~„y„}„p„~„y„u: TensorFlow „„„p„{„w„u „r„{„|„„‰„p„u„„ „r „ƒ„u„q„‘ „…„ƒ„„„p„‚„u„r„Š„y„z „{„|„p„ƒ„ƒ
`Estimator` „r `tf.contrib.learn.Estimator`, „{„€„„„€„‚„„z „y„ƒ„„€„|„Ž„x„€„r„p„„„Ž „~„u „ƒ„„„€„y„„.


## „P„‚„u„y„}„…„‹„u„ƒ„„„r„p Estimators

Estimators „€„q„u„ƒ„„u„‰„y„r„p„„„ „ƒ„|„u„t„…„„‹„y„u „„‚„u„y„}„…„‹„u„ƒ„„„r„p:

*   „M„€„w„~„€ „x„p„„…„ƒ„{„p„„„Ž „}„€„t„u„|„y „~„p „€„ƒ„~„€„r„u Estimators „|„€„{„p„|„Ž„~„€ „y„|„y „~„p „‚„p„ƒ„„‚„u„t„u„|„u„~„~„€„}
    „ƒ„u„‚„r„u„‚„u „q„u„x „y„x„}„u„~„u„~„y„z „ƒ„„„‚„…„{„„„…„‚„ „}„€„t„u„|„y. „A„€„|„u„u „„„€„s„€, „„„ „}„€„w„u„Š„Ž „x„p„„…„ƒ„{„p„„„Ž „}„€„t„u„|„y
    „~„p CPU, GPU „y TPU „q„u„x „r„~„u„ƒ„u„~„y„‘ „y„x„}„u„~„u„~„y„z „r „{„€„t
*   „R „„€„}„€„‹„Ž„ Estimators „…„t„€„q„~„u„u „t„u„|„y„„„Ž„ƒ„‘ „ƒ„r„€„y„}„y „}„€„t„u„|„‘„}„y „ƒ „t„‚„…„s„y„}„y „‚„p„x„‚„p„q„€„„„‰„y„{„p„}„y
*   „M„€„w„~„€ „‚„p„x„‚„p„q„p„„„„r„p„„„Ž „ƒ„€„r„‚„u„}„u„~„~„„u „}„€„t„u„|„y „ƒ „‰„y„„„p„u„}„„} „r„„ƒ„€„{„€„…„‚„€„r„~„u„r„„} „{„€„t„€„}. „P„‚„€„‹„u „s„€„r„€„‚„‘,
    „s„€„‚„p„x„t„€ „|„u„s„‰„u „ƒ„€„x„t„p„r„p„„„Ž „}„€„t„u„|„y „ƒ Estimators, „‰„u„} „ƒ „~„y„x„{„€„…„‚„€„r„~„u„r„„} API TensorFlow
*   „R„p„}„y Estimators „„€„ƒ„„„‚„€„u„~„ „~„p `tf.keras.layers`, „{„€„„„€„‚„„u „…„„‚„€„‹„p„„„ „~„p„ƒ„„„‚„€„z„{„… „}„€„t„u„|„y
    „„€„t „ƒ„u„q„‘
*   Estimators „ƒ„„„‚„€„‘„„ „s„‚„p„†
*   Estimators „€„q„u„ƒ„„u„‰„y„r„p„„„ „„‚„€„ƒ„„„€„z „‚„p„ƒ„„‚„u„t„u„|„u„~„~„„z „ˆ„y„{„| „€„q„…„‰„u„~„y„‘, „{„€„„„€„‚„„z „{„€„~„„„‚„€„|„y„‚„…„u„„
    „{„p„{ „y „{„€„s„t„p:

    *   „ƒ„„„‚„€„y„„„Ž „s„‚„p„†
    *   „y„~„y„ˆ„y„p„|„y„x„y„‚„€„r„p„„„Ž „„u„‚„u„}„u„~„~„„u
    *   „x„p„s„‚„…„w„p„„„Ž „t„p„~„~„„u
    *   „€„q„‚„p„q„p„„„„r„p„„„Ž „y„ƒ„{„|„„‰„u„~„y„‘
    *   „ƒ„€„x„t„p„r„p„„„Ž „{„€„~„„„‚„€„|„Ž„~„„u „„„€„‰„{„y „y „r„€„ƒ„ƒ„„„p„~„p„r„|„y„r„p„„„Ž„ƒ„‘ „„‚„y „~„u„…„t„p„‰„~„„‡ „„€„„„„„{„p„‡
    *   „ƒ„€„‡„‚„p„~„‘„„„Ž „ƒ„„„p„„„y„ƒ„„„y„{„… „r TensorBoard

„P„‚„y „~„p„„y„ƒ„p„~„y„y „„‚„y„|„€„w„u„~„y„‘ „ƒ Estimators „„„ „t„€„|„w„u„~ „€„„„t„u„|„‘„„„Ž „x„p„s„‚„…„x„{„… „r„‡„€„t„~„„‡ „t„p„~„~„„‡
„€„„ „ƒ„p„}„€„z „}„€„t„u„|„y. „^„„„€ „‚„p„x„t„u„|„u„~„y„u „…„„‚„€„‹„p„u„„ „„{„ƒ„„u„‚„y„}„u„~„„„ „ƒ „‚„p„x„~„„}„y „~„p„q„€„‚„p„}„y „t„p„~„~„„‡.


## „C„€„„„€„r„„u Estimators

„C„€„„„€„r„„u Estimators „„€„x„r„€„|„‘„„„ „„„u„q„u „‚„p„q„€„„„p„„„Ž „~„p „q„€„|„u„u „r„„ƒ„€„{„€„} „…„‚„€„r„~„u, „„€ „ƒ„‚„p„r„~„u„~„y„
„ƒ „q„p„x„€„r„„} API TensorFlow. „S„u„q„u „q„€„|„Ž„Š„u „~„u „~„…„w„~„€ „r„€„|„~„€„r„p„„„Ž„ƒ„‘ „€ „„€„ƒ„„„‚„€„u„~„y„y „r„„‰„y„ƒ„|„y„„„u„|„Ž„~„€„s„€
„s„‚„p„†„p „y„|„y „ƒ„u„ƒ„ƒ„y„‘„‡ „€„q„…„‰„u„~„y„‘, „„€„ƒ„{„€„|„Ž„{„… Estimators „ƒ„p„}„y „t„u„|„p„„„ „x„p „„„u„q„‘ „r„ƒ„ „‚„p„q„€„„„….
„S„p„{„y„} „€„q„‚„p„x„€„} Estimators „ƒ„p„}„y „ƒ„€„x„t„p„„„ „y „…„„‚„p„r„|„‘„„„ „€„q„Œ„u„{„„„p„}„y `tf.Graph` „y 
`tf.Session`. „A„€„|„u„u „„„€„s„€, „s„€„„„€„r„„u Estimators „„€„x„r„€„|„‘„„„ „„„u„q„u „„{„ƒ„„u„‚„y„}„u„~„„„y„‚„€„r„p„„„Ž „ƒ 
„‚„p„x„~„„}„y „p„‚„‡„y„„„u„{„„„…„‚„p„}„y „ƒ „}„y„~„y„}„p„|„Ž„~„„}„y „y„x„}„u„~„u„~„y„‘„}„y „y„ƒ„‡„€„t„~„€„s„€ „{„€„t„p. „N„p„„‚„y„}„u„‚,
`tf.estimator.DNNClassifier` - „„„„€ „s„€„„„€„r„„z „{„|„p„ƒ„ƒ Estimator, „{„€„„„€„‚„„z „€„q„…„‰„p„u„„
„{„|„p„ƒ„ƒ„y„†„y„{„p„ˆ„y„y „}„€„t„u„|„y „~„p „€„ƒ„~„€„r„u „~„u„z„‚„€„~„~„€„z „ƒ„u„„„y „„‚„‘„}„€„s„€ „‚„p„ƒ„„‚„€„ƒ„„„‚„p„~„u„~„y„‘, „{„€„„„€„‚„p„‘ 
„ƒ„€„ƒ„„„€„y„„ „y„x *Dense* „ƒ„|„€„u„r.


### „R„„„‚„…„{„„„…„‚„p „„‚„€„s„‚„p„}„} „ƒ „s„€„„„€„r„„}„y Estimators

„P„‚„€„s„‚„p„}„}„p TensorFlow „~„p „€„ƒ„~„€„r„u „s„€„„„€„r„„‡ Estimators „€„q„„‰„~„€ „ƒ„€„ƒ„„„€„y„„ „y„x „ƒ„|„u„t„…„„‹„y„‡
„‰„u„„„„‚„u„‡ „Š„p„s„€„r:

1.  **„N„p„„y„ƒ„p„~„y„u „€„t„~„€„z „y„|„y „q„€„|„u„u „†„…„~„{„ˆ„y„z „t„|„‘ „x„p„s„‚„…„x„{„y „t„p„„„p„ƒ„u„„„p**. „N„p„„‚„y„}„u„‚,
    „ƒ„€„x„t„p„t„y„} „†„…„~„{„ˆ„y„ „t„|„‘ „y„}„„€„‚„„„p „„„‚„u„~„y„‚„€„r„€„‰„~„€„s„€ „ƒ„u„„„p „y „r„„„€„‚„…„ „†„…„~„{„ˆ„y„ „t„|„‘
    „y„}„„€„‚„„„p „„‚„€„r„u„‚„€„‰„~„€„s„€ „ƒ„u„„„p „t„p„~„~„„‡. „K„p„w„t„p„‘ „†„…„~„{„ˆ„y„‘ „t„|„‘ „x„p„s„‚„…„x„{„y „t„p„„„p„ƒ„u„„„p
    „t„€„|„w„~„p „r„€„x„r„‚„p„‹„p„„„Ž „t„r„p „€„q„Œ„u„{„„„p:

    *   „ƒ„|„€„r„p„‚„Ž, „r „{„€„„„€„‚„€„} „{„|„„‰„y „‘„r„|„‘„„„„ƒ„‘ „y„}„u„~„p„}„y „„p„‚„p„}„u„„„‚„€„r, „p „x„~„p„‰„u„~„y„‘
        „‘„r„|„‘„„„„ƒ„‘ „„„u„~„x„€„‚„p„}„y („y„|„y *SparseTensors*), „ƒ„€„t„u„‚„w„p„‹„y„u „ƒ„€„€„„„r„u„„„ƒ„„„r„…„„‹„y„u
        „t„p„~„~„„u „„p„‚„p„}„u„„„‚„€„r
    *   „„„u„~„x„€„‚, „ƒ„€„t„u„‚„w„p„‹„y„z „€„t„~„… „y„|„y „q„€„|„u„u „}„u„„„€„{

    „N„p„„‚„y„}„u„‚, „r „{„€„t„u „~„y„w„u „„€„{„p„x„p„~ „„‚„y„}„u„‚ „€„ƒ„~„€„r„~„€„s„€ „ƒ„{„u„|„u„„„p „t„|„‘ „†„…„~„{„ˆ„y„y „r„r„€„t„p
    „t„p„~„~„„‡:

```python
        def input_fn(dataset):
           ...  # „}„p„~„y„„…„|„y„‚„…„u„„ „t„p„„„p„ƒ„u„„„€„}, „y„x„r„|„u„{„p„‘ „ƒ„|„€„r„p„‚„Ž „„p„‚„p„}„u„„„‚„€„r „y „}„u„„„{„y
           return feature_dict, label
```

„R„}„€„„„‚„y „„€„t„‚„€„q„~„u„u „r „ƒ„„„p„„„Ž„u [„H„p„s„‚„…„x„{„p „t„p„~„~„„‡ „y „t„p„„„p„ƒ„u„„„€„r](../guide/datasets.md)

2.  **„O„„‚„u„t„u„|„u„~„y„u „{„€„|„€„~„€„{ „„p„‚„p„}„u„„„‚„€„r.** „K„p„w„t„p„‘ „{„€„|„€„~„{„p `tf.feature_column`
    „€„„‚„u„t„u„|„‘„u„„ „y„}„‘ „„p„‚„p„}„u„„„‚„p, „u„s„€ „„„y„ „y „|„„q„…„ „„‚„u„t„r„p„‚„y„„„u„|„Ž„~„…„ „€„q„‚„p„q„€„„„{„…
    „r„‡„€„t„~„„‡ „t„p„~„~„„‡. „N„p„„‚„y„}„u„‚, „r „ƒ„|„u„t„…„„‹„u„} „„‚„y„}„u„‚„u „{„€„t„p „}„ „ƒ„€„x„t„p„t„y„} „„„‚„y
    „{„€„|„€„~„{„y „„p„‚„p„}„u„„„‚„€„r, „r „{„€„„„€„‚„„‡ „q„…„t„…„„ „‡„‚„p„~„y„„„Ž„ƒ„‘ „t„p„~„~„„u „r „†„€„‚„}„p„„„u „ˆ„u„|„„‡
    „‰„y„ƒ„u„| „y„|„y „‰„y„ƒ„u„| „ƒ „„|„p„r„p„„‹„u„z „x„p„„‘„„„€„z. „P„u„‚„r„„u „t„r„u „{„€„|„€„~„{„y „„p„‚„p„}„u„„„‚„€„r „q„…„t„…„„
    „„‚„€„ƒ„„„€ „y„t„u„~„„„y„†„y„ˆ„y„‚„€„r„p„„„Ž „y„}„‘ „y „„„y„ „„p„‚„p„}„u„„„‚„p. „S„‚„u„„„Ž„‘ „{„€„|„€„~„{„p „„p„‚„p„}„u„„„‚„€„r „…„{„p„x„„r„p„u„„
    „~„p „|„‘„}„q„t„…-„r„„‚„p„w„u„~„y„u, „{„€„„„€„‚„€„u „q„…„t„…„„ „r„„x„„r„p„„„Ž„ƒ„‘ „t„|„‘ „€„ˆ„u„~„{„y „~„u„€„q„‚„p„q„€„„„p„~„~„„‡
    „t„p„~„~„„‡:

```python
# „O„„‚„u„t„u„|„y„} „„„‚„y „‰„y„ƒ„|„€„r„„‡ „{„€„|„€„~„{„y „„p„‚„p„}„u„„„‚„€„r.
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                       normalizer_fn=lambda x: x - global_education_mean)
```

3.  **„T„{„p„w„u„} „„€„t„‡„€„t„‘„‹„y„z „s„€„„„€„r„„z Estimator.**  „N„p„„‚„y„}„u„‚ „„„p„{ „}„ „…„{„p„w„u„}
    „s„€„„„€„r„„z Estimator „t„|„‘ „‚„u„Š„u„~„y„‘ „}„€„t„u„|„y `„|„y„~„u„z„~„€„s„€ „{„|„p„ƒ„ƒ„y„†„y„{„p„„„€„‚„p`:

```python
# „T„{„p„x„„r„p„u„} estimator, „„u„‚„u„t„p„u„} „{„€„|„€„~„{„y „„p„‚„p„}„u„„„‚„€„r.
estimator = tf.estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
)
```

4.  **„B„„x„€„r „}„u„„„€„t„p „€„q„…„‰„u„~„y„‘, „€„ˆ„u„~„{„y „y„|„y „„‚„u„t„ƒ„{„p„x„p„~„y„‘**
    „N„p„„‚„y„}„u„‚, „r„ƒ„u Estimators „y„}„u„„„ „}„u„„„€„t `train` „t„|„‘ „~„p„‰„p„|„p „€„q„…„‰„u„~„y„‘ „}„€„t„u„|„y:

```python
# `input_fn` - „†„…„~„{„ˆ„y„‘, „ƒ„€„x„t„p„~„~„p„‘ „r „ƒ„p„}„€„} „„u„‚„r„€„} „Š„p„s„u
estimator.train(input_fn=my_training_set, steps=2000)
```

### „P„‚„u„y„}„…„‹„u„ƒ„„„r„p „y„ƒ„„€„|„Ž„x„€„r„p„~„y„‘ „s„€„„„€„r„„‡ Estimators

„B „s„€„„„€„r„„‡ Estimators „y„ƒ„„€„|„Ž„x„…„„„„ƒ„‘ „|„…„‰„Š„y„u „„‚„p„{„„„y„{„y, „{„€„„„€„‚„„u „€„q„u„ƒ„„u„‰„y„r„p„„„
„ƒ„|„u„t„…„„‹„y„u „„‚„u„y„}„…„‹„u„ƒ„„„r„p:

*   „L„…„‰„Š„y„u „„‚„p„{„„„y„{„y „t„|„‘ „€„„‚„u„t„u„|„u„~„y„‘ „{„p„{„y„u „‰„p„ƒ„„„y „r„„‰„y„ƒ„|„y„„„u„|„Ž„~„€„s„€ „s„‚„p„†„p
    „x„p„„…„ƒ„{„p„„„Ž „ƒ„~„p„‰„p„|„p, „p „„„p„{„w„u „ƒ„„„‚„p„„„u„s„y„y „t„|„‘ „€„q„…„‰„u„~„y„‘ „~„p „€„t„~„€„} „…„ƒ„„„‚„€„z„ƒ„„„r„u
    „y„|„y „ˆ„u„|„€„} „{„|„p„ƒ„„„u„‚„u
*   „R„„„p„~„t„p„‚„„„y„x„y„‚„€„r„p„~„~„p„‘ „„‚„p„{„„„y„{„p „r„u„t„u„~„y„‘ „|„€„s„€„r „y „„€„|„…„‰„u„~„y„‘ „„€„|„u„x„~„€„z „ƒ„„„p„„„y„ƒ„„„y„{„y

„E„ƒ„|„y „„„ „~„u „ƒ„€„q„y„‚„p„u„Š„Ž„ƒ„‘ „y„ƒ„„€„|„Ž„x„€„r„p„„„Ž „s„€„„„€„r„„u Estimators, „„„€ „„„€„s„t„p „„„u„q„u
„„‚„y„t„u„„„ƒ„‘ „…„{„p„x„„r„p„„„Ž „r„ƒ„u „~„u„€„q„‡„€„t„y„}„„u „„p„‚„p„}„u„„„‚„ „ƒ„p„}„€„}„….


## „R„€„q„ƒ„„„r„u„~„~„„u Estimators

„`„t„‚„€„} „{„p„w„t„€„s„€ Estimator, „s„€„„„€„r„€„s„€ „y„|„y „~„p„„y„ƒ„p„~„~„€„s„€ „ƒ „~„…„|„‘, „‘„r„|„‘„u„„„ƒ„‘
**„†„…„~„{„ˆ„y„‘ „}„€„t„u„|„y**, „{„€„„„€„‚„p„‘ „„‚„u„t„ƒ„„„p„r„|„‘„u„„ „y„x „ƒ„u„q„‘ „}„u„„„€„t „t„|„‘ „„€„ƒ„„„‚„€„u„~„y„‘
„s„‚„p„†„p „t„|„‘ „€„q„…„‰„u„~„y„‘, „€„ˆ„u„~„{„y „y „„‚„u„t„ƒ„{„p„x„p„~„y„z. „K„€„s„t„p „„„ „y„ƒ„„€„|„Ž„x„…„u„Š„Ž „s„€„„„€„r„„z
Estimator, „{„„„€-„„„€ „…„w„u „~„p„„y„ƒ„p„| „†„…„~„{„ˆ„y„ „}„€„t„u„|„y „t„|„‘ „„„u„q„‘. „B „„„€„} „ƒ„|„…„‰„p„u,
„{„€„s„t„p „„„ „„€„|„p„s„p„u„Š„Ž„ƒ„‘ „~„p „ƒ„r„€„z „ƒ„€„q„ƒ„„„r„u„~„~„„z Estimator, „„„ „t„€„|„w„u„~ „ƒ„p„}
„~„p„„y„ƒ„p„„„Ž „„„„… „†„…„~„{„ˆ„y„. „A„€„|„u„u „„€„t„‚„€„q„~„€ „€ „„„€„}, „{„p„{ „~„p„„y„ƒ„p„„„Ž „†„…„~„{„ˆ„y„ „}„€„t„u„|„y
„„„ „}„€„w„u„Š„Ž „…„x„~„p„„„Ž „r „ƒ„„„p„„„Ž„u [„N„p„„y„ƒ„p„~„y„u „ƒ„€„q„ƒ„„„r„u„~„~„„‡ Estimators](../guide/custom_estimators.md)


## „Q„u„{„€„}„u„~„t„…„u„}„„z „‡„€„t „‚„p„q„€„„„

„M„ „‚„u„{„€„}„u„~„t„…„u„} „ƒ„|„u„t„…„„‹„y„z „„€„‚„‘„t„€„{ „ƒ„€„x„t„p„~„y„‘ „}„€„t„u„|„y „ƒ „„€„}„€„‹„Ž„ Estimators:

1.  „P„‚„u„t„„€„|„€„w„y„}, „‰„„„€ „u„ƒ„„„Ž „s„€„„„€„r„„z Estimator, „y „}„ „y„ƒ„„€„|„Ž„x„…„u„} „u„s„€ „t„|„‘
    „„€„ƒ„„„‚„€„u„~„y„‘ „~„p„Š„u„z „}„€„t„u„|„y, „p „„„p„{„w„u „y„ƒ„„€„|„Ž„x„…„u„} „‚„u„x„…„|„Ž„„„p„„„ „€„ˆ„u„~„{„y „t„|„‘ 
    „†„€„‚„}„y„‚„€„r„p„~„y„‘ „„„„p„|„€„~„~„€„z „}„€„t„u„|„y
2.  „R„€„x„t„p„u„} „y „„„u„ƒ„„„y„‚„…„u„} „„‚„€„ˆ„u„ƒ„ƒ „x„p„s„‚„…„x„{„y „t„p„~„~„„‡, „„‚„€„r„u„‚„‘„u„} „ˆ„u„|„€„ƒ„„„~„€„ƒ„„„Ž „y
    „~„p„t„u„w„~„€„ƒ„„„Ž „~„p„Š„y„‡ „t„p„~„~„„‡ „ƒ „s„€„„„€„r„„} Estimator
3.  „E„ƒ„|„y „u„ƒ„„„Ž „t„‚„…„s„y„u „„€„t„‡„€„t„‘„‹„y„u „p„|„Ž„„„u„‚„~„p„„„y„r„, „„„€„s„t„p „„{„ƒ„„u„‚„y„}„u„~„„„y„‚„…„u„} „ƒ „~„y„}„y
    „t„|„‘ „„€„y„ƒ„{„p Estimator, „{„€„„„€„‚„„z „„€„{„p„w„u„„ „|„…„‰„Š„y„u „‚„u„x„…„|„Ž„„„p„„„
4.  „B„€„x„}„€„w„~„€ „„€„„„‚„u„q„…„u„„„ƒ„‘ „…„|„…„‰„Š„y„„„Ž „~„p„Š„… „}„€„t„u„|„Ž „„‚„y „„€„}„€„‹„y „ƒ„€„x„t„p„~„y„‘ „~„p„Š„u„s„€
    „ƒ„€„q„ƒ„„„r„u„~„~„€„s„€ Estimator.


## „R„€„x„t„p„~„y„u Estimators „y„x „}„€„t„u„|„u„z Keras 

„S„ „}„€„w„u„Š„Ž „{„€„~„r„u„‚„„„y„‚„€„r„p„„„Ž „…„w„u „y„}„u„„‹„y„u„ƒ„‘ „… „„„u„q„‘ „}„€„t„u„|„y Keras „r Estimators. „^„„„€ „„€„x„r„€„|„y„„
„„„u„q„u „y„ƒ„„€„|„Ž„x„€„r„p„„„Ž „r„ƒ„u „„‚„u„y„}„…„‹„u„ƒ„„„r„p Estimators „t„|„‘ „t„p„~„~„€„z „}„€„t„u„|„y, „~„p„„‚„y„}„u„‚, „t„|„‘ „‚„p„ƒ„„‚„u„t„u„|„u„~„~„€„s„€
„€„q„…„‰„u„~„y„‘. „B„„x„€„r„y „†„…„~„{„ˆ„y„ `tf.keras.estimator.model_to_estimator` „{„p„{ „r „„‚„y„}„u„‚„u „~„y„w„u:

```python
# „R„€„x„t„p„u„} „}„€„t„u„|„Ž Inception v3 „r Keras:
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)

# „K„€„}„„y„|„y„‚„…„u„} „}„€„t„u„|„Ž „ƒ „€„„„„y„}„y„x„p„„„€„‚„€„}, „†„…„~„{„ˆ„y„u„z „„€„„„u„‚„Ž „y „}„u„„„‚„y„{„p„}„y „€„q„…„‰„u„~„y„‘ „„€ „r„„q„€„‚„….
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')

# „R„€„x„t„p„u„} Estimator „y„x „ƒ„{„€„}„„y„|„y„‚„€„r„p„~„~„€„z „}„€„t„u„|„y Keras. „O„q„‚„p„„„y „r„~„y„}„p„~„y„u, „‰„„„€ „y„x„~„p„‰„p„|„Ž„~„€„u
# „ƒ„€„ƒ„„„€„‘„~„y„u „}„€„t„u„|„y Keras „ƒ„€„‡„‚„p„~„‘„u„„„ƒ„‘ „„‚„y „ƒ„€„x„t„p„~„y„y Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)

# „S„u„„u„‚„Ž „}„ „}„€„w„u„} „y„ƒ„„€„|„Ž„x„€„r„p„„„Ž „„€„|„…„‰„u„~„~„„z Estimator „{„p„{ „|„„q„€„z „t„‚„…„s„€„z.
# „D„|„‘ „~„p„‰„p„|„p „r„€„ƒ„ƒ„„„p„~„€„r„y„} „r„r„€„t„~„€„u „y„}„‘ („y„|„y „y„}„u„~„p) „}„€„t„u„|„y Keras, „‰„„„€„q„ „}„ „}„€„s„|„y „y„ƒ„„€„|„Ž„x„€„r„p„„„Ž „y„‡
# „{„p„{ „y„}„u„~„p „{„€„|„€„~„€„{ „„p„‚„p„}„u„„„‚„€„r „†„…„~„{„ˆ„y„y „r„r„€„t„p „t„p„~„~„„‡ Estimator:
keras_inception_v3.input_names  # „r„„r„€„t„y„„: ['input_1']

# „K„p„{ „„„€„|„Ž„{„€ „}„ „„€„|„…„‰„y„} „r„r„€„t„~„„u „y„}„u„~„p, „}„ „}„€„w„u„} „ƒ„€„x„t„p„„„Ž „†„…„~„{„ˆ„y„ „r„r„€„t„p „t„p„~„~„„‡, „~„p„„‚„y„}„u„‚,
# „t„|„‘ „r„‡„€„t„p „t„p„~„~„„‡ „r „†„€„‚„}„p„„„u NumPy ndarray:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)

# „D„|„‘ „€„q„…„‰„u„~„y„‘ „r„„x„„r„p„u„} „†„…„~„{„ˆ„y„ `train` „„€„|„…„‰„u„~„~„€„s„€ „~„p„}„y Estimator:
est_inception_v3.train(input_fn=train_input_fn, steps=2000)
```

„O„q„‚„p„„„y „r„~„y„}„p„~„y„u, „‰„„„€ „y„}„u„~„p „{„€„|„€„~„€„{ „„p„‚„p„}„u„„„‚„€„r „y „}„u„„„€„{ Esitmator „}„ „„€„|„…„‰„y„|„y
„y„x „ƒ„€„€„„„r„u„„„ƒ„„„r„…„„‹„u„z „}„€„t„u„|„y Keras. „N„p„„‚„y„}„u„‚, „y„}„u„~„p „r„r„€„t„~„„‡ „{„|„„‰„u„z „t„|„‘ `train_input_fn`
„r„„Š„u „}„€„s„…„„ „q„„„„Ž „„€„|„…„‰„u„~„ „y„x `keras_inception_v3.input_names`, „y „„„p„{„y„} „w„u „€„q„‚„p„x„€„}
„„‚„u„t„ƒ„{„p„x„p„~„~„„u „y„}„u„~„p „}„€„s„…„„ „q„„„„Ž „„€„|„…„‰„u„~„ „y„x `keras_inception_v3.output_names`.

„P„€„t„‚„€„q„~„u„u „ƒ„}„€„„„‚„y „t„€„{„…„}„u„~„„„p„ˆ„y„ „r „ƒ„„„p„„„Ž„u `tf.keras.estimator.model_to_estimator`.
