# Predict-Fuel-Efficiency-using-TensorFlow
The TensorFlow solution can be found [here](https://www.tensorflow.org/tutorials/keras/regression).

[Data source](http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data)

# Attribute Information
|||
|-|-|
|mpg|continuous|
|cylinders|multi-valued discrete|
|displacement|continuous|
|horsepower|continuous|
|weight|continuous|
|acceleration|continuous|
|model year|multi-valued discrete|
|origin|multi-valued discrete|
|car name|string (unique for each instance)|

# Observations
- Found na in Horsepower column
- Cylinders is the categorical column
- Manipulated data using pd.get_dummies
- Using the pairplot, it's understood that regression is the best approach

# Model 1
```Python
horsepower_normalizer = tf.keras.layers.Normalization(input_shape=[1, ], axis=None)

horsepower_normalizer.adapt(np.array(X_train.Horsepower))

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    tf.keras.layers.Dense(units=1)
])

horsepower_model.summary()

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 normalization_1 (Normalizat  (None, 1)                3         
 ion)                                                            
                                                                 
 dense (Dense)               (None, 1)                 2         
                                                                 
=================================================================
Total params: 5
Trainable params: 2
Non-trainable params: 3
_________________________________________________________________
```

# Results

| |Mean absolute error [MPG]|
|-|-|
|horsepower_model|4.211570|
|linear_model|2.553518|
|model|2.284585|
|reloaded|2.284585|