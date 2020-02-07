# Statoil/C-CORE Iceberg Classfier Challenge

https://www.kaggle.com/c/statoil-iceberg-classifier-challenge

## Description

바다 위에 떠다니는 빙하는 위협적인 존재입니다. 타이나닉 호도 빙하에 부딪혀 침몰했으니 말이죠. 빙하가 어디에 위치해있는지 판단할 수 있다면 이런 사고를 막을 수 있을 겁니다. 본 대회는 위성으로 찍은 사진을 기반으로 그 물체가 빙하인지 배인지 분류하는 대회입니다. (Statoil과 C-CORE는 회사 이름입니다.)

데이터의 속성(feature)는 `band_1`, `band_2`, `inc_angle` 로 단 3가지입니다. `band_1`과 `band_2` column은 한 element 당 75x75 = 5,625개의 float값으로 구성되어 있습니다. 이는 75x75의 pixel값으로 이미지 데이터라고 보시면 되겠습니다. 

본 대회의 주요 커널에서는 대부분 딥러닝 CNN(합성곱 신경망)을 활용하여 결과를 예측했습니다.

## 1st Kernel: Exploration & Transforming

본 커널에서는 실제 예측은 하지 않고, 빙하와 배 이미지를 그려봅니다. 우선, `imshow`를 활용하여 빙하와 배 이미지를 있는 그대로 먼저 그렸습니다. 그리고 몇가지 Transforming 기법을 적용한 이미지도 그려봤습니다. Transforming이란 카메라 필터처럼 필터를 적용한 이미지를 도출하는 것을 의미합니다. 사용한 Transforming 기법은 smooth, X-derivative, Gradient (= x-derivative + y-derivative), second derivative, Laplacian (= sum of second derivative)입니다.

**smooth Transforming**

아래와 같이 `signal.convolve2d`를 사용하여 `smooth`필터를 적용한 transforming을 할 수 있습니다.

```python
icebergs = train[train.is_iceberg==1].sample(n=9, random_state=123)
smooth = np.array([[1,1,1],[1,5,1],[1,1,1]])

signal.convolve2d(np.reshape(np.array(icebergs.iloc[i,1]),(75,75)),smooth, mode='valid')
```

 [(Reference1)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html)

## 2nd Kernel: Keras Model for Beginners

본 커널에서는 가장 기본적인 합성곱 신경망(Convolutional Neural Network, CNN)을 적용하여 결과를 예측했습니다. CNN의 기초에 대해 배울 수 있는 좋은 커널이었습니다. 프레임워크는 Keras를 사용했습니다.

**Sequential API**

Keras를 활용하여 Model을 만드는 방법은 두가지가 있습니다. 하나는 `Sequential` API를 사용하는 것이고 하나는 함수형 API인 `Model` 함수를 사용하는 것입니다. Sequential API는 계층별로 층층이 (layer-by-layer) 연결하여 쉽게 사용할 수 있다는 장점이 있지만 다중입출력(multiple input, multiple output)이 불가능하다는 단점이 있습니다. 단일 입출력을 갖는 대부분의 딥러닝 모델은 `Sequential`로 구현이 가능하지만 다양한 변형을 위해서는 함수형 API를 사용하는 것이 좋습니다. 아래는 `Sequential` API를 사용한 예제입니다. [(Reference2)](https://jovianlin.io/keras-models-sequential-vs-functional/)

```python
def getModel():
    gmodel = Sequential()
    
    # Convolutional Layer 1
    gmodel.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(75, 75, 3)))
    gmodel.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    gmodel.add(Dropout(0.2))
    
    # Flatten the data for upcoming dense layers
    gmodel.add(Flatten())

    #Dense Layers 1
    gmodel.add(Dense(512))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    #Sigmoid Layer
    gmodel.add(Dense(1))
    gmodel.add(Activation('sigmoid'))
    
    myoptim = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss='binary_crossentropy',
                  optimizer=myoptim,
                  metrics=['accuracy'])
    return gmodel
```

보시는 바와 같이 처음에 `Sequential()` 모델을 먼저 만든 뒤 해당 모델에 `.add()`를 통해 여러 계층(Layer)를 층층이 연결해주면 됩니다.

## 3rd Kernel: A Keras Prototype

2번째 커널인 Keras Model for Beginners와 큰 흐름은 유사합니다. 가장 큰 차이점은 Keras로 CNN 모델을 만드는 방법입니다. 본 커널은 함수형 API를 사용해서 CNN 모델을 만듭니다.

**Functional API**

```python
def get_model():
    bn_model = 0
    p_activation = "elu"
    input_1 = Input(shape=(75, 75, 3), name="X_1")
    input_2 = Input(shape=[1], name="angle")
    
    img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_1))
    img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = GlobalMaxPooling2D() (img_1)
    
    img_2 = Conv2D(128, kernel_size = (3,3), activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_1))
    img_2 = MaxPooling2D((2,2)) (img_2)
    img_2 = Dropout(0.2)(img_2)
    img_2 = GlobalMaxPooling2D() (img_2)
    
    img_concat =  (Concatenate()([img_1, img_2, BatchNormalization(momentum=bn_model)(input_2)]))
    
    dense_layer = Dropout(0.5) (BatchNormalization(momentum=bn_model) ( Dense(256, activation=p_activation)(img_concat) ))
    output = Dense(1, activation="sigmoid")(dense_layer)
    
    model = Model([input_1,input_2],  output)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model
```

함수형 API는 `Sequential`과 다르게 다중입출력(Multiple input 및 Multiple ouput)이 가능합니다. 맨 처음 `Input`을 정의해주고, 마지막에 `Output`을 만들어 준 뒤 `Model`함수에 `input`과 `output`을 넣어줘 Model을 만들었습니다. 위 예제에서는 input이 두개입니다.

**Sequential Model vs Functional Model**

아래는 두 모델 간의 차이를 나타내는 간략한 코드입니다.

```python
# Sequential Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(1))
```

```python
# Functional Model
from kears.models import Model
from kears.layers import Input, Dense

input = Input(shape=(2,))
output = Dense(2)(input)

# Create the model
model = Model(inputs=visible, outputs=output)
```

## 4th Kernel: Keras+TF

본 커널에서 특이한 점은 이미지 복구와 ImageDataGenerator를 사용했다는 점입니다. 이미지 복구 모듈을 사용해 노이즈를 제거해줬고, ImageDataGenerator를 사용해 이미지 데이터를 다양하게 변형시켜 훈련해줬습니다.

**이미지 복구**

이미지 복구 모듈인 `skimage.restoration`를 활용하여 노이즈를 제거했습니다. [(Reference3)](https://scikit-image.org/docs/dev/api/skimage.restoration.html)

**ImageDataGenerator**

Keras의 ImageDataGenerator는 이미지 데이터를 다양하게 변형시켜 학습 데이터를 늘려줍니다. 공식문서를 참고하시면 아시겠지만 주어진 input 이미지 데이터를 좌우반전, 상하반전, 회전, 이동을 시켜 새로운 데이터를 만들어주는 것입니다. 

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
```

[(Reference4)](https://keras.io/preprocessing/image/)

`ImageDataGenerator.flow`는 다양하게 변형하여 새로 만들어진 이미지 데이터에 대한 Iterator를 반환합니다. 이 Iterator는 Input과 Label값을 가지고 있고, 한번에 batch_size만큼 변형된 이미지 데이터를 반환합니다. `fit_generator`는 Iterator에서 반환한 이미지만큼 계속 학습시킵니다. `epoch`은 전체 훈련 데이터 셋에 대한 학습 반복 횟수입니다. `steps_per_epoch`은 한번의 epoch마다 몇번의 batch를 iterator로 반환시킬지를 의미합니다. 예를들어, x_train 훈련 데이터 수가 128개이고, batch_size=32, steps_per_epoch=4 라고 하면 iterator에서 데이터를 반환할 때 32개씩 반환하고 총 4번 반복한다는 뜻입니다. 즉, 한번의 epoch마다 사용되는 훈련 데이터 수는 steps_per_epoch * batch_size입니다.

```python
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
```

위 코드를 풀어쓰면 아래와 같습니다. epochs만큼 루프를 돌며 `datagen.flow` Iterator에서 반환된 이미지 데이터를 이용해 훈련을 합니다. (아래 코드는 공식 문서에 있는 코드입니다.)

```python
# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```

### References

[Reference1: scipy.signal.convolve2d Document](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html)

[Reference2: Keras Models Sequential vs Functional](https://jovianlin.io/keras-models-sequential-vs-functional/)

[Reference3: scikit-image restoration Document](https://scikit-image.org/docs/dev/api/skimage.restoration.html)

[Reference4: Keras Image Preprocessing Document](https://keras.io/preprocessing/image/)

1st Kernel: https://www.kaggle.com/muonneutrino/exploration-transforming-images-in-python

2nd Kernel: https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d

3rd Kernel: https://www.kaggle.com/knowledgegrappler/a-keras-prototype-0-21174-on-pl

4th Kernel: https://www.kaggle.com/wvadim/keras-tf-lb-0-18