import argparse #argumentos via terminal
import json #salvar classes
import os #manipulação de arquivos
import numpy as np
from tensorflow.keras.applications import MobileNetV2 #modelo pré-treinado (ImageNet)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input #camadas que adaptam o modelo
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator #aumento de dados para evitar overfitting

# imagens serão redimensionadas
# processamento em lotes para otimizar uso de memória
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 5
DEFAULT_IMAGE = 'dataset/single_prediction/pizza/12718.jpg'
MODEL_PATH = 'models/pizza_notpizza_mobilenetv2.keras'
CLASS_INDEX_PATH = 'models/class_indices.json'


def build_model():
  # usa um modelo já treinado em milhões de imagens
  # remove a parte final (classificador original)
  base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights='imagenet',
  )
  # não treina as camadas iniciais, que já aprenderam a extrair características básicas
  base_model.trainable = False

  inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
  x = base_model(inputs, training=False)

  # reduz a saída da CNN para um vetor de características
  # evita muitos parâmetros e overfitting
  x = GlobalAveragePooling2D()(x)

  # desliga 30% dos neurônios
  # evita overfitting
  x = Dropout(0.3)(x)

  # saída binária (0 ou 1)
  # sigmoide → probabilidade de ser pizza (1) ou not_pizza (0)
  outputs = Dense(1, activation='sigmoid')(x)

  model = Model(inputs, outputs)

  model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy'],
  )
  return model, base_model


def create_data_generators():
  # aumenta a variedade de imagens para o modelo aprender melhor
  # preprocess_input: normaliza as imagens como o MobileNetV2 espera
  # as transformações ajudam o modelo a generalizar melhor, evitando overfitting
  train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
  )
  test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

  training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
  )
  test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
  )
  return training_set, test_set


def save_class_indices(class_indices):
  # salva o mapeamento de classes para usar na predição
  os.makedirs(os.path.dirname(CLASS_INDEX_PATH), exist_ok=True)
  with open(CLASS_INDEX_PATH, 'w', encoding='utf-8') as file:
    json.dump(class_indices, file)


def load_class_indices():
  if os.path.exists(CLASS_INDEX_PATH):
    with open(CLASS_INDEX_PATH, 'r', encoding='utf-8') as file:
      loaded_indices = json.load(file)
      return {name: int(index) for name, index in loaded_indices.items()}
  return {'not_pizza': 0, 'pizza': 1}


def train_and_save(model_path):
  training_set, test_set = create_data_generators()
  model, base_model = build_model()

  # EarlyStopping
  #    - para parar o treinamento se não houver melhora na acurácia de validação
  #    - evita overfitting

  # ReduceLROnPlateau
  #    - reduz a taxa de aprendizado se a perda de validação não melhorar
  #    - ajuda a ajustar o modelo finamente
  callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
  ]

  # treina apenas as camadas finais primeiro, para aprender a classificar pizza e not_pizza
  model.fit(
    training_set,
    epochs=INITIAL_EPOCHS,
    validation_data=test_set,
    callbacks=callbacks,
  )

  # depois treina mais camadas para refinar o modelo, mas mantém as camadas iniciais congeladas para evitar esquecer o que já aprenderam
  base_model.trainable = True
  for layer in base_model.layers[:-30]:
    layer.trainable = False

  # usa uma taxa de aprendizado menor para não destruir o que já foi aprendido
  model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy'],
  )

  # treina por mais algumas épocas para refinar o modelo, mas com cuidado para evitar overfitting
  model.fit(
    training_set,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=INITIAL_EPOCHS,
    validation_data=test_set,
    callbacks=callbacks,
  )

  # salva o modelo treinado para usar depois na predição
  os.makedirs(os.path.dirname(model_path), exist_ok=True)
  model.save(model_path)
  save_class_indices(training_set.class_indices)
  print(f'Modelo salvo em: {model_path}')

  return model, training_set.class_indices


def predict_image(model, class_indices, image_path):
  if not os.path.exists(image_path):
    raise FileNotFoundError(f'Imagem não encontrada: {image_path}')

  test_image = image.load_img(image_path, target_size=IMAGE_SIZE)
  test_image = image.img_to_array(test_image)
  test_image = preprocess_input(test_image)
  test_image = np.expand_dims(test_image, axis=0)

  score = float(model.predict(test_image, verbose=0)[0][0])
  predicted_class = 1 if score >= 0.5 else 0
  class_by_index = {index: name for name, index in class_indices.items()}
  prediction = class_by_index[predicted_class]

  print(f'Predição: {prediction} (score={score:.4f})')
  return prediction


def main():
  parser = argparse.ArgumentParser(description='Pizza vs Not Pizza: treino e predição')
  parser.add_argument(
    '--mode',
    choices=['train', 'predict'],
    default='predict',
    help='train: treina e salva modelo | predict: apenas predição com modelo salvo',
  )
  parser.add_argument(
    '--image',
    default=DEFAULT_IMAGE,
    help='caminho da imagem para predição',
  )
  parser.add_argument(
    '--model-path',
    default=MODEL_PATH,
    help='caminho do arquivo do modelo salvo (.keras)',
  )
  args = parser.parse_args()

  if args.mode == 'train':
    model, class_indices = train_and_save(args.model_path)
    predict_image(model, class_indices, args.image)
    return

  if not os.path.exists(args.model_path):
    print(f'Modelo não encontrado em {args.model_path}. Execute primeiro com --mode train.')
    return

  model = load_model(args.model_path)
  class_indices = load_class_indices()
  predict_image(model, class_indices, args.image)


if __name__ == '__main__':
  main()

# python3 app.py --mode train

#python3 app.py --mode predict --image dataset/single_prediction/pizza/12718.jpg