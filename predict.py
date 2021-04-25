from tensorflow.keras.models import load_model
from utils import encoder_data, decoder_data, add_noise
import numpy as np

sentence = 'helloo tất cả các bạn'

sentence_error = add_noise(sentence, 0.9, 0.985)

model = load_model('./checkpoint/spell_0.99.h5')

input_model = encoder_data(sentence_error)
print(sentence_error, input_model.shape)

result = model.predict(np.array([input_model]))[0]

print(decoder_data(result))
