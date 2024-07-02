import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt

# Define path of  dataset
csv_filename = "../Code_Dataset/java_errors_dataset.csv"
df = pd.read_csv(csv_filename)

# Preprocess dataset
def preprocess_code(code):
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code
df['sampleCode'] = df['sampleCode'].apply(preprocess_code)


# Tokenize the code
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['sampleCode'])
sequences = tokenizer.texts_to_sequences(df['sampleCode'])
word_index = tokenizer.word_index

maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=maxlen, padding='post')

# Converting error types to numeric labels
error_types = df['typeofError'].unique()
error_type_to_index = {error_type: idx for idx, error_type in enumerate(error_types)}
y = df['typeofError'].map(error_type_to_index).values

# define  LSTM algorithm
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=64, input_length=X.shape[1]),
    LSTM(128),
    Dense(len(error_types), activation='softmax')
])

# Train the model with LSTM algorithm
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X, y, epochs=800, batch_size=2, validation_split=0.2)


# Save the model trainned
model.save("../java_error_detection_model.h5")
print("Model saved to java_error_detection_model.h5")


# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()