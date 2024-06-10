from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Load the pre-trained model
model = load_model("speech.h5")

# Encode labels
label_encoder = OneHotEncoder()
y_encoded = label_encoder.fit_transform(df[['label']]).toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert one-hot encoded predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Create confusion matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

# Display confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
