import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(range(1,len(model_history.history[acc])+1),model_history.history[acc])
    axs[0].plot(range(1,len(model_history.history[val_acc])+1),model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history[acc])+1),len(model_history.history[acc])/10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

class Confusion_Matrix():
    def __init__(self, model, dataset_test):
        loss, accuracy = model.evaluate_generator(dataset_test)
        print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))
        y_pred = model.predict_generator(dataset_test)
        self._y_p = np.where(y_pred > 0.5, 1, 0)
        test_data = dataset_test.unbatch()
        self._y_g = []
        for image, label in test_data:
            self._y_g.append(label.numpy())

    def showmatrix(self):
        confusion_mtx = confusion_matrix(self._y_g, self._y_p)
        # plot the confusion matrix
        f, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Blues", linecolor="gray", fmt='.1f', ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

    def get_classification_report(self):
        report = classification_report(self._y_g, self._y_p, target_names=['0', '1'])
        return report