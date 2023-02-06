import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from Model1 import Model1
import itertools 
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np


NUM_EPOCHS=60; INIT_LR=0.001; BS=32

train_path = '../Test_Technique_Image/Neuroflux_disorder_splitted/train'
valid_path = '../Test_Technique_Image/Neuroflux_disorder_splitted/val'
test_path = '../Test_Technique_Image/Neuroflux_disorder_splitted/test'

totalTrain = len(list(paths.list_images(train_path)))
print(totalTrain)
totalVal = len(list(paths.list_images(valid_path)))
totalTest = len(list(paths.list_images(test_path)))

trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")

valAug = ImageDataGenerator(
    rescale=1 / 255.0)
testAug = ImageDataGenerator(
    rescale=1 / 255.0)


train_batches = trainAug.flow_from_directory(directory=train_path, class_mode="categorical",target_size=(256,256), color_mode="rgb", classes=['EO','IO','IPTE','LO','PTE'], shuffle=True,batch_size=32)
valid_batches = valAug.flow_from_directory(directory=valid_path,class_mode="categorical",target_size=(256,256), color_mode="rgb", classes=['EO','IO','IPTE','LO','PTE'], batch_size=32,shuffle=False)
test_batches = testAug.flow_from_directory(directory=test_path, class_mode="categorical",target_size=(256,256), color_mode="rgb", classes=['EO','IO','IPTE','LO','PTE'], batch_size=32 , shuffle=False)

trainLabels = train_batches.classes
trainLabels = np_utils.to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = dict()
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]




model=Model1.build(width=256,height=256,depth=3,classes=5)

filepath = "Model_1.h5"
checkpoint = ModelCheckpoint( filepath , monitor = 'val_acc', verbose =1, mode = 'max' )
callbacks_list = [checkpoint]
opt=Adam(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


print("[INFO] training ...")
M=model.fit(
	x=train_batches,
	steps_per_epoch=totalTrain // BS,
	validation_data=valid_batches,
	validation_steps=totalVal // BS,
    class_weight=classWeight,
	epochs=NUM_EPOCHS,
    callbacks = callbacks_list)

print("Now evaluating the model")
test_batches.reset()
pred_indices=model.predict(test_batches,steps=(totalTest//BS)+1)

pred_indices=np.argmax(pred_indices,axis=1)

print(classification_report(test_batches.classes, pred_indices, target_names=test_batches.class_indices.keys()))

cm=confusion_matrix(test_batches.classes,pred_indices)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_matrix.png')
    print('The confusion matrix is saved in the file: confusion_matrix.png  ')
	


cm_plot_labels = ['EO','IO','IPTE','LO','PTE']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')



N = NUM_EPOCHS

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), M.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), M.history["val_loss"], label="val_loss")
plt.title("Training Loss on the dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig('training_loss.png')
print('The training loss in saved in the file: training_loss.png  ')



# plot the training accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), M.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), M.history["val_accuracy"], label="val_acc")
plt.title("Training accuracy on the dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.savefig('training_accuracy.png')
print('The training accuracy is saved in the file: training_accuracy.png ')


