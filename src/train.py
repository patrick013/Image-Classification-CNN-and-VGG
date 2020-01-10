from .models import CNNmodel, VGGmodel
from .dataset import LoadByDataframe, LoadByGenerator


EPOCHS=20

datasetloader=LoadByGenerator()
training_set,validation_set=datasetloader.load_dataset()
model=VGGmodel()
hist=model.fit_generator(training_set,epochs=EPOCHS,validation_data=validation_set)