import MyMnist
import numpy as np
from keras.models import model_from_json
import os
import config
import csv

json_name = 'nn.json'
weights_name = 'wieghts.h5'
test_path = 'C:\\Users\\Nadav\\Desktop\\OCR\\Test'

def main(path='C:\\NNModels\\4'):

    # Reads model data
    json_path = os.path.join(path, json_name)
    wieghts_path = os.path.join(path, weights_name)
    model = model_from_json(open(json_path).read())
    model.load_weights(wieghts_path)

    # Reads test data
    img, lbl = MyMnist.read(test_path)

    conf_mat = np.zeros([config.number_class, config.number_class], int)
    count = 0;

    # Predicts each test data and put it in confusion matrix
    for i in range(0,lbl.size):
        predicted = np.argmax(model.predict(np.array(img[i]).reshape((1,1,config.height,config.width))))
        print int(lbl[i]), int(predicted)
        conf_mat[int(lbl[i]), int(predicted)] += 1
        if (int(lbl[i]) == int(predicted)):
            count += 1

    # Writes confusion matrix to file
    with open(os.path.join(path, "conf_test.csv"), 'wb') as conf_file:
        writer = csv.writer(conf_file)
        writer.writerows(conf_mat)
    conf_file.close()

    # Prints accuracy
    print float(count) / float(lbl.size)

if __name__ == '__main__':
    main()