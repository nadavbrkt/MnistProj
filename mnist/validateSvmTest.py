import MyMnist
from SvmOutput import SvmOutput
import os
import numpy as np
import config
import Hog
import csv
test_path = 'C:\\Users\\Nadav\\Desktop\\OCR\\Test'

def main(path='C:\\NNModels\\21'):
    os.chdir('C:\\SvmModels\\K-Fold\\HOG\\Kernel')
    model = SvmOutput("HOG 7x7 d36 C11.0 G1000000.0",readfile=True)
    model = model.model

    img, lbl = MyMnist.read(test_path)
    conf_mat = np.zeros([config.number_class, config.number_class], int)

    count = 0;
    for i in range(0,lbl.size):
        tmp = Hog.getimghog(img[i],[[1, 0, -1], [2, 0, -1], [1, 0, -1]],
                            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                            7,7,36)
        predicted = model.predict(tmp)
        print int(lbl[i]), int(predicted)
        conf_mat[int(lbl[i]), int(predicted)] += 1
        if (int(lbl[i]) == int(predicted)):
            count += 1

    with open(os.path.join(path, "conf_test.csv"), 'wb') as conf_file:
        writer = csv.writer(conf_file)
        writer.writerows(conf_mat)
    conf_file.close()

    print float(count) / float(lbl.size)

if __name__ == '__main__':
    main()