import MyMnist
from SvmOutput import SvmOutput
import os
import Hog


def main(modelpath='C:\\SvmModels\\K-Fold\\HOG\\Kernel',test_path = 'C:\\Users\\Nadav\\Desktop\\1\\1'):

    # Change dir to model dir and create model
    os.chdir(modelpath)
    model = SvmOutput("HOG 7x7 d36 C11.0 G1000000.0",readfile=True)
    model = model.model

    # Reads images and predicts them
    images = list()
    for i in os.listdir(test_path):
        img = MyMnist.create_img(os.path.join(test_path,i))
        img = Hog.getimghog(img,[[1, 0, -1], [2, 0, -1], [1, 0, -1]],
                            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                            7,7,36)
        prediction = model.predict(img)
        print i ,prediction

if __name__ == '__main__':
    main()