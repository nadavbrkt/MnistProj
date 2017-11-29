import cv2
import numpy as np
from visual import Visual
from SvmOutput import SvmOutput
import config
import Hog
import MyMnist
from collections import deque
import os
from sklearn.utils import shuffle

def main():
    svm_params = dict(svm_type=cv2.SVM_C_SVC, kernel_type=cv2.SVM_LINEAR, degree=None, gamma=5.383, C=1)

    # Creates visual platrorm
    vis = Visual()

    # User input ################################################################
    ans = '1' # raw_input("Would like to create model (1) or to view outputs (2) : ")

    if ans == '1':

        print "Reading data set : "
        raw_img, raw_lbl = MyMnist.read()
        print "Done \n"

        lbl_img = zip(raw_lbl, raw_img)
        lbl_img = shuffle(lbl_img)

        raw_lbl = np.asarray([t[0] for t in lbl_img])
        raw_img = np.asarray([t[1] for t in lbl_img]).reshape((raw_lbl.size, config.height * config.width))


        print "Now lets start training : \n"
        # User input ################################################################
        is_hog = 'y' # raw_input("Would you like to use Hog (y/n) : ")
        if (is_hog == 'y'):
            hog_scale_w = int(raw_input("What width would you like to use : "))
            hog_scale_h = int(raw_input("What height would you like to use : "))
            hog_degrees = int(raw_input("What amount of degrees would you like to use : "))

            hog_data = np.zeros(
                [raw_lbl.size, config.height * config.width / hog_scale_h / hog_scale_w * hog_degrees], 'float32')
            for i in range(0, raw_lbl.size):
                hog_data[i] = Hog.getimghog(raw_img[i].reshape((config.height,config.width)),
                                             [[1, 0, -1], [2, 0, -1], [1, 0, -1]],
                                             [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                             hog_scale_h, hog_scale_w, hog_degrees)

            raw_img = hog_data

        # User input ################################################################
        out = 2 # int(raw_input("Would you like to use Linear (1) or Kernel (2) :"))

        svm_params['C'] = float(raw_input("Please enter C parameter : "))
        if out == 1:
            svm_params['kernel_type'] = cv2.SVM_LINEAR
        else:
            svm_params['kernel_type'] = cv2.SVM_POLY
            svm_params['degree'] = int(2)# float(raw_input("Please enter SVM degree parameter : "))
            svm_params['gamma'] = float(raw_input("Please enter gamma : "))

        name = "HOG " + str(hog_scale_h) + "x" + str(hog_scale_w) + " d" + str(hog_degrees) + " C" + str(svm_params['C']) + " G" + str(svm_params['gamma']) #raw_input("How would you like to name your model : ")

        step = int(round(raw_lbl.size * 0.1)) - 1

        test_lbls = list()
        pred_lbls = list()

        for i in range(0,10):
            print "Start " + str(i) + " training ",
            train_img = np.concatenate((raw_img[:i,:], raw_img[i+step:,:]))
            train_lbl = np.append(raw_lbl[:i], raw_lbl[i+step:])
            test_img = raw_img[i:i + step,:]
            test_lbl = raw_lbl[i:i + step]

            test_lbls.append(test_lbl)

            svm = cv2.SVM()
            svm.train(train_img, train_lbl, params=svm_params)
            print "=> Predicting",
            pred_lbls.append(svm.predict_all(test_img))
            print "=> Done"

        s = SvmOutput(name, svm_params, svm, np.array(test_lbls).ravel(), np.array(pred_lbls).ravel())
        s.save()
        s.showdata()

    else:

        ans = raw_input("Would you like to view a specific model (1) \nor the roc curve of a few models (2) :")
        models_dir = os.walk(config.default_path).next()[1]

        if ans == '1':
            print "Available models : "
            for i in range(1, len(models_dir) + 1):
                print "\t(" + str(i) + ") " + models_dir[i - 1]

            ans = int(raw_input("Which model would you like to view : "))

            model = SvmOutput(name=models_dir[ans - 1], readfile=True)
            model.showdata()
        else:
            roc_models = list()
            ans = 'y'

            while (ans == 'y') & (len(models_dir) > 0):
                print "Available models : "
                for i in range(1, len(models_dir) + 1):
                    print "\t(" + str(i) + ") " + models_dir[i - 1]

                model_add = int(raw_input("Which model would you like to view : "))
                roc_models.append(SvmOutput(name=models_dir[model_add - 1], readfile=True))
                ans = raw_input("Would you like to choose another model (y/n) : ")
                models_dir.pop(model_add - 1)

            vis.showROC(roc_models)

if __name__ == "__main__":
    main()
