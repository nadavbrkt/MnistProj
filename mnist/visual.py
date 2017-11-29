import numpy as np
import config
from matplotlib import pyplot as plt
import os


class Visual:
    def __init__(self):
        self.images = []

    def createimages(self, raw_images, h=config.height, w=config.width):
        # Create empty image array
        self.images = np.zeros((raw_images.shape[0], h, w))

        # Reads each image
        for im_number in range(0, raw_images.shape[0]):
            for i in range(0, h):
                for j in range(0, w):
                    self.images[im_number][i][j] = raw_images[im_number, ((i * w) + j)]

    def showimage(self, i):
        # Show specific image
        plt.imshow(self.images[i], 'Greys')
        plt.show()

    def showconfmatrix(self, conf, title='conf', modelname=''):

        # Creates normalized conf matrix
        norm_conf = []
        for i in conf:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                if a == 0:
                    tmp_arr.append(0)
                else:
                    tmp_arr.append(float(j) / float(a))
            norm_conf.append(tmp_arr)

        conf_fig = plt.figure()
        plt.clf()
        ax = conf_fig.add_subplot(111)

        # Show confusion matrix colors
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                        interpolation='nearest')

        width = len(conf)
        height = len(conf[0])

        # Sets data in each
        for x in xrange(width):
            for y in xrange(height):
                ax.annotate(str(conf[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

        cb = conf_fig.colorbar(res)

        # Sets Axes
        alphabet = range(0, config.number_class)
        plt.xticks(range(width), alphabet[:width])
        plt.yticks(range(height), alphabet[:height])
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
        plt.title(title)
        plt.show()
        conf_fig.savefig(os.path.join(config.default_path, modelname, title + ".png"))

    def showROC(self, tests=None):
        # Creates roc
        if not tests:
            tests = []
        fig = plt.figure(figsize=(7, 7), dpi=80)
        plt.xlabel("FP Rate")
        plt.ylabel("Accuracy")
        plt.title("ROC Curve all")

        names = list()
        fpr = np.zeros([np.size(tests)])
        tpr = np.zeros([np.size(tests)])

        # For each test data put in plot
        i = 0
        for test in tests:
            fpr[i] = test.data['fp_rate_all']
            tpr[i] = test.data['accuracy_all']
            plt.annotate(test.data['model_name'], xycoords='data',
                         xy=(test.data['fp_rate_all'], test.data['accuracy_all']),
                         xytext=(-20, -20), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->"))
            i += 1

        names.append('')

        plt.xlim(0.0, fpr.max() * 1.1)
        plt.ylim(tpr.min() * 0.9, 1.0)

        plt.savefig('c:\SvmModels\ROC.png')
        plt.show()
