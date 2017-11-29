import numpy as np
import cv2
import os
import config
import json
import csv
import visual


class SvmOutput:
    data = {
        'model_name': '',
        'model_data': '',
        'accuracy': '',
        'fp_rate_test': '' }
    conf_mat = ''
    model = ''

    def __init__(self, name, model_data='', model='', test_labels='',
                 predicted_data='', no_class=config.number_class, readfile=False):

        if readfile:
            # Reads svm data
            read_data = open(os.path.join(config.default_path, name, "data.json"), 'r')
            self.data = json.load(read_data)
            read_data.close()

            self.model = cv2.SVM()
            self.model.load(os.path.join(config.default_path, name, "model.xml"))
            self.data['accuracy'] = float(self.data['accuracy'])
            self.data['fp_rate_test'] = float(self.data['fp_rate_test'])

            # Reads confusion matrix csv
            with open(os.path.join(config.default_path, name, "conf_test.csv"), 'rb') as conf_file:
                reader = csv.reader(conf_file)
                self.conf_mat = np.zeros([config.number_class, config.number_class], int)
                row_n = 0
                for row in reader:
                    cell_n = 0
                    for cell in row:
                        self.conf_mat[row_n, cell_n] = int(cell)
                        cell_n += 1
                    row_n += 1
            conf_file.close()

        else:
            # Creates confusion matrix
            self.conf_mat = np.zeros([no_class, no_class], int)
            for i in range(0, np.size(predicted_data)):
                self.conf_mat[test_labels.item(i), predicted_data.item(i)] += 1


            # Sets data
            self.data['model_name'] = name
            self.data['model_data'] = model_data
            self.model = model
            self.data['number_class'] = no_class
            self.data['accuracy'] = float(sum(self.conf_mat.diagonal(), 0) / float(sum(sum(self.conf_mat, 0), 0)))

            # Finds fp rate
            fp_per_class = list()
            fp_per_class_all = list()
            for i in range(0, no_class):
                if (sum(self.conf_mat[:, i])) == 0:
                    fp_per_class.append(0)
                else:
                    fp_per_class.append(float(sum(self.conf_mat[:, i], 0) - self.conf_mat[i, i]) /
                                        float(sum(self.conf_mat[:, i])))

            self.data['fp_rate_test'] = np.average(fp_per_class)

    def save(self):
        # Save svm data into directory
        mkdir = os.path.join(config.default_path, self.data['model_name'])
        os.mkdir(mkdir)

        # Writes json data
        write_to = open(os.path.join(mkdir, "data.json"), 'w')
        json.dump(self.data, write_to, indent=4, separators=(',', ': '))
        write_to.close()

        # Writes confusion matrix
        with open(os.path.join(config.default_path, self.data['model_name'], "conf_test.csv"), 'wb') as conf_file:
            writer = csv.writer(conf_file)
            writer.writerows(self.conf_mat)
        conf_file.close()

        self.model.save(os.path.join(mkdir, "model.xml"))

    def showdata(self):
        # Show confusion matrix
        vis = visual.Visual()
        vis.showconfmatrix(self.conf_mat, "Test Confusion", self.data['model_name'])
