import numpy as np
import os
from PIL import Image
import glob

dir = "./dataset/chinese"


def getFileArr(dir):
    result_arr = []
    label_list = []
    map = {}
    map_file_result = {}
    map_file_label = {}
    map_new = {}
    count_label = 0
    count = 0

    file_list = os.listdir(dir)
    for file in file_list:
        file_path = os.path.join(dir, file)

        label = file.split(".")[0].split("_")[1]
        map[file] = label
        if label not in label_list:
            label_list.append(label)
            map_new[label] = count_label
            count_label = count_label + 1
        img = Image.open(file_path)
        result = np.array(img)

        result = result.reshape((3, 64, 64))
        result = result / 255.0
        map_file_result[file] = result
        result_arr.append(result)
        count = count + 1
    for file in file_list:
        map_file_label[file] = map_new[map[file]]
        # map[file]=map_new[map[file]]

    ret_arr = []
    for file in file_list:
        each_list = []
        label_one_zero = np.zeros(count_label)
        result = map_file_result[file]
        label = map_file_label[file]
        label_one_zero[label] = 1.0
        # print(label_one_zero)
        each_list.append(result)
        each_list.append(label_one_zero)
        ret_arr.append(each_list)
    # os.makedirs("F:/Python project/GANMNIST/dataset")
    np.save('F:/Python project/GANMNIST/dataset/train_data.npy', ret_arr)
    return ret_arr


def getnpy(url):
    data=np.load(url)
    X=data[:,0]
    imgs=[]
    for sample in X:
        imgs.append(np.array(sample))
    imgs=np.array(imgs)
    return imgs


if __name__ == '__main__':
    ret_arr = getFileArr(dir)
