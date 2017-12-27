from lxml import objectify, etree
import os
from scipy.io import savemat
from scipy.ndimage import imread

dataset_dir = '/data/adoptimal/hand_dataset/detectiondata/train/'
dataset_train = os.path.join(dataset_dir, 'pos')
dataset_bounde = os.path.join(dataset_dir, 'posGt')
dataset_voc = os.path.join(dataset_dir, 'anns')


def read_images_dataset(folder_name):
    list_images = os.listdir(folder_name)
    for image_path in list_images:
        dic = {}
        dic["folder"] = "train"
        file_name = image_path.split(".")[0]
        # image_path = os.path.join(dataset_train, image_path)
        print(image_path)
        dic["filename"] = image_path
        image_bound = os.path.join(dataset_bounde, file_name + ".txt")
        with open(image_bound, 'r') as file:
            for line in file:
                print(line)
                line = line.split(" ")
                if len(line) < 5:
                    continue
                dic["x_bound"] = line[1]
                dic["y_bound"] = line[2]
                dic["width_bound"] = line[3]
                dic["height_bound"] = line[4]

        images = imread(image_path)
        dic["width"], dic["height"], dic["depths"] = images.shape

        E = root(dic)
        etree.ElementTree(E).write(os.path.join(dataset_voc,'{}.xml'.format(file_name)))


def root(dic):
    E = objectify.ElementMaker(annotate=False)
    return E.annotation(
        E.folder(dic["folder"]),
        E.filename(dic["filename"]),
        E.source(
            E.database('VIVA Dataset'),
            E.annotation('VIVA Dataset'),
            E.image('Collections'),
        ),
        E.size(
            E.width(dic["width"]),
            E.height(dic["height"]),
            E.depth(dic["depths"]),
        ),
        E.segmented(0),
        E.object(
            E.name('hand'),
            E.bndbox(
                E.xmin(dic["x_bound"]),
                E.ymin(dic["y_bound"]),
                E.xmax(dic["x_bound"] + dic["width_bound"]),
                E.ymax(dic["y_bound"] + dic["height_bound"]),
            ),
        )
    )


if __name__ == '__main__':
    read_images_dataset(dataset_train)
