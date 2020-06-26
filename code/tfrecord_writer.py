import tensorflow as tf

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def readImage(path):
    file = open(path, 'rb')
    byte = file.read()
    return byte

def main():
    ANNOTATION_PATH = '../data/train.txt' #annotation set (train/val/test) text file
    IMAGE_DIRECTORY = 'image_data' #image directory
    SAVE_PATH = 'train.tfrecord' #save path for tfrecord

    print('Tensorflow version:', tf.__version__) #tensorflow version should be 1.x

    file = open(ANNOTATION_PATH, 'r')
    lines = file.readlines()
    file.close()

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP) #compress option
    writer = tf.python_io.TFRecordWriter(path=SAVE_PATH, options=options)

    for line in lines:
        parsed = line.split(' ')
        print('Current Doing...', parsed[1]) #debug messages
        image = readImage(IMAGE_DIRECTORY + '/' + parsed[1])
        boxes = []

        for i in range(4, len(parsed)):
            boxes.append(int(parsed[i]))
        
        data = tf.train.Example(features=tf.train.Features(feature={
            'index': int64_feature(int(parsed[0])),
            'image': bytes_feature(image),
            'width': int64_feature(int(parsed[2])),
            'height': int64_feature(int(parsed[3])),
            'boxes': int64_feature(boxes) # boxes = [label1, xmin1, ymin1, xmax1, ymax1, label2, xmax2, ...]
        }))

        writer.write(data.SerializeToString())

    writer.close()

if __name__ == '__main__':
    main()