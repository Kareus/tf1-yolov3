import os
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

all_path = '../data/annotation.txt'
train_path = '../data/train.txt'
val_path = '../data/val.txt'
test_path = '../data/test.txt'

allFile = open(all_path, 'r')
allFile_lines = allFile.readlines()
allFile.close()

dataset_size = len(allFile_lines)

train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = int(0.15 * dataset_size)

full_dataset = tf.data.TextLineDataset(all_path)
full_dataset = full_dataset.shuffle(dataset_size)
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)

train_dataset = list(train_dataset)
val_dataset = list(val_dataset)
test_dataset = list(test_dataset)

writer = open(train_path, 'w')
for data in train_dataset:
  writer.write(data.numpy().decode('ascii') + '\n')
writer.close()

writer = open(val_path, 'w')
for data in val_dataset:
  writer.write(data.numpy().decode('ascii') + '\n')
writer.close()

writer = open(test_path, 'w')
for data in test_dataset:
  writer.write(data.numpy().decode('ascii') + '\n')
writer.close()

print('train dataset counts:', len(train_dataset))
print('validation dataset counts:', len(val_dataset))
print('test dataset counts:', len(test_dataset))

reader = open(train_path, 'r')
train_file = reader.readlines()
reader.close()
train_file.sort()

writer = open(train_path, 'w')
for i in range(len(train_file)):
  writer.write(str(i) + ' ' + train_file[i])
writer.close()

reader = open(val_path, 'r')
val_file = reader.readlines()
reader.close()
val_file.sort()

writer = open(val_path, 'w')
for i in range(len(val_file)):
  writer.write(str(i) + ' ' + val_file[i])
writer.close()

reader = open(test_path, 'r')
test_file = reader.readlines()
reader.close()
test_file.sort()

writer = open(test_path, 'w')
for i in range(len(test_file)):
  writer.write(str(i) + ' ' + test_file[i])
writer.close()