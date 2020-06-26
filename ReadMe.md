# tf1-yolov3


### 1. Introduction
This is a yolov3 implementation in pure tensorflow v1, which based on [git link](https://github.com/wizyoung/YOLOv3_TensorFlow).

I modified a little for my project in college. (it was the dog detection project)

As this is my first try on tensorflow, there may be some errors or mistakes on code.



- changes

  - used TFRecordDataset instead of TextLineDataset.

  - fixed saver to save the parameter only when save-optimizer option is true

  - changed parameter 'mode' to bool value 'is_training' in data util functions

    (because it causes error due to byte string recognition)

  - wrote TFRecord binary iterator, which runs without tf session

  - removed tensorboard and summary code (Maybe there are some unused variables...)

    

### 2. Preparation

##### 2.1 TFRecord files

You would need to prepare TFRecord dataset file for training.

Convert annotation xml (Pascal VOC format) files to txt files, using `code/annotation_xml_parser.py`

Text files are in the format `image_index image_path width height box_array`, and each box is like `box_index xmin ymin xmax ymax`.

And you can create TFRecord file with `code/tfrecord_writer.py`



As the project needs datasets for train/eval/test procedure, you can split the TFRecord file using `code/dataset_splitter.py`

All the result files will be placed in `data` directory, if default options not changed.



##### 2.2 Anchor

You would need Yolov3 anchor file for training the dataset.

`code/yolov3/get_kmeans.py` will print out anchors to the console screen.

Save and place it to `data/yolo_anchors.txt` 

or you can change the path, and change the variable value, `anchor_path` , in `code/yolov3/args.py` , `code/yolov3/eval.py`, etc.



##### 2.3 pretrained model

You should place pretrained Tensorflow checkpoint file in `data/darknet_weights/` directory. (check `restore_path` in `code/yolov3/args.py`)

You can download darknet weights file [here](https://pjreddie.com/media/files/yolov3.weights), and convert darknet weights to Tensorflow checkpoint file using `code/yolov3/convert_weights.py`.



### 3. Train & Test

- Training

  You can train the model with:

  ``` python train.py```

  Check `code/yolov3/args.py` for details.

  These codes did not change much from the original code.
  

- Evaluating

  You can evaluate the model with:

  `python eval.py`

  Check the file for details.

  Evaluation options are also not differenct from the original. I just added some default values.

  

- Test with image & video

  Test images with:

  ```python test_single_image.py --input_image=test.png```

  I changed ```input_image``` to ```--input_image``` to run directly on VSCode, but that would not be a problem.

  Default value of ```restore_path``` is the pretrained yolov3 model.

  So, you need to change the path if you want to test your trained model.



​		Test videos with:

​		```python video_test.py --input_video=test.mp4```

​		Same with ```test_single_image.py```

​		

### 4. PB files (extra)

You can create pb file from your model, using ```code/yolov3/pbCreator.py```.

The code is slightly modifed version of ```test_single_image.py```.

The input/output node name from the code would be

```input_data``` and ```[yolov3/yolov3_head/feature_map_1,yolov3/yolov3_head/feature_map_2,yolov3/yolov3_head/feature_map_3]```.



If you want to freeze model, like android implementation, use ```code/yolov3/freeze_pb.py```



##### Notes

Here are some notes for people working on yolov3 android implementation.

I tested the model with tensorflow android example project and encountered lots of troubles.

- tensorflow-android library supports up to v1.13.1:

  As this project code runs on tensorflow v1, not v2, you would not be able to use tflite.

  So, you have to train or create pb file with tensorflow which version <= 1.13.1.

  In my case, I already trained the model with v1.15, so I tried to create pb file with the library with downgraded version.

  I succeeded to create the file as aresult, and implemented in android but I don't know the file is valid.

  

- yolov3 model is too large to work on android:

  I ran the application but it did not detect on real-time, I had to wait still for detection boxes to pop up,

  and even the result didn't come out very well.

  I did not try (actually, my teammate. he could not follow the process, lol) some optimization work for the model,

  but it would be better to try on yolov3-tiny instead of yolov3 model.



&nbsp;

#### Reference
---

[1]  https://github.com/wizyoung/YOLOv3_TensorFlow for entire yolov3 code

[2]  https://github.com/pgmmpk/tfrecord for TFRecord iterator without running a session