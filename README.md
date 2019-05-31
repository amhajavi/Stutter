### Dependencies
- [Python_3](https://www.continuum.io/downloads)
- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)

### Training the model
To train the model on the Voxceleb2 dataset, you can run

- python src/main.py --net resnet34s --batch_size 160 --gpu 2,3 --lr 0.001 --optimizer adam --epochs 48 --multiprocess 8 --loss softmax --data_path ../path_to_voxceleb2

