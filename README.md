### Dependencies
- [Python_3](https://www.continuum.io/downloads)
- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)

### Training the model
To train the model on the Voxceleb2 dataset, you can run

- python main.py --spec_len 50 --gpu 0 --lr 0.0001 --person [the number for each person to test e.g. 20] --batch_size 20 --ohem_level 2
