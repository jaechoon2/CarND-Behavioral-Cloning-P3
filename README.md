#**Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/data_augmentation_before.png "Histogram of Steering Angles Before Data Augmentation"
[image2]: ./examples/data_augmentation_after.png "Model Visualization"
[image_recovery1]: ./examples/recovery1.jpg "Recovery Image 1"
[image_recovery2]: ./examples/recovery2.jpg "Recovery Image 2"
[image_recovery3]: ./examples/recovery3.jpg "Recovery Image 3"
[image_right]: ./examples/right_image.png "Normal Image"
[image_right_flipped]: ./examples/right_image_flipped.png "Flipped Image"
[image_nvidia_model]: ./examples/nvidia_model.png "Nvidia model"
[image_nvidia_model_tf]: ./examples/nvidia_model_tf.png "Nvidia model (Keras)"
[image_center]: ./examples/center_image.png "Center Image"

---

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I adapted the [Nvidia's End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) model (a.k.a. Nvidia model) with slight modification by adding ReLU activation on each layer. I chose this model because Nvidia model is well documented, proven and easy to implement.

![alt text][image_nvidia_model]


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 134). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 148).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the [Udacity's Sample Training Data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) for initial training. After observing that the vehicle was not able to 
recover in track 1, I created recovery data using the simulator by recovering from off track to the center lane. In addition, I drove Track 2 so that the model can handle both Track 1 and Track 2.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My solution design approach was evolved through multiple iterations, starting from leveraging the proven Nvidia model with Udacity's sample training data using Jupyter notebook. Once I verified that the training is working with small epochs (~10), I created the model.py from the code from Jupyter notebook, and tested the drive.py with the simulator. 

After multiple iterations, I came up with the following key design approaches:
1. Leverage all images from left, center and right cameras by applying the steering offset of 0.3 for left and right camera.
2. Prepare each image by cropping the top 1/3 image which does not contain the road information.
3. Split the data by 80:20 ratio -- 80% for training data and 20% for validation data.
4. Adjust the distribution of the steering angles to form Gaussian distribution.
5. Augment dataset through random brightness, shadows and jiggles so that the training encounters these random scenarios.
6. Tweak the batch size, nb_epoch and sample_per_epoch to find the "good enough" parameter values without needing to spend too long time for training.
7. Let the car drive around tracks and observe the behavior.
8. Create training data by manually driving the car with a joystick for recovery scenarios and the track two based on the feedback from step 7.

I repeated the above steps multiple iterations until the car was driving around both track one and track two.

At the end of the process, the vehicle is able to drive autonomously around the track one and track two without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 89-123) consisted of a convolution neural network with the following layers and layer sizes based on the slight modification of Nvidia model:


![alt text][image_nvidia_model_tf]

####3. Creation of the Training Set & Training Process

The creation of the training set and training process was done through multiple iterations. Initially, I studied Udacity's sample training data using the provided video.py to see how the training data looked like, and used the sample training data set for training.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from the car stuck on the right shoulder:

![alt text][image_recovery1]
![alt text][image_recovery2]
![alt text][image_recovery3]

Then, I recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image_center]

Then, I drove the car counter-clockwise to have a more balanced set of data.

Finally, I applied the above techniques (center lane driving, reverse driving and recovery) for track two.

To augment the data sat, I also flipped images and angles thinking that this would provide extra data for training. For example, here is an image that has then been flipped:

![alt text][image_right]
![alt text][image_right_flipped]


After the collection process, I had 53,100 data points. I then preprocessed this data by cropping the 1/3 of the top image which does not contain car lanes information, 
and resized the cropped image to 200 pixels width by 66 pixels height so that the image is readily used for the nvidia model.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
Even with 20 epochs, the vehicle was able to drive small portion of track 1. With 400 epochs, the vehicle was able to keep going on the track one, and finish the track two. I used an adam optimizer so that manually training the learning rate wasn't necessary.


###Final Thoughts

I think this project is a great example of how the machine learning can solve the real world problem which would be almost impossible to solve or at least not economical to solve in a conventional programming approach. With the ease of Keras and power of TensorFlow and other python libraries, implementing the behavioral cloning of vehicle driving was quite straightforward. 

Here are further enhancements to this project:
- Training time -- Even with Nvidia GTX 1070 graphic card, Intel i5 processor and 32GB of RAM, it took several hours to train the model with ~400 epochs.
- Filtering of training data collection -- Udacity's Sample Training Data was compact and efficient. When I captured the training data, each second contained large amount of frames, making the training data quite large, and I ended up having to use more than 50,000 data points. I need to find a way to trim down the frames per second when capturing the train data.
- The automated driving was not stable -- While the vehicle was able to complete both tracks, I noticed that the vehicle was moving left and right even on the straight road. I suspect this is either a training data issue or not filtering out outliers of steering angles.

With that, here are the video recordings of the vehicle driving the track one and track two:
* Track One
[![Alt text](https://img.youtube.com/vi/h0Bbencn_1I/0.jpg)](https://www.youtube.com/watch?v=h0Bbencn_1I)

* Track Two
[![Alt text](https://img.youtube.com/vi/ZRE97JTQ1Nc/0.jpg)](https://www.youtube.com/watch?v=ZRE97JTQ1Nc)