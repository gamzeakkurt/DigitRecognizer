# DigitRecognizer
Digit Recognizer Using Convolution Neural Network


In this project, we used the **MNIST**(Modified National Institute of Standards and Technology)  dataset which is available in [Kaggle](kaggle.com).  

**MNIST** dataset contains handwritten numbers images to work for the classification algorithm.  These numbers are between **0** and **9.** Also, you can see these numbers with their labels in **Figure 1.0**. Digit labels are under each image.
<p align="center"><img src="https://user-images.githubusercontent.com/37912287/113613364-ce294900-9659-11eb-9ea7-98f9a8874375.png" /></p>
<p align="center">
  <b>Figure 1.0</b>
</p>

**Figure 1.1** shows the distribution of digits.  As you can be seen, each digit has a different sample size.
<p align="center"><img src="https://user-images.githubusercontent.com/37912287/113613655-337d3a00-965a-11eb-9f54-28d34a1f0ac9.png" /></p>
<p align="center">
  <b>Figure 1.1</b>
</p>

We used a convolution neural network (**CNN**) for digit classification. Before the classification, we converted images to a two-dimensional array. Each image size was **28x28** and then, applied zero-padding the feature maps and each image size converted  **32x32** dimensional array. The purpose of using the padding method is that the first layer is learnt nothing in the network. The input layer is built to take **32x32** dimensional and the dimensions of images are passed into the next layer and keep the **28x28** dimensional image.  Moreover,  the images normalized from **0** to **255**. The reason for normalization is to ensure that the batch of images has a mean of **0** and a standard deviation of **1**. The advantage of this is seen in the reduction in the amount of training time.

We used **LeNet-5 CNN** which is made up of **7** layers. The combinations consist of **3** convolutional layers, **2** subsampling layers and **2** fully connected layers. We used **32, 48, and 256** filters respectively for convolutional layers. The output layer was **10**.
We used adam optimizer and accuracy metric. For fitting the network model,  we practised optimizer **32** rows of training data to do that **30** times through the dataset.  Also, you can see the model summary and results in **Figure 1.2** and **Figure 1.3**.
<p align="center"><img src="https://user-images.githubusercontent.com/37912287/113633973-941a7000-9676-11eb-91dc-c1a36971d005.PNG" /></p>
<p align="center">
  <b>Figure 1.2</b>
</p>

<p align="center"><img src="https://user-images.githubusercontent.com/37912287/113634188-01c69c00-9677-11eb-8ecd-e113b39e7c2f.PNG" /></p>
<p align="center">
  <b>Figure 1.3</b>
</p>




Lastly, we evaluated the model on the test data and gained a **99.43** percentage of success. The result is available in below **Figure 1.4**

<p align="center"><img src="https://user-images.githubusercontent.com/37912287/113635048-97aef680-9678-11eb-9cf5-09f812525479.PNG" /></p>
<p align="center">
  <b>Figure 1.4</b>
</p>
