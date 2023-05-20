# A Study of Detecting COVID-19 Severity Using Lung Ultrasonography


  ### Team - Abdul Wahed, Amar Sayani, Bhargav Nishanth Katikey



## Abstract : 
The wide spread of COVID-19 has made the world's healthcare systems vulnerable, particularly in nations with poor health infrastructure. Wuhan, China, confirmed the first-ever case of COVID-19 in 2019. The virus impacts the body's respiratory organs, which causes difficulties in breathing, and affected individuals develop symptoms similar to pneumonia. The infection is identified using a real-time reverse transcriptase-polymerase chain reaction (RT-PCR) kit. The COVID-19 Coronavirus Real Time PCR Kit is an in vitro diagnostic test for qualitatively detecting nucleic acid from SARS-CoV-2 and is based on real-time fluorescence reverse transcription PCR technology. Due to an inadequate supply of kits and the expensive nature of the RT-PCR testing method, suspected patients cannot receive immediate medical attention, which allows the virus to multiply. Radiologists studied advances in radiographic imaging, such as CT scans, which generate detailed images of the body of outstanding quality, to devise an alternative. Deep learning techniques are utilized to differentiate between an unaffected individual and a COVID-19 patient based on a suspected patient's computed tomography (CT) image. Multiple deep learning techniques have been put forth for detecting COVID-19. This study used CNN architectures such as MobileNetV2, DenseNet121, basic CNN architecture, VGG16, and NASNet. There are 4335 total CT scan images in the collection, with "Non-COVID," "COVID," and "Viral Pneumonia" scans. Train, validation, and test datasets are separated. The accuracy results for MobileNetV2 are 81.9%, DenseNet121 is 62.8%, Basic CNN is 81.6%, VGG16 is 66.3%, and NASNet is 81.9%. The study that was conducted led to the conclusion that the NASNet and MobileNetV2 architectures offered more accuracy when compared to others.


## Introduction : 
A virus identified as SARS-CoV-2 is the cause of the illness COVID-19. Lung infections may vary from a simple cold to a potentially fatal illness. Respiratory symptoms frequently accompany coronavirus infections. Infected people may experience minor, self-limiting illnesses like influenza that have adverse effects. Respiratory problems, exhaustion, and a sore throat are among the symptoms, as are fever, coughing, and trouble breathing [1]. Medical health standards recommend chest imaging as a rapid and effective procedure, and it has been cited in multiple studies as the initial tool in epidemic screening. Segmentation and classification are prominent examples of the various computer vision techniques employed. An automated method that can provide disintegration and assess the infection region of patients every three to five days and as well as track the progression of infection in individuals through CT scan imaging and clinical detection is necessary when a quick and straightforward procedure running on constrained computing devices, is required as COVID-19 is a viral infection that even expert medical professionals have trouble in detecting [2]. Applying deep learning techniques to understand radiological images has become the subject of several investigations. They began to address the limitations of COVID-19 on a collection of radiological image-based medical procedures. Among the most important deep learning algorithms, the CNN architecture is the best method for identifying it. The automatic diagnosis of lung infections using CT scans provides a significant opportunity to expand on current healthcare practices to treat COVID-19.
However, CT has many challenges [3]. CNN can detect pulmonary disorders such as emphysema, pleurisy, pneumonia, and TB. The CT systems have a disadvantage because there is an exposure to X-ray radiation since the contrast of the soft tissues is less than that of the MRI [4]. Deep learning algorithms may identify the difference between an unaffected person and a COVID-19 patient from the suspected patient's X-ray and CT scan. The development of COVID-19 diagnostic systems uses deep learning models such as DenseNet, ResNet, and MobileNetV2. Regular patients and COVID-19-positive patients are taken into consideration. Images from chest X-rays can show pneumonia, the flu, and other chest-related conditions. 
To help radiologists diagnose cases more quickly, machine learning methods based on X-ray imaging are used as a decision support system. It was initially suggested to use natural image processing to help radiologists differentiate COVID-19 disorders from radiographic images of the chest using a critical analysis of 12 traditional CNN architectures. Along with an extensive set of viral infections, regular radiographs, and diseases other than COVID viruses, COVID-19 X-ray images were also used. Simple CNN architectures can perform better than architectures like Xception when trained on a small picture dataset. Despite CNN's great accuracy in classification, therapists may only look at the results once they can visually inspect the area of the input image that the CNN picked up [5]. By isolating their features, different deep learning techniques were compared for automatic COVID-19 categorization. An essential component of learning is producing the most accurate feature, and five convolutional neural networks, i.e., basic CNN architecture, MobileNetV2, DenseNet121, VGG16, and NASNet, were chosen from many convolutional neural networks.


## Methods: 
Deep learning Techniques: Machine learning techniques are related to artificial intelligence methods that imitate human learning. Deep learning is an essential part of data science, encompassing data analysis and prediction modeling. The convolutional neural network is a specific type of deep neural network used to evaluate visual imagery in deep understanding. In order to distinguish between different items in a picture, CNN uses a deep learning technique that reads a picture as an input and allocates weights to each one. CNN is utilized for image classification and identification due to its high accuracy [6].
Classification: The statistical evaluation of the data is done using deep learning architectures like DenseNet121, MobileNetV2, CNN, VGG16, and NASNet. These models are trained via transfer learning. The training, validation, and testing datasets for the image classification job are conveniently loaded and preprocessed using data generators. These are implemented using the ImageDataGenerator class from the Keras library.

### MobileNetV2:
MobileNet employs depthwise separable convolutions. It significantly lowers the number of parameters compared to a network with standard convolutions of identical depth in the nets. Thus, portable deep neural networks are developed. In order to build MobileNets, depthwise separable convolution layers are used. There are two convolution layers for every depth-wise removable layer: pointwise and depth-wise layers. A standard MobileNet has 4.2 million parameters; this number can be decreased by adjusting the width multiplier hyperparameter. The input image size is 224 × 224 pixels [7]. The model starts with an input layer that takes images of shapes (224, 224, 3). To extract low-level features, a number of basic convolutional layers are applied to the input data. These layers consist of batch normalization, ReLU activation, and a 3x3 convolutional layer with a stride of 2. There are 17 Bottleneck blocks in total in the MobileNet design. An expansion block, a depthwise block, and a projection block comprise each Bottleneck block. These blocks gradually increase the complexity of the characteristics retrieved from the earlier layers. The model concludes with a dense layer with softmax activation to get the final classification probabilities and global average pooling to minimize the spatial dimensions of the data. 

![image](https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/85d3aefd-23e8-488b-a5a2-81cb082309c6)
<p align="center">Figure 1. MobileNet Architecture</p>


### DenseNet121 :
A potential method to extend deep convolutional networks without encountering issues like increasing gradients and disappearing gradients is DenseNet121, a densely linked neural network. Most information and gradient flow are exchanged between directly connected layers, which resolves the problems. Rather than depending on massive, deep, or broad CNN architectures for significance, the goal is to concentrate on reusing features. DenseNets use fewer or a comparable number of nodes than traditional CNN. Due to the fact that DenseNets does not train feature maps, hence the parameters are not required [8]. The architecture of DenseNet-121 for image classification includes dense blocks and transition blocks with batch normalization, relu activation, and dropout regularization. Global average pooling and a dense layer with softmax activation are used as the model's last two components. This design is renowned for its dense connections that enable feature reuse and its proficiency in addressing gradient vanishing/exploding issues. The model is then expanded with several dense blocks and transition blocks, each with different layers and growth rates. These blocks gradually increase the number of channels while decreasing the spatial dimensions. The widely used DenseNet-121 convolutional neural network design solves the vanishing gradient issue and encourages feature reuse by creating dense connections across layers. It has been demonstrated to reach cutting-edge performance on several image classification tasks.

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/3caffe0b-9336-44be-81c0-0a6cdd860b99">
</p>
<p align="center" >Figure 2. DenseNet121 Architecture</p> 

### Basic CNN Architecture: 
CNNs are a subclass of Deep Neural Networks frequently used for visual image analysis. CNNs can identify and classify the characteristics of images. Their applications involve natural language processing, image classification, video and image analysis for medical purposes, and image and video recognition. CNN is helpful for picture identification due to its excellent accuracy. Medical image analysis is only one of the many areas where image recognition is used. This CNN feature extraction model seeks to minimize the number of features in a dataset. It generates new features that compile an initial set of features existing features into a single new feature. As seen in the CNN architectural diagram, there are several CNN layers.

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/ab1adeb5-bddf-45e6-bd44-1819ea09186c">
</p>
<p align="center" >Figure 3. CNN Architecture</p> 


### VGG16 : 
One more popular convolutional neural network architecture for image classification applications is VGG16. VGG16 is employed for various reasons, including its efficacy and simplicity. The network of the VGG16 employs tiny 3x3 convolutional filters, enabling the network to learn more intricate aspects of the images. The pre-trained network weights are utilized to enhance performance on new datasets, and VGG16 has also been employed as a starting point for this process. The input picture for this new model is of size 224x224x3, it is processed using the pre-trained VGG16 model, and the output is passed through a dense layer with 256 neurons and a ReLU activation function.  

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/75c6163a-97c9-4d28-b9ea-f8169245d61d">
</p>
<p align="center" >Figure 4. VGG16 Architecture</p> 


### NASNet :
The most modern deep learning architecture for image classification tasks is NASNet (Neural Architecture Search Network). NASNet was developed by Google researchers and has excelled on several benchmark datasets, including ImageNet. One of NASNet's key advantages is its flexibility in handling various input sizes and resolutions, which makes it ideal for various applications ranging from mobile devices to high-performance computer systems. In order to avoid overfitting and expedite training, the model's pre-trained layers are frozen to stop its weights from being updated. Overall, NASNet is a compelling architecture that can deliver cutting-edge results on various image classification tasks.

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/ec1f9413-39f5-427c-9221-139b9ae31fe9">
</p>
<p align="center" >Figure 5. NASNet Architecture</p> 


## Data Wrangling: 
CT scans of the lungs are included in the dataset collection. A CT scan uses cutting-edge X-ray technology to assess internal organs precisely. Several publicly accessible sources, including the Italian Society of Radiology, NCBI, Eurorad, and Radiopedia, were used to gather the images for this dataset and had 4335 images in them. The data is labeled and classified into COVID, non-COVID, and viral pneumonia. The COVID class comprises CT scan images of COVID patients, whereas the non-COVID class includes CT scan images of individuals in good health. The CT scans of people with pneumonia infections other than covid are included in the viral pneumonia class.
Data Preprocessing : The aspect ratio of a squared picture is resized to have about the same height and width. The size of each input sample is filtered using the image-filtering preprocessing approach. The photos are rescaled in the proposed system to 224*224. The ImageDataGenerator function is used to create a generator for the training, and validation data, which rescales the pixel values of the images. The rescaled images can be seen below.

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/2abdd9c5-4d67-40e3-a343-aca55b295481">
</p>

 
## Result and Analysis : 
In this study, we developed and evaluated five models for detecting COVID-19 based on lung CT scans. The models were trained and tested using a dataset containing images from three classes: COVID-19 positive, COVID-19 negative, and viral pneumonia cases.
### 1. Basic CNN Architecture:
The initial model was a basic CNN architecture consisting of multiple convolutional and pooling layers, followed by fully connected layers. The model achieved an overall accuracy of 81.6%.

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/0d7890ee-b93a-43d9-b174-454cbc666d58">
</p>

The training accuracy and validation graph shows the model's accuracy on the training and validation data throughout training epochs. It represents how well the model learns and improves its predictions as it sees more training examples. It also indicates how well the model generalizes to unseen data and helps identify whether it is overfitting or underfitting. Here training accuracy is not consistently higher than the validation accuracy; it suggests that the model is balanced.

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/af0d234b-8f9e-448e-add0-74db111a7c7a">
</p>

The training and validation loss graph shows the value of the loss function (typically cross-entropy) during training on the training data. It represents how well the model minimizes its prediction errors and adjusts its parameters to improve performance. It also indicates how well the model generalizes to new data and helps identify whether it is overfitting or underfitting. Ideally, the training and validation loss decreases and converges, with a small gap between them. A decreasing loss indicates that the model is improving its performance and minimizing errors.

### 2. MobileNetV2:  
One of the models used was MobileNetV2, a pre-trained convolutional neural network architecture known for its efficiency and performance. The MobileNetV2 model was fine-tuned and trained on our dataset containing images from three classes. The model achieved an overall accuracy of 81.9%.

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/ef9c1f22-e271-4828-8d89-7d95c7316d91">
</p>
 
Comparing the MobileNetV2 model to the previous CNN architecture, we observed that MobileNetV2 outperformed in terms of overall accuracy. The model demonstrated better discrimination between the COVID-19 positive, negative, and viral pneumonia cases resulting in higher accuracy. This highlights the effectiveness of MobileNetV2 in capturing relevant features for COVID-19 detection in lung CT scans.

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/9ac35140-ee36-46a7-98bf-32518975883b">
</p>
 
We observed a decreasing trend in training and validation loss throughout the training process. This suggests that the model was effectively learning the patterns and features in the training data and successfully generalizing its knowledge to the validation data. The decreasing loss indicates that the model gradually converged towards an optimal solution, minimizing the discrepancy between predicted and actual outputs. A decreasing loss signifies that the model is learning and improving, while an increasing or stagnant loss could indicate issues such as overfitting or underfitting. Overall, the decreasing training and validation loss for the MobileNetV2 model indicate its effectiveness in learning from the data and making accurate predictions.

### 3. DenseNet121: 
DenseNet121 is a deep convolutional neural network architecture that utilizes dense connections, enabling information flow across layers. We trained and tested the DenseNet121 model on our dataset of lung CT scans. The model achieved an overall accuracy of 62.8%.

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/5dcf530b-e3fc-4e4a-aaec-498b01afc552">
</p>
 
The accuracy graph for the DenseNet121 model showed an increase in both training and validation accuracy over the epochs. This upward trend indicates that the model was effectively learning from the training data and improving its ability to classify lung CT scans.

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/2b93bb05-4dfe-4680-8f01-880db6bd3030">
</p>
 
The loss graph for the DenseNet121 model displayed a steady decrease in both training and validation loss over the epochs. This reduction in loss signifies that the model was successfully minimizing the difference between its predicted outputs and the actual labels of the images. 

### 4.VGG16:
VGG16 is a popular convolutional neural network architecture known for its simplicity and effectiveness. We fine-tuned the pre-trained VGG16 model on our dataset and evaluated its performance. The VGG16 model achieved an overall accuracy of 66.3%

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/116d7e32-069e-4097-87ce-66d6d2c92f98">
</p>
 
The accuracy graph for the VGG16 model demonstrated an increasing trend for both the training and validation datasets. As the number of epochs increased, the model's accuracy on the training data improved, indicating its ability to classify lung CT scans correctly. The validation accuracy followed a similar pattern but might exhibit slight fluctuations. The general upward trend suggests that the model was learning and generalizing well to unseen data.

<p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/44a90838-f067-463e-b836-65bc71a8cb20">
</p>
 
The loss graph for the VGG16 model showed a decreasing trend for both the training and validation datasets. As the model underwent training over multiple epochs, the loss steadily decreased, indicating that the model was minimizing the difference between its predicted outputs and the true labels of the images.

### 5.NASNet: 
NASNet is a neural architecture search-based model that automatically discovers the optimal architecture for a given task. We trained and evaluated the NASNet model on our dataset of lung CT scans. The model achieved an overall accuracy of 81.9%. Comparing the performances of the five models, we observed that the NASNet and MobileNetV2 models achieved the highest overall accuracy of 81.9%.
 
 <p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/0c4d48b0-d7fe-4e9b-9fe9-294dff0f4472">
</p>

The accuracy graph for the NASNet model exhibited a steady increase in accuracy for both the training and validation datasets as the number of epochs increased. This indicates that the model effectively learned and improved its ability to classify lung CT scans into the appropriate classes. The accuracy of the training data increased over time, showing that the model could learn and memorize the training samples. Similarly, the accuracy of the validation data increased, demonstrating the model's ability to generalize well to unseen data.
 
 <p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/146f7ad1-4d4e-4261-b302-40eded9ca02b">
</p>

The loss graph for the NASNet model demonstrated a decreasing trend for both the training and validation datasets. As the model underwent training, the loss gradually decreased, indicating that the model minimized the discrepancy between its predicted outputs and the true labels of the images. The decreasing loss suggests that the model learned the relevant patterns and features necessary for COVID-19 detection in lung CT scans. The accuracy and loss graphs for the NASNet model provide valuable insights into the model's learning progress and performance. The increasing accuracy and decreasing loss over the epochs indicate that the model was successfully learning from the training data and generalizing it to the validation data.

## Parameters used for training the models : 

 <p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/b36415cd-2230-46ae-b95d-022326d592a1">
</p>


## Conclusion: 
This paper uncovers the prevailing and efficacious deep learning architectures used to identify COVID-19 in suspected patients by analyzing CT scan images. By exploring key concepts, these deep learning algorithms offer effective results in detecting the presence of COVID-19. The study utilizes the most notable models: MobileNet, DenseNet, VGG16, and NASNet. However, specific measures need to be taken to enhance the accuracy of the suggested CNN models. These include incorporating a larger dataset, employing additional preprocessing techniques, and leveraging pre-trained models in transfer learning.
 
  <p align="center">
  <img src="https://github.com/Nishanth9702/DATA-606-Capstone-Project/assets/99062389/4fb8cbe7-9922-4ead-b496-4241777c6a72">
</p>
 
Among all the models, MobileNet and NASNet achieve the highest accuracy at 81.9%. Thus, the proposed system identifies MobileNet and NASNet as the optimal models for classifying CT scan images into COVID and non-COVID categories. However, a limitation of the study is its failure to identify COVID-affected lung areas. Further enhancements are necessary, including a larger dataset, additional preprocessing techniques, and the application of pre-trained models in transfer learning to improve the accuracy of the proposed CNN models. Additionally, future research may explore using the Yolo architecture to achieve even better accuracy.

## Future Scope :
Several potential future directions can be explored to enhance the accuracy and usefulness of the provided method. One promising avenue is to expand the dataset by collecting more CT scan images and incorporating additional medical imaging data, such as X-rays or MRI scans. This will bolster the models' robustness, generalizability, and accuracy in detecting COVID-19. Moreover, there is potential for developing more sophisticated models that classify COVID-19 from CT scan images and pinpoint the affected lung regions using attention mechanisms or advanced neural network architectures. Integrating these models into a more extensive automated COVID-19 diagnosis system for clinical use can alleviate the burden on healthcare providers and expedite the accuracy and speed of diagnosis. Through further exploration and development of these approaches, significant advancements in COVID-19 diagnosis using CT scan images can be achieved, ultimately improving patient outcomes.




## References

[1] 	M. V. P. J. Sandeep Kumar Mathivanan, "Forecasting of the SARS-CoV-2 epidemic in India using SIR model, flatten curve and herd immunity," Journal of Ambient Intelligence and Humanized Computing, 2020.<br>
[2] 	A. O. T. T. Sos Agaian, "Automatic COVID-19 lung infected region segmentation and measurement using CT-scans images," Elsevier - PMC COVID-19 Collection , 2020. <br>
[3] 	S. K. M. Maheshwari V, "Social economic impact of COVID-19 outbreak in India," International Journal of Pervasive Computing and Communications, 2020. <br>
[4] 	A. O. A. K. L. d. Sos Agaian b, "Automatic COVID-19 lung infected region segmentation and measurement using CT-scans images". <br>
[5] 	D. A. T. M. A. Asaad, "COVID-19 Detection Using CNN Transfer Learning from x-Ray Images,," 2020. <br>
[6] 	M. E. A. A. S. Saman Fouladi, "Efficient deep neural networks for classification of COVID-19 based on CT images: Virtualization via software defined radio," 2021. <br>
[7] 	A. R. K. Ali Abbasian Ardakani, "Application of deep learning technique to manage COVID-19 in routine clinical practice using CT images: Results of 10 convolutional neural networks," 2020. <br>
[8] 	Z. L. Gao Huang, "Densely Connected Convolutional Networks," 2018. <br>











