# Capstone_GA

### Introduction
This was a project using convolutional neural networks to classify between two styles of art, impressionism and romanticism. More information can be found on my [blog](https://medium.com/@anapetronijevic56/cnn-for-art-classification-d69db5bbec31)

### Data
Images and a csv file containing metadata were used from kaggle's painters by numbers. I was able to filter through, by creating a new csv in my frist notebook containg the style and names of the photos, and then in my second notebook looping through.

### Process
A CNN with 4 convolutional layers and a maximum of 256 neurons was used. The model was original trained on 20 epochs and than 2 more locally. With almost an 80% accuracy, I created a dataframe with the image name, the actual class and the actual predictions. I created a mask where I searched for famous works in each period and later used that to identify if the works were correctly or incorrectly predicted. I finally looked through the rgb values in my final notebook of the most famous works in the validation set.

### Futher Goals
- Use localization/object detection to find common subject matter and how that might affect predictions.
- For example, an impressionist portrait was in accurately predicted. Portrait are not common in impressionism.
- Train sketches and artist separately and use transfer learning.
- Paintings at the beginning and end of certain period might have influences of a prior or changing style.
- Higher quality images
- Monet painted many images of the same location at different times and analyzing how pixel values can change the prediction.
- More data. Scrapping museums, websites of artist, institutions.
- Use a cloud platform to train a stronger model.
