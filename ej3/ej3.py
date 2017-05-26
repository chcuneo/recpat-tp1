import numpy as np
import sys
from scipy.misc import imread, imsave
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

classes = ['desert', 'red', 'dry']
test_percentage = 0.2

print('reading images...', end="", flush=True)

photo = imread('circular.png')
photo_by_classes = []

# Get and clean training data
photo_by_classes.append(imread('desert.png'))
photo_by_classes.append(imread('red.png'))
photo_by_classes.append(imread('dry.png'))
print(' ok')

print('cleaning training data...', end="", flush=True)
classes_training_data = []
classes_test_data = []
for i in range(len(classes)):
  clean_data = []
  for row in photo_by_classes[i]:
    for pixel in row:
      if pixel[3] != 0:
        clean_data.append(pixel[:3])
  training_data = np.array(clean_data[int(len(clean_data)*test_percentage):])
  test_data = np.array(clean_data[:int(len(clean_data)*test_percentage)])
  classes_training_data.append(training_data)
  classes_test_data.append(test_data)
print(' ok')


# Estimate means and covariance matrixes
print('estimating mean and covariances...', end="", flush=True)
means = []
covariance_matrixes = []
for data in classes_training_data:
  print('\restimating mean and covariances...{}'.format(len(means)+1), end="", flush=True)

  mean = data.mean(0)

  covariance_matrix = np.zeros([data.shape[1], data.shape[1]])
  for sample in data:
    distance = np.array([sample - mean])
    covariance_matrix += (distance * distance.T)

  means.append(data.mean(0))
  covariance_matrixes.append(covariance_matrix / data.shape[0])
print('\restimating mean and covariances... ok', flush=True)

for i, klass in enumerate(classes):
  print('mean-{}:'.format(klass))
  print('{}'.format(means[i].tolist()))
  print('covariance-{}:'.format(klass))
  print('{}'.format(covariance_matrixes[i].tolist()))

# Generate multivariate normals to classify
print('generating multivariate normals...', end="", flush=True)
gaussians = []
for i, _ in enumerate(classes):
  gaussians.append(multivariate_normal(cov=covariance_matrixes[i], mean=means[i]))
print(' ok')

# Classify original image
print('classifiyng original image...', end="", flush=True)
result = np.zeros(photo.shape[:2])
result_phantom = np.zeros([photo.shape[0], photo.shape[1], 3])

for i, row in enumerate(photo):
  print('\rclassifiyng original image... row{}'.format(i), end="", flush=True)
  for j, pixel in enumerate(row):
    pixel_guessed_class = max(enumerate(gaussians), key=(lambda x: x[1].pdf(pixel[:3])))[0]
    result[i][j] = pixel_guessed_class
    result_phantom[i][j] = means[pixel_guessed_class]
print('\rclassifiyng original image... ok')

print('saving resulting image...', end="", flush=True)
imsave('result.png', result_phantom)
print(' ok')

# Build confusion matrix
confusion_matrix = np.zeros([len(classes), len(classes)])

def get_pixel_class(i, j):
  for klass_number, klass in enumerate(photo_by_classes):
    # Check alpha of pixel (sign to know if pixel has information or not)
    if klass[i][j][3] != 0:
      return klass_number
  return -1

print('generating confussion matrix...', end="", flush=True)
for i, row in enumerate(result):
  print('\rgenerating confussion matrix... row{}'.format(i), end="", flush=True)
  for j, pixel_guessed_class in enumerate(row):
    pixel_actual_class = get_pixel_class(i,j)
    if pixel_actual_class != -1:
      confusion_matrix[pixel_actual_class][int(pixel_guessed_class)] += 1

print('\rgenerating confussion matrix... ok')

print('saving confusion matrix...', end="", flush=True)
plt.imshow(confusion_matrix, interpolation='none')
plt.colorbar()
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)
plt.savefig('confusion-matrix.png')
print(' ok')

# Classify testing regions
print('classifying and generating confussion matrix for test regions...', end="", flush=True)
test_confusion_matrix = np.zeros([len(classes), len(classes)])
for klass, data in enumerate(classes_test_data):
  for sample in data:
    pixel_guessed_class = max(enumerate(gaussians), key=(lambda x: x[1].pdf(sample[:3])))[0]
    test_confusion_matrix[klass][pixel_guessed_class] += 1
print(' ok')

print('saving test regions confusion matrix...', end="", flush=True)
plt.clf()
plt.imshow(test_confusion_matrix, interpolation='none')
plt.colorbar()
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)
plt.savefig('test-confusion-matrix.png')
print(' ok')
