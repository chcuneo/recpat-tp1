import numpy as np
from scipy.misc import imread, imsave
from scipy.stats import multivariate_normal

IMG_WIDTH = 640
IMG_HEIGHT = 400

print('reading image...', end="")
phantom = imread('phantom.png')
print(' ok')

print('generating multivariate normals and outputs...', end="")
green_mean = phantom[0][0]
orange_mean = phantom[399][0]
purple_mean = phantom[399][IMG_HEIGHT]

synthetic_1 = np.empty([IMG_HEIGHT,IMG_WIDTH,3])
synthetic_2 = np.empty([IMG_HEIGHT,IMG_WIDTH,3])
synthetic_3 = np.empty([IMG_HEIGHT,IMG_WIDTH,3])

# Equal isotropic covariance matrixes
cov_1 = [[100,0,0], [0,100,0], [0,0,100]]
green_density_1 = multivariate_normal(mean=green_mean, cov=cov_1)
orange_density_1 = multivariate_normal(mean=orange_mean, cov=cov_1)
purple_density_1 = multivariate_normal(mean=purple_mean, cov=cov_1)

# Different diagonal covariance matrixes
green_density_2 = multivariate_normal(mean=green_mean, cov=[[100,0,0], [0,50,0], [0,0,130]])
orange_density_2 = multivariate_normal(mean=orange_mean, cov=[[170,0,0], [0,170,0], [0,0,170]])
purple_density_2 = multivariate_normal(mean=purple_mean, cov=[[123,0,0], [0,233,0], [0,0,87]])

# Different matrixes
green_density_3 = multivariate_normal(mean=green_mean, cov=cov_1)
orange_density_3 = multivariate_normal(mean=orange_mean, cov=cov_1)
purple_density_3 = multivariate_normal(mean=purple_mean, cov=cov_1)

densities = [[green_density_1, orange_density_1, purple_density_1],
             [green_density_2, orange_density_2, purple_density_2],
             [green_density_3, orange_density_3, purple_density_3]]

synthetics = [synthetic_1,
              synthetic_2,
              synthetic_3]
print(' ok')

for i in range(IMG_HEIGHT):
  if (i % 10 == 0):
    print('\rprocessing phantom... row {}'.format(i), end="")
  for j in range(IMG_WIDTH):
    for case in range(3):
      pixel_class = max(enumerate(densities[case]), key=(lambda x: x[1].pdf(phantom[i][j])))[0]
      synthetics[case][i][j] = densities[case][pixel_class].rvs().astype(int)
print('\rprocessing phantom... ok     ')

print('generating synthetic images... ', end="")
max_pixel = [255,255,255]
min_pixel = [0,0,0]

img_1 = np.empty([IMG_HEIGHT,IMG_WIDTH,3])
img_2 = np.empty([IMG_HEIGHT,IMG_WIDTH,3])
img_3 = np.empty([IMG_HEIGHT,IMG_WIDTH,3])

imgs = [img_1,
        img_2,
        img_3]

for i in range(IMG_HEIGHT):
  if (i % 10 == 0):
    print('\rgenerating synthetic images... row {}'.format(i), end="")
  for j in range(IMG_WIDTH):
    for case in range(3):
      imgs[case][i][j] = np.clip(synthetics[case][i][j], min_pixel, max_pixel)

for case in range(3):
  imsave('synthetic-{}.png'.format(case), imgs[case])
print('\rgenerating synthetic images... ok     ')

test_1 = np.empty([IMG_HEIGHT,IMG_WIDTH,3])
test_2 = np.empty([IMG_HEIGHT,IMG_WIDTH,3])
test_3 = np.empty([IMG_HEIGHT,IMG_WIDTH,3])

tests = [test_1,
        test_2,
        test_3]

classes = [green_mean,
          orange_mean,
          purple_mean]

for i in range(IMG_HEIGHT):
  if (i % 10 == 0):
    print('\rprocessing tests... row {}'.format(i), end="")
  for j in range(IMG_WIDTH):
    for case in range(3):
      pixel_class = max(enumerate(densities[case]), key=(lambda x: x[1].pdf(synthetics[case][i][j])))[0]
      tests[case][i][j] = classes[pixel_class]
print('\rprocessing tests... ok     ')

print('saving tests images... ', end="")
for case in range(3):
  imsave('test-{}.png'.format(case), tests[case])
print('ok')
