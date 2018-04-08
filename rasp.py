from sklearn.externals import joblib
from os import listdir
from os.path import join
from PIL import Image
from time import strftime
import imutils
import cv2
import base64
import rethinkdb as r

path1 = "raw"
path2 = "resize"
path3 = "schannel"
path4 = "acne"

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

files = listdir(path1)
files.sort()

for f in files:
    file_path1 = join(path1, f)
    file_path2 = join(path2, f)

    img = cv2.imread(file_path1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.16, minNeighbors = 5, minSize = (25,25))
    for (x, y, w, h) in faces:
        roi_color = img[y : y + h, x : x + w]

    cut_img = img[y : y + h, x : x + w]
    resize_img = cv2.resize(cut_img, (300,300), interpolation = cv2.INTER_AREA)
    cv2.imwrite(file_path2, resize_img)

#please check pictures in resize folder

for x in files:
    file_path2 = join(path2, x)
    file_path3 = join(path3, x)
    hsv_img = cv2.imread(file_path2)
    hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)
    cv2.imwrite(file_path3, hsv_img[:,:,1])

for z in files:
    file_path2 = join(path2, z)
    file_path4 = join(path4, z)
    img = Image.open(file_path2)
    width = img.size[0]
    height = img.size[1]
    
    leftCheek = ( width / 6, height / 2, width / 3, height / 3 * 2)
    cropped_img = img.crop(leftCheek)
    
    rightCheek = ( width / 3 * 2, height / 2, width / 6 * 5, height / 3 * 2)
    cropped_img2 = img.crop(rightCheek)
    #Reserved picture
    toImage = Image.new('RGB', (100, 50))
    #Paste two pictures
    toImage.paste(cropped_img, (0, 0))
    toImage.paste(cropped_img2, (50, 0))
    toImage.save(file_path4)

# resize the image to a fixed size, then flatten the image into
# a list of raw pixel intensities
def image_to_feature_vector(image, size = (100, 50)):
    return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins = (8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()

model_1 = joblib.load('20180408_final_knn_hsv_schannel.pkl')
image_1 = cv2.imread('schannel/raspberry.jpg')
hist = extract_color_histogram(image_1)
result_1 = model_1.predict(hist.reshape(1, -1))[0]

model_2 = joblib.load('20180408_final_knn_rgb_cheek.pkl')
image_2 = cv2.imread('acne/raspberry.jpg')
pixels = image_to_feature_vector(image_2)
result_2 = model_2.predict(pixels.reshape(1, -1))[0]

dict = {'oil': {'good': 'oil_good', 'notgood': 'oil_bad'},
        'middle': {'good': 'middle_good', 'notgood': 'middle_bad'},
        'dry': {'good': 'dry_good', 'notgood': 'dry_bad'}}

### Client
image = open('raw/raspberry.jpg', 'rb')
image_64_encode = base64.b64encode(image.read())

data = {}
data['face'] = result_1
data['skin'] = result_2
data['model'] = dict[result_1][result_2]
data['timestamp'] = strftime('%Y%m%d_%H%M%S')
data['z_img'] = image_64_encode.decode('utf-8')

conn = r.connect(host = '35.196.140.167', port = 28015)
try:
    r.db_create('faceai').run(conn)
except:
    pass

conn.use('faceai')

try:
    r.table_create('project', primary_key = "id").run(conn)
except:
    pass

r.db('faceai').table('project').filter({'id': '916bcc4a-85a4-4cac-93b7-a00f9505f073'}).update(data).run(conn)
