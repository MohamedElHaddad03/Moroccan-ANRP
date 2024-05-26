





# image_processing/views.py
import os
import cv2
import numpy as np
from django.conf import settings
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from ultralytics import YOLO
import easyocr
from scipy.ndimage import interpolation as inter

model_path = os.path.join(settings.BASE_DIR, 'detect/train13/weights/best.pt')
model = YOLO(model_path)
reader = easyocr.Reader(['en', 'ar'])
threshold = 0.5

def imclearborder(imgBW, radius):
    imgBWcopy = imgBW.copy()
    contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    imgRows, imgCols = imgBW.shape[:2]
    contourList = []

    for idx in np.arange(len(contours)):
        cnt = contours[idx]
        for pt in cnt:
            rowCnt, colCnt = pt[0][1], pt[0][0]
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)
            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

    return imgBWcopy

def filter_white_lines(image):
    gray = image
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    for y in range(binary.shape[0]):
        white_pixels = np.sum(binary[y] == 255)
        white_percentage = white_pixels / binary.shape[1]

        if white_percentage > 0.6:
            binary[y] = 0

    return binary

def morpholo(image):
    if len(image.shape) == 2:
        grayImage = image
    else:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    long = image.shape[0]
    dst = cv2.Canny(grayImage, 50, 100)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > long/2.5:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    return image

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

def preprocess_image(cropped_image):
    if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = cropped_image

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded_image



from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os
import cv2
import easyocr

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en', 'ar'])

def replace_characters(text):
    text = text.replace('?', '9')
    text = text.replace('!', '|')
    text = text.replace('؟', '6')
    text = text.replace('S', '5')
    text = text.replace('b', '6')
    text = text.replace('B', '8')
    text = text.replace('d', '6')
    return text

def image_upload(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_path = default_storage.save('uploads/' + image.name, ContentFile(image.read()))
            image_full_path = os.path.join(settings.MEDIA_ROOT, image_path)

            frame = cv2.imread(image_full_path)
            if frame is None:
                print(f"Failed to read image {image_full_path}")
            else:
                results = model(frame)[0]
                output_directory = os.path.join(settings.MEDIA_ROOT, 'outputs')
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                cropped_images = []
                texts = []
                for i, result in enumerate(results.boxes.data.tolist()):
                    x1, y1, x2, y2, score, class_id = result

                    if score > threshold:
                        cropped_image = frame[int(y1): int(y2), int(x1): int(x2)]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                        angle, corrected = correct_skew(cropped_image)
                        preprocessed_image = preprocess_image(corrected)
                        preprocessed_image = 255 - preprocessed_image
                        test = filter_white_lines(preprocessed_image)
                        morph_image = morpholo(test)

                        # Perform OCR on the images
                        result1 = reader.readtext(cropped_image, allowlist='S|!?Bdb؟0123456789ابجدهوحطيأ', low_text=0.3)
                        result2 = reader.readtext(preprocessed_image, allowlist='bBd|!?S؟0123456789ابجدهوحطيأ', low_text=0.3)
                        result3 = reader.readtext(morph_image, allowlist='|Bbd!?S؟0123456789ابجدهوحطيأ', low_text=0.3)

                        for res in [result1, result2, result3]:
                            if res:
                                text = ''.join(entry[1] for entry in res if entry[1].strip())
                                text = replace_characters(text)
                                texts.append((text, res[0][-1]))
                            else:
                                texts.append(("", 0))

                        cropped_image_path = os.path.join(output_directory, f"cropped_{i}.jpg")
                        cv2.imwrite(cropped_image_path, cropped_image)
                        cropped_images.append(cropped_image_path)

                annotated_image_path = os.path.join(output_directory, "annotated.jpg")
                cv2.imwrite(annotated_image_path, frame)

                # Determine the most confident OCR result
                sorted_texts = sorted(texts, key=lambda x: x[0])
                most_confident_text, most_confident_confidence = sorted_texts[-1]

                print("Most confident OCR result:")
                print("Text:", most_confident_text)
                print("Confidence:", most_confident_confidence)

                cropped_images_urls = os.path.join(settings.MEDIA_URL, 'outputs',"cropped_0.jpg")
                annotated_image_url = os.path.join(settings.MEDIA_URL, 'outputs', "annotated.jpg")

                return render(request, 'results.html', {
                    'cropped_images_urls': cropped_images_urls,
                    'annotated_image_url': annotated_image_url,
                    'plateresult':most_confident_text,
                    'confidence':most_confident_confidence,
                })
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})










