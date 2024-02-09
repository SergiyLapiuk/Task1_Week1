import csv
import json
import urllib.request
from sklearn.metrics import f1_score

import cv2
import numpy as np
from matplotlib import pyplot as plt


# Функція для завантаження зображення з URL
def load_image_from_url(url):
    # Використовуємо urllib для завантаження зображення з інтернету
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def get_image_index(image_url):
    # Завантажуємо зображення за допомогою функції вище
    image = load_image_from_url(image_url)

    # Конвертуємо зображення з BGR у HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h = hsv_image[:,:,0]/256
    s = hsv_image[:,:,1]/256
    v = hsv_image[:, :, 2]/256

    # Розраховуємо перший момент (середнє) для кожного каналу (Відтінок, Насиченість, Значення)
    mean_hue = np.mean(h)
    mean_saturation = np.mean(s)
    mean_value = np.mean(v)

    # Розраховуємо другий момент (стандартне відхилення) для кожного каналу
    std_dev_hue = np.std(h)
    std_dev_saturation = np.std(s)
    std_dev_value = np.std(v)

    # Розраховуємо третій момент для кожного каналу
    hue_values = h.flatten()
    saturation_values = s.flatten()
    value_values = v.flatten()

    skewness_hue = ((hue_values - mean_hue) ** 3)
    skewness_saturation = (((saturation_values - mean_saturation)) ** 3)
    skewness_value = ((value_values - mean_value) ** 3)

    skewness_hue = np.cbrt(np.mean(skewness_hue))
    skewness_saturation = np.cbrt(np.mean(skewness_saturation))
    skewness_value = np.cbrt(np.mean(skewness_value))

    # Повертаємо значення у форматі матриці
    return np.array([[mean_hue, mean_saturation, mean_value],
                     [std_dev_hue, std_dev_saturation, std_dev_value],
                     [skewness_hue, skewness_saturation, skewness_value]])

def get_dist_image(index1, index2):
    dist = np.sum(np.abs(index1 - index2))
    return dist

def compare_images(json_file_path):
    csv_file_path = 'output_results2.csv'
    # Load JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Open a file to write the results
    output_file_path = 'output_results_val.txt'
    total_pairs = len(data['data']['results'])  # Get the total number of pairs
    counter = 0  # Initialize a counter variable
    with open(csv_file_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['taskId', 'answer'])  # Writing header
        # Iterate over each pair of images
        for result in data['data']['results']:
            taskId = result['taskId']
            image_url1 = result['representativeData']['image1']['imageUrl']
            image_url2 = result['representativeData']['image2']['imageUrl']

            index_image1 = get_image_index(image_url1)
            index_image2 = get_image_index(image_url2)
            dist = get_dist_image(index_image1, index_image2)
            answer = 1 if dist < 0.89 else 0
            csvwriter.writerow([taskId, answer])

            counter += 1
            print(f"{counter}/{total_pairs}")


def compare_images_txt(json_file_path):
    # Load JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Open a file to write the results
    output_file_path = 'output_results.txt'
    total_pairs = len(data['data']['results'])  # Get the total number of pairs
    counter = 0  # Initialize a counter variable
    with open(output_file_path, 'w') as output_file:
        # Iterate over each pair of images
        for result in data['data']['results']:
            image_url1 = result['representativeData']['image1']['imageUrl']
            image_url2 = result['representativeData']['image2']['imageUrl']

            index_image1 = get_image_index(image_url1)
            index_image2 = get_image_index(image_url2)
            dist = get_dist_image(index_image1, index_image2)
            answer = result['answers'][0]['answer'][0]['id']
            # Write the distance to the output file
            output_file.write(f"Distance between images: {dist}  Valid answer: {answer}\n")

            counter += 1
            print(f"{counter}/{total_pairs}")

# This function will parse the file content and return lists of distances and answers
# Function to read data from a text file
def read_data_from_file(file_path):
    distances = []
    answers = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split('Valid answer:')
            distance = float(parts[0].split(':')[1].strip())
            answer = int(parts[1].strip())
            distances.append(distance)
            answers.append(answer)
    return distances, answers

# Function to plot the data
def plot_data(distances, answers):
    plt.figure(figsize=(10, 5))
    plt.scatter(distances, answers, color='blue')
    plt.title('Distance between and Valid answer')
    plt.xlabel('Distance between images')
    plt.ylabel('Valid answer')
    plt.xlim(0.5, 0.9)
    plt.grid(True)
    plt.show()

def get_score(file_path, json_file_path):
    # Initialize an empty list to store the results
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split('Valid answer:')
            distance = float(parts[0].split(':')[1].strip())
            result = 1 if distance < 0.89 else 0
            results.append(result)

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    answers = []
    for result in data['data']['results']:
        answer = result['answers'][0]['answer'][0]['id']
        answers.append(int(answer))

    f1 = f1_score(results, answers)

    return f1


def main():
    json_path = 'C:\\Users\\acer\\Desktop\\pythonProject\\train_task1.json'
    json_path_test = 'C:\\Users\\acer\\Desktop\\pythonProject\\test_task1.json'
    json_path_val = 'C:\\Users\\acer\\Desktop\\pythonProject\\val_task1.json'
    file_path = 'C:\\Users\\acer\\Desktop\\pythonProject\\output_results.txt'
    file_path_test = 'C:\\Users\\acer\\Desktop\\pythonProject\\output_results_test.txt'
    file_path_rgb = 'C:\\Users\\acer\\Desktop\\pythonProject\\output_results_rgb.txt'
    file_path_train_rgb = 'C:\\Users\\acer\\Desktop\\pythonProject\\output_results_train_rgb.txt'
    # Read data from file
    #distances, answers = read_data_from_file(file_path)

    # Plot the data
    #plot_data(distances, answers)

    #compare_images_txt(json_path)
    #compare_images_txt(json_path_test)
    #compare_images_txt(json_path)
    #f1 = get_score(file_path_test, json_path_test)
    #print(f"F1 Score: {f1}")
    compare_images(json_path_val)

# Запуск програми
if __name__ == "__main__":
    main()