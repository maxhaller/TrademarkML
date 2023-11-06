import pandas as pd
import numpy as np
import cv2
import math
import keras
import pickle
import os

from keras.models import Model
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity


class CharacterSimilarity:

    def __init__(self, df: pd.DataFrame, c_img_path: str, target_path: str, recompute_weights: bool):
        self.alphabet = []
        self.img_path = c_img_path
        self.target_path = target_path
        self.recompute_weights = recompute_weights
        self.weights = None
        self.basemodel = None

        if os.path.exists(f'{self.target_path}/weights.pickle') and not self.recompute_weights:
            with open(f'{self.target_path}/weights.pickle', 'rb') as f:
                self.weights = pickle.load(f)
        else:
            earlier_marks = list(set(df['Earlier Trademark'].tolist()))
            contested_marks = list(set(df['Contested Trademark'].tolist()))
            all_marks = list(set(earlier_marks + contested_marks))
            for mark in all_marks:
                for c in mark:
                    self.alphabet.append(c)

            self.alphabet = list(set(self.alphabet))


    def get_weights(self):
        if self.weights and not self.recompute_weights:
            return self.weights
        else:
            vgg16 = keras.applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
            self.basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)
            self._alphabet_to_image()
            return self._compute_weights()


    def _alphabet_to_image(self):
        fnt = ImageFont.truetype('arial.ttf', 224)

        # store initial representation
        for i, character in enumerate(self.alphabet):
            file_name = f'{self.img_path}/{i}_image.png'
            image = Image.new(mode = "RGB", size = (300,300), color = "black")
            draw = ImageDraw.Draw(image)
            draw.text((0,0), character, font=fnt, fill=(255,255,255))
            image.save(file_name)

        # normalize representations
        for f in glob(f'{self.img_path}/*_image.png'):
            f = f.replace('\\', '/')
            im = cv2.imread(f)

            h,w,d = im.shape
            b_border = 0
            u_border = 0
            l_border = 0
            r_border = 0

            for i in range(h):
                if np.sum(im[i,:,:]) > 0:
                    b_border = i
                    break

            for i in reversed(range(h)):
                if np.sum(im[i,:,:]) > 0:
                    u_border = i
                    break

            for i in range(w):
                if np.sum(im[:,i,:]) > 0:
                    l_border = i
                    break

            for i in reversed(range(w)):
                if np.sum(im[:,i,:]) > 0:
                    r_border = i
                    break

            if not (b_border == 0 and u_border == 0 and l_border == 0 and r_border == 0):
                im = im[b_border:u_border,l_border:r_border,:]
                cv2.imwrite(f, img=im)

        for f in glob(f'{self.img_path}/*_image.png'):
            f = f.replace('\\', '/')
            im = cv2.imread(f)
            im = cv2.bitwise_not(im)
            cv2.imwrite(f, img=im)

        for f in glob(f'{self.img_path}/*_image.png'):
            f = f.replace('\\', '/')
            im = cv2.imread(f)
            h,w,d = im.shape
            image_with_border = cv2.copyMakeBorder(
                im,
                top=math.ceil(((300 - h) / 2)),
                bottom=math.floor(((300 - h) / 2)),
                left=math.floor(((300 - w) / 2)),
                right=math.ceil(((300 - w) / 2)),
                borderType=cv2.BORDER_CONSTANT,
                value=(255,255,255)
            )
            cv2.imwrite(f, img=image_with_border)

    def _get_feature_vector(self, img):
        img1 = cv2.resize(img, (224, 224))
        feature_vector = self.basemodel.predict(img1.reshape(1, 224, 224, 3))
        return feature_vector

    @staticmethod
    def _calculate_similarity(vector1, vector2):
        return cosine_similarity(vector1, vector2)

    def _compute_weights(self):
        weights = {}
        for a in glob(f'{self.img_path}/*_image.png'):
            a_index = int(a.split('\\')[-1].split('_')[0])
            a_character = self.alphabet[a_index]
            a_file = a.replace('\\', '/')
            weights[a_character] = {}
            for b in glob(f'{self.img_path}/*_image.png'):
                b_index = int(b.split('\\')[-1].split('_')[0])
                b_character = self.alphabet[b_index]
                b_file = b.replace('\\', '/')
                similarity = self._calculate_similarity(self._get_feature_vector(cv2.imread(a_file)),
                                                        self._get_feature_vector(cv2.imread(b_file)))
                weights[a_character][b_character] = similarity

        with open(f'{self.target_path}/weights.pickle', 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return weights