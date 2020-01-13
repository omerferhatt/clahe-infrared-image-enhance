clear all; close all; clc;

clip_size = 8;
image = imread('lena.tif');
image = rgb2gray(image);

clahe_image = clh(image, clip_size, 64, 4, 4);

figure; imshow(uint8(image));
figure; imshow(uint8(clahe_image))
