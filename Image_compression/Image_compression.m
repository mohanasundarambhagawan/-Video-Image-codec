% Load and convert the source image to grayscale
original_image = imread('natural.jpg');
gray_image = rgb2gray(original_image);
gray_image = im2double(gray_image);
[width, length] = size(gray_image);
imwrite(gray_image, 'natural_gray_image.jpg');

% Discrete Cosine Transform (DCT) on 8x8 macro blocks
DCT_image = blkproc(gray_image, [8 8], 'dct2');
DCT_final = ceil(DCT_image * 1000);

% Quantization
Q = 4;
Quantized = ceil(DCT_final / Q);

% Huffman encoding
[g, ~, intensity_val] = grp2idx(Quantized(:));
Frequency = accumarray(g, 1);
probability = Frequency / (width * length);
dict = huffmandict(intensity_val, probability);
encoded_image = huffmanenco(Quantized(:), dict);

% Save encoded data to a text file
file3 = fopen('compressed_image_data.txt', 'w');
[k, ~] = size(encoded_image);
for c = 1:k
    fprintf(file3, '%d', encoded_image(c));
end
fclose(file3);

% Huffman decoding
decoded_image = huffmandeco(encoded_image, dict);
re_image = reshape(decoded_image, [width, length]);

% Inverse Quantization
IDCT = re_image * Q;

% Inverse DCT on 8x8 blocks
IDCT = IDCT / 1000;
compressed_image = blkproc(IDCT, [8 8], 'idct2');

% Display images and calculate PSNR
figure, imshow(original_image), title('Original Image');
figure, imshow(gray_image), title('Grayscale Image');
figure, imshow(compressed_image), title('Compressed Image (Q=4)');
figure, imshow(DCT_image), title('DCT Image');

PSNR_DECODE_IMAGE = psnr(compressed_image, gray_image);
disp(['PSNR of the Decoded Image: ' num2str(PSNR_DECODE_IMAGE)]);
