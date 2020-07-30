clear; clc; close all
%% Part01 K-Means =======================================================
File = 'img2.jpeg';
img = im2double(imread(File));
figure
subplot(1,2,1)
imshow(img)
title("origial photo");
T = reshape(img, size(img,1)*size(img,2),3);

% some modification
% for i = 1:size(T,1)
%     if (T(i,1) <= 0.2 && T(i,2) <= 0.2 && T(i,3) <= 0.5)
%         T(i,:) = [1, 1, 0];
%     end
% end
% T = reshape(T, size(img,1), size(img,2),3);
% figure
% imshow(T)
% T = reshape(T, size(img,1)*size(img,2),3);
% title("modified photo");

shape = size(img);
k = 3; % number of clusters
iteration = 5;
tic
cluster = k_means(iteration, k, T, shape);
toc
subplot(1,2,2)
imshow(cluster)
title("clustered photo");

%% Part02 Otsu Algorithm ================================================
clear all; clc; close all;
File = 'Airplane2.jpg';
img = imread(File);
img = img(:,:,3); %----- Must be commented for gray scale images
figure
subplot(2,2,1)
imshow(img)
title("Original Image")
tic
threshold_mat = Otsu(img, 1);
toc
index = find(threshold_mat(1,:) == max(threshold_mat(1,:)));
T = threshold_mat(2,index);
im2 = img;
im2(find(img<=T)) = 0;
im2(find(img>T)) = 255;
subplot(2,2,2)
imshow(im2)
title("Segmented Image")

%% Part05 Comparison ====================================================
clc; clear all; close all;
figure
File = 'Airplane6.jpg';
img = im2double(imread(File));
tic
T = reshape(img, size(img,1)*size(img,2),3);
shape = size(img);
k = 2; % number of clusters
iteration = 5;
cluster = k_means(iteration, k, T, shape);
toc
subplot(1,2,1)
imshow(cluster)
title("K-Means");

img = imread(File);
img = img(:,:,3); %----- Must be commented for gray scale images
tic
threshold_mat = Otsu(img, 0);
index = find(threshold_mat(1,:) == max(threshold_mat(1,:)));
T = threshold_mat(2,index);
im2 = img;
im2(find(img<=T)) = 0;
im2(find(img>T)) = 255;
toc
subplot(1,2,2)
imshow(im2)
title("Otsu")

%% Functions ============================================================
%Otsu =====================================================
function threshold_mat = Otsu(img, is_plot)
    intensity = reshape(img,size(img,1)*size(img,2),1);
    T_min = min(intensity);
    T_max = max(intensity);
    T = T_min:T_max;
    [counts,binLocations] = imhist(img);
    [N,edges] = histcounts(img);
    sigma = [];
    for i = 1:size(T,2)
        T_temp = T(i);
        j = 1:T_temp;
        j = double(j);
        w0 = sum(counts(j)) / sum(counts);
        counts_temp = j' .* counts(j);
        u0 = sum(counts_temp) / w0;
%         u0 = mean(intensity(j));
        j = T_temp+1:255;
        j = double(j);
        w1 = sum(counts(j)) / sum(counts);
        counts_temp = j' .* counts(j);
        u1 = sum(counts_temp) / w1; 
%         u1 = mean(intensity(j));
        sigma(1,i) = w0 * w1 * (u0 - u1)^2;
    end
    threshold_mat = [sigma ; double(T)];
    if (is_plot)
        subplot(2,2,[3 4])
        histogram('BinEdges',edges,'BinCounts',N)
        hold on;
        plot(double(T) , sigma / max(sigma) * max(counts));
        title("inter-variance per threshold")
    end
end

% K-Means ==============================================
function cluster = k_means(iteration, k, T, original_shape)
    random_index = ceil(rand(k,1) * size(T,1)); % This choses k pixels randomly
    c = T(random_index, :); % Algorithm Part01
%     c(1,:) = T(600*1000,:); % For Airplane2.jpg ----------<<<
    container = zeros(size(T,1), k+1); % defining a row of data for each pixel
    for i = 1:iteration
        % Finding distances from each cluster, Algorithm Part02
        for j = 1:size(T,1)
            for m = 1:k
                d_temp = norm(T(j,:) - c(m,:));
                container(j,m) = d_temp;
            end
            % Algorithm Part03, Assigning points to clusters
            [M, I] = min(container(j,1:k));
            % I is the closest center
            container(j,k+1) = I;
        end
        
        % Algorithm Part04, calculating new centers
        for j = 1:k
            index_temp = find(container(:,k+1) == j);
            mean_temp = mean(T(index_temp,:));
            c(j,:) = mean_temp;
        end
        % Algorithm Part05 is to repeat,
    end
    temp = T;
    for j = 1:size(T,1)
        temp(j,:) = container(j,k+1);
    end
    X = zeros(size(T));
    color = [0.312, 0.768, 0.95 ; 0.95, 0.312, 0.456 ; 0.95, 0.908, 0.05];
    for i = 1:k
        index_temp = find(container(:,k+1) == i);
%         X(index_temp,:) = repmat(color(i,:), size(index_temp,1), 1); % repmat(c(i,:),size(idx,1),1);
        X(index_temp,:) = repmat(c(i,:),size(index_temp,1),1);
    end
    cluster = reshape(X,original_shape(1),original_shape(2),3);
    
end


function G = SobelFeldman(A)
    hx = [-1 0 1;-2 0 2 ;-1 0 1];
    hy = hx';
    gx = conv2(A,hx, 'same');
    gy = conv2(A,hy, 'same');
    G = sqrt(gx.^2 + gy.^2);
%     G = uint8(G);
    avg = 3*mean(G,'all');
    G = 1/avg * G;
end