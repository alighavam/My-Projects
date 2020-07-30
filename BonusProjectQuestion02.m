clear all; clc; close all
file = 'AsalCircle.jpg';
A = imread (file);
A = rgb2gray (A);
tic
G = SobelFeldman(A);
% G = circle([240, 340], 60, shape);
shape = size(G);
toc
% for i = 1:shape(1)
%     for j = 1:shape(2)
%         if G(i,j) < 0.1
%             G(i,j) = 0;
%         end
%     end
% end
figure
imshow(G);
d = imdistline;

% G_temp = zeros(shape);
% radii = 23;

tic
center_mat = zeros(shape);
finded_centers = {};
cnt = 1;
for radii = [37 72 106 143 178 213] % for circles3.jpg : 20:10:90 ,, for circles.jpg: 22:24
    G_temp = zeros(shape);
    for i = 1:1:shape(1)
        for j = 1:1:shape(2)
            if G(i,j) >= 1
                center = [i,j];
                circle_mat = circle(center, radii, shape);
                G_temp = G_temp + circle_mat;
            end
        end 
    end
    temp_max = max(G_temp,[],'all')
    if (temp_max > 20) % Another important threshold to set --------------<<<<< circles3.jpg:30 ,, circles.jpg:40
        finded_centers{cnt} = find_center(G_temp, radii, 0.7); %-------- Here you can set threshold-------<<<<<
        % for circles3.jpg:0.9 for circles.jpg: 0.6
        
        cnt = cnt + 1;
        center_mat = center_mat + G_temp;
    end
end
toc
figure
maximum = max(center_mat,[],'all');
center_mat = center_mat / maximum;
imshow(center_mat)

figure
shape_finded_centers = size(finded_centers);
final_centers = zeros(shape);
for i = 1:shape_finded_centers(2)
    temp = finded_centers{i};
    temp(shape(1)+1,:) = [];
    final_centers = final_centers + temp;
end
imshow(final_centers)
d1 = imdistline;

% Draw Circles On Original Image
figure
imshow(file)
for i = 1:shape_finded_centers(2)
    temp = finded_centers{i};
    shape_temp = size(temp);
    radii_temp = temp(shape_temp(1),1);
    temp(shape_temp(1),:) = [];
    shape_temp = size(temp);
    [row, col] = find(temp ~= 0);
    row_shape_temp = size(row);
    radii_temp = ones(row_shape_temp(1),1) * radii_temp;
    centers_temp = [col , row];
    h = viscircles(centers_temp,radii_temp);
end

% Finding Number of Circles with clustering
min_dist = 19;
[row, col] = find(final_centers ~= 0);
final_centers_index = [row, col];
cluster = clustering(final_centers_index, min_dist);
x = find(cluster ~= 0);
disp("number of circles = ")
disp(size(x,1))

%% Part02 Matlab Premade funcrions
clc; close all; clear all;
rgb = imread('circles.jpg');
[centersd,radiid] = imfindcircles(rgb,[20 25],'ObjectPolarity','dark', 'sensitivity', 0.95);
[centersb,radiib] = imfindcircles(rgb,[20 25],'ObjectPolarity','bright', 'sensitivity', 0.95);
imshow(rgb)
h = viscircles(centersd,radiid);
h = viscircles(centersb,radiib, 'color', 'b');

%% Functions ------------------------------------------------------------
function G = Kirsch(A)
    g(:,:,1) = [5 5 5; -3 0 -3; -3 -3 -3];
    g(:,:,2) = [5 5 -3; 5 0 -3; -3 -3 -3];  
    g(:,:,3) = [5,-3,-3; 5,0,-3; 5,-3,-3];
    g(:,:,4) = [-3,-3,-3; 5,0,-3; 5,5,-3];
    g(:,:,5) = [-3,-3,-3; -3,0,-3; 5,5,5];
    g(:,:,6) = [-3,-3,-3; -3,0,5; -3,5,5];
    g(:,:,7) = [-3,-3,5; -3,0,5; -3,-3,5];
    g(:,:,8) = [-3,5,5; -3,0,5; -3,-3,-3];

    for i = 1:8
       G(:,:,i) = conv2(A , g(:,:,i), 'same'); 
    end

%     G =uint8( max (G,[],3));
    G = max (G,[],3);
    avg = 8*mean(G,'all');
    G = 1/avg * G;    
end

function cluster = clustering(points, min_dist)
    min_dist = min_dist^2;
    cluster = ones(size(points,1),1);
    for i = 1:size(points,1)-1
        if (cluster(i,1) ~= 0)
            for j = i+1 : size(points,1) 
                if (cluster(j,1) ~= 0)
                    distance = (points(i,1) - points(j,1))^2 + (points(i,2) - points(j,2))^2;
                    if (distance < min_dist)
                        cluster(j,1) = 0;
                    end
                end
            end
        end
    end
end

function finded_centers = find_center(G, radii, threshold)
    maximum = max(G,[],'all');
    G = G / maximum;
    shape = size(G);
    finded_centers = zeros(shape);
    for i = 1:shape(1)
        for j = 1:shape(2)
            if (G(i,j) >= threshold)
                finded_centers(i,j) = 1;
            end
        end
    end
    radii_vector = ones(1,shape(2)) * radii;
    finded_centers = [finded_centers ; radii_vector];
end


function circle_mat = circle(center, radii, shape)
    circle_mat = zeros(shape);
    for theta = 0:pi/50:2*pi
        xunit = radii * cos(theta) + center(1);
        yunit = radii * sin(theta) + center(2);
        if (xunit<=shape(1) && xunit>0 && yunit>0 && yunit<=shape(2))
            circle_mat(ceil(xunit), ceil(yunit)) = 1;
        end
%         xunit = radii * cos(theta) + center(1) + 1;
%         yunit = radii * sin(theta) + center(2) + 1;
%         if (xunit<=shape(1) & xunit>0 & yunit>0 & yunit<=shape(2))
%             circle_mat(ceil(xunit), ceil(yunit)) = 1;
%         end
%         xunit = radii * cos(theta) + center(1) - 1;
%         yunit = radii * sin(theta) + center(2) - 1;
%         if (xunit<=shape(1) & xunit>0 & yunit>0 & yunit<=shape(2))
%             circle_mat(ceil(xunit), ceil(yunit)) = 1;
%         end
    end
end

function G = SobelFeldman (A)
    hx = [-1 0 1;-2 0 2 ;-1 0 1];
    hy = hx';
    gx = conv2(A,hx, 'same');
    gy = conv2(A,hy, 'same');
    G = sqrt(gx.^2 + gy.^2);
%     G = uint8(G);
    avg = 3*mean(G,'all');
    G = 1/avg * G;
end
