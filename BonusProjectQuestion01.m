%% Question01
clear;clc;close all
A = imread ('Jaguar.jpg');
A = rgb2gray (A);
tic
G1 = SobelFeldman(A);
toc
figure
imshow(G1);
d = imdistline;
title("Sobel")
tic
G2 = Kirsch(A);
toc
figure
imshow (G2);
title("Kirsch")

corr2(G1,G2)
%% functions --------------------------------------------------
function G = SobelFeldman (A)
    hx = [-1 0 1;-2 0 2 ;-1 0 1];
    hy = hx';
    gx = conv2(A,hx, 'same');
    gy = conv2(A,hy, 'same');
    G = sqrt(gx.^2 + gy.^2);
%     G = uint8(G);
    avg = 8*mean(G,'all');
    G = 1/avg * G;
end    

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