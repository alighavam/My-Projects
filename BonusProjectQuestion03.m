%% part3
clear;clc;close all
A = imread ('ballon.jpg');
B = im2double(A);
saltpepper = imnoise(A,'salt & pepper',0.1);
gaussian = imnoise(A,'gaussian',0,0.05);
poisson = imnoise(A,'poisson');
speckle = imnoise(A,'speckle',0.05);
figure
imshowpair(A,saltpepper,'montage')
title('salt & pepper');

figure
imshowpair(A,gaussian,'montage')
title('gaussian');

figure
imshowpair(A,poisson,'montage')
title('poisson');

figure
imshowpair(A,speckle,'montage')
title('speckle');
%% salt & pepper
filt1 = MedianFilter(saltpepper,7);
figure
imshowpair(saltpepper,filt1,'montage')

filt2 = GaussianFilter(saltpepper , 13 , 5);
figure
imshowpair(saltpepper,filt2,'montage')


%% gaussian
filt1 = MedianFilter(gaussian,13);
figure
imshowpair(gaussian,filt1,'montage')
title('Median')

filt2 = GaussianFilter(gaussian , 9 , 5);
figure
imshowpair(gaussian,filt2,'montage')
title('Gaussian')

%% poisson

filt1 = MedianFilter(poisson,9);
figure
imshowpair(poisson,filt1,'montage')
title('Median')

filt2 = GaussianFilter(poisson , 9 , 5);
figure
imshowpair(poisson,filt2,'montage')
title('Gaussian')

%% speckle

filt1 = MedianFilter(speckle,9);
figure
imshowpair(speckle,filt1,'montage')
title('Median')

filt2 = GaussianFilter(speckle , 8 , 5);
figure
imshowpair(speckle,filt2,'montage')
title('Gaussian')
%% ------functions--------
function newA = ClearZero(B,num_zero)
    [i j k] = size(B);
    
    
    for m=1:k
        newA(:,:,m) = B(num_zero+1:i-num_zero , num_zero+1:j-num_zero,m);
        
        
    end
end
function newA = PutZero(B,num_zero)
    [i j k] = size(B);
    
    
    for m=1:k
        c = zeros(num_zero,j);
        a(:,:,m) = [ c ; B(:,:,m) ;c];
        c = zeros( i+2*num_zero , num_zero);
        newa(:,:,m) = [c a(:,:,m) c];
        newA(:,:,m) = newa(:,:,m);
    end
end
function filt = MedianFilter( G1 , n)
    
    num_zero = floor(n/2);
    newA = PutZero(G1,num_zero);
    [r c k] = size(G1);
    filt = [];
   
    for z=1:k
       for i=1+num_zero:r+num_zero
           for j=1+num_zero:c+num_zero
               temp =1;
               for x = -num_zero:num_zero
                   for y = -num_zero:num_zero 
                         Med(temp) = newA(i-x,j-y,z);
                         temp = temp+1;
                   end
               end
               Med = sort(Med);
               filt(i-num_zero,j-num_zero,z) = Med(ceil(size(Med,2)/2));
           end

       end 
        z
    end
    
    filt = uint8(filt);
end

function filt = GaussianFilter (G1 , n ,sigma)
    num_zero = floor(n/2);
    kernel =[];
    for i =-num_zero : num_zero

        for j =-num_zero : num_zero
            kernel(i+num_zero+1,j+num_zero+1) = (1/(2*pi*sigma^2)) * exp((-1/(2*sigma^2))*(i^2+j^2));
       end
    end
    filt = [];
    x = sum(sum(kernel));
    kernel = kernel/x;
    newA = PutZero(G1,num_zero);
    for i=1:3
        filt(:,:,i) = conv2(newA(:,:,i) , kernel , 'same');
    end
    filt = ClearZero(filt , num_zero);
    filt = uint8(filt);
end