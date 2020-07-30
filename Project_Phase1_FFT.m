%% section1
%% QUestion01:fft mag & phase 
clear;clc;close all
data = load ('y.mat');
y = data.y;
Y = fft (y);
L = length (y);
n =( 1 : L)*2*pi/L;
mag = abs(Y);
phase = angle (Y);

plot (n , mag);
title (" fft Magnitude ")
xlabel ("f(Hz)")
figure()
plot (n , phase);
title ( " fft Phase")
xlabel ("f(Hz)")

mid = floor(L/2);

check_mag =sum( ( mag (2:mid) - mag(L:-1:mid+3) ) .^2);
check_phase =sum( ( phase (2:mid) + phase(L:-1:mid+3) ) .^2);

if (check_mag ==0)
    disp("Magnitude is even")
else
    disp("Magnitude is not even")
end

if (check_phase ==0)
    disp("phase is odd")
else
    disp("phase is not odd")
end

%% Question02 filtering the noises

clear;clc;close all
data = load ('y.mat');
y = real(data.y);
Y = fft (y);
L = length (y);
n =( 1 : L ) * 2*pi/L;
mag = abs(Y);
phase = angle (Y);

plot (n , mag);
title (" fft Magnitude without noise ")
xlabel ("f(Hz)")
figure()
plot (n , phase);
title ( " fft Phase without noise")
xlabel ("f(Hz)")

mid = floor(L/2);

check_mag =sum( ( mag (2:mid) - mag(L:-1:mid+3) ) .^2);
check_phase =sum( ( phase (2:mid) + phase(L:-1:mid+3) ) .^2);

if (check_mag ==0)
    disp("Magnitude is even")
else
    disp("Magnitude is not even")
end

if (check_phase ==0)
    disp("phase is odd")
else
    disp("phase is not odd")
end

