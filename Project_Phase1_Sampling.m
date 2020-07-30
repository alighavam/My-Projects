%-------------------------- Sampling Question01
clear all; close all; clc;
Fs = 1000;
%f = pi;
f = (2/10)*pi;
t = 0:1/Fs:1.5-1/Fs;
x = 2*cos(2*pi*200*t) + 3*cos(2*pi*50*t);

SemiBandFFT(x, Fs, f);

%% ------------------------------- Question03
clear all; close all; clc;
n1 = linspace(0,pi/2,1000);
n2 = linspace(pi/2,3/2*pi,1000);
n3 = linspace(3/2*pi,2*pi,1000);

part01 = -(2/pi) * n1 + 1;
part02 = 0*n2;
part03 = (2/pi) * n3 - 3;

n = [n1,n2,n3];
transform = [part01,part02,part03];

p1 = plot(n,transform,'red','LineWidth',2)
xlim([0,2*pi]);
line([pi/3,5*pi/3],[1/3,1/3],'LineWidth',2);
%% --------------------------------Functions
function SemiBandFFT(InputSignal, Fs, f)
X = fft(InputSignal);
L = length(InputSignal);
X = X(1:floor(L/2)); % half of the data
Mag_X = abs(X)/L*2;
fspace = linspace(0,pi,floor(L/2));
fspace = fspace * Fs/(2*pi);
if (f<pi)
    CutOff = floor(f/pi * floor(L/2));
    fspace = fspace(1:CutOff);
    Mag_X = Mag_X(1:CutOff);
end
plot(fspace,Mag_X);
title("Fourier Transform");
xlabel("f(Hz)");
ylabel("Amplitude");
end