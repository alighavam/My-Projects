%% Question01 ============================================================
clc; clear all; close all;
Fs = 1000;
load test_filter.mat;
h = Num;
t = 0:1/Fs:5;
x = (exp(-(t-3).^2) .* cos(2*pi*5*t) + exp(-(t-3).^2) .* cos(2*pi*10*t + 1))/1000;
y = filter(h,1,x);
subplot(1,2,1)
plot(1:length(x),x);
title("Original signal")
subplot(1,2,2)
plot(1:length(x),y)
title("Filtered signal")

[gd, omega] = grpdelay(h,1,100);
gd1 = groupdelay(h,100);

%% Question02 ============================================================
clc; clear all; close all;
Fs = 1000;
load test_filter.mat;
h = Num;
t = 0:1/Fs:5;
x = (exp(-(t-3).^2) .* cos(2*pi*5*t) + exp(-(t-3).^2) .* cos(2*pi*10*t + 1))/1000;
y = zphasefilter(h,x);
subplot(1,2,1)
plot(t,x);
title("original signal")
subplot(1,2,2)
plot(t,y);
title("filtered signal + groupdelay modification")




%% Functions =============================================================
function filtered_signal = zphasefilter(h,x)
    filtered_signal = filter(h,1,x);
    gd = groupdelay(h,length(h)+100);
    gd = gd(1);
    gd = floor(gd);
    if (gd > 0)
        temp = filtered_signal(1+gd:end);
        filtered_signal(1:end-gd) = temp;
    end
    if (gd < 0)
        temp = filtered_signal(1:end-abs(gd));
        filtered_signal(abs(gd):end) = temp; 
    end
end

function gd = groupdelay(h,N)
    n = 0:1:length(h)-1;
    h2 = n .* h;
    gd = real(fft(h2,N) ./ fft(h,N));
    gd = gd(abs(gd) < 100000000);
    gd = gd(~isnan(gd));
    gd = gd(~isinf(gd));
end

function h = BPF(L,lowF,higF,FS,plot)
    %{
        Inputs:
            L: Filter order
            lowF: Low frequency
            higF: High frequency
            FS: Sampling frequency
            plot: To plot or not
            if nothing is entered for "plot" then BPF will not plot
        Output:
            h: Impulse response of the filter
    %}
    beta = 3;
    h = fir1(L-1,[2*lowF/FS,2*higF/FS], kaiser(L,beta));
    h = h(:);
    if nargin==5
    figure
    freqz(h)
    end
end

function SemiBandFFT(InputSignal, Fs, f)
    X = fft(InputSignal);
    L = length(InputSignal);
    X = X(1:floor(L/2)); % half of the data
    Mag_X = abs(X)/L;
    Mag_X(1:end-1) = Mag_X(1:end-1) * 2;
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