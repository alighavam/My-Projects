%% ------------------------------Section01 
close all; clc; clear all;
pic1 = imread('pic1.png');
pic2 = imread('pic2.png');

subplot(1,2,1)
imshow(pic1)
title('pic1')
subplot(1,2,2)
imshow(pic2)
title('pic2')

FT1 = fft2(pic1);
FT1_phase = angle(FT1);

FT2 = fft2(pic2);
FT2_abs = abs(FT2);


NewImage = uint8(ifft2(FT2_abs .* exp(1i .* FT1_phase)));
figure, imshow(NewImage), title('Phase of pic1 with Magnitude of pic2')

%% ---------------------------Section02 fMRI
clear all; close all; clc;
data = load('fmri.mat');
data = data.image;

img1 = data(:,:,1);
img2 = data(:,:,2);


%----------------------------Finding Theta
iteration = 2500;
theta = 60;
alpha = 5;
epsilon = 0.01;
error_mat = [];
tic
for i = 1:iteration
    x1 = theta - epsilon;
    x2 = theta + epsilon;
    y1 = corr2(img1, imrotate(img2, x1, 'crop'));
    y2 = corr2(img1, imrotate(img2, x2, 'crop'));
    gradient = (y2-y1)/(x2-x1);
    theta = theta + alpha * gradient + 5*randn;
    corr = corr2(img1, imrotate(img2, theta, 'crop'));
    error = 0.97 - corr;
    error_mat = [error_mat , error];
    if (error < 0.01)
        break;
    end
end
toc
img2 = imrotate(img2, theta, 'crop'); % angle aligned image
corr
theta
figure;
plot(error_mat);
title('error minimazation for theta');
xlabel('iteration');
ylabel('error');

%----------------------------Finding x shift
alpha = 0.01;
x = 0;
error_mat = [];
tic
for i = 1:iteration
    x1 = x - epsilon;
    x2 = x + epsilon;
    y1 = corr2(img1, imtranslate(img2, [x1 0]));
    y2 = corr2(img1, imtranslate(img2, [x2 0]));
    gradient = (y2-y1)/(x2-x1);
    x = x + alpha * gradient + 0.005*randn;
    corr = corr2(img1, imtranslate(img2, [x 0]));
    error = 0.98 - corr;
    error_mat = [error_mat , error];
    if (error < 0.01)
        break;
    end
end
toc
corr
x
figure;
plot(error_mat);
title('error minimazation for x shift'); 
xlabel('iteration');
ylabel('error');

%----------------------------Finding y shift
alpha = 0.01;
y = 0;
error_mat = [];
tic
for i = 1:iteration
    x1 = y - epsilon;
    x2 = y + epsilon;
    y1 = corr2(img1, imtranslate(img2, [0 x1]));
    y2 = corr2(img1, imtranslate(img2, [0 x2]));
    gradient = (y2-y1)/(x2-x1);
    y = y + alpha * gradient + 0.005*randn;
    corr = corr2(img1, imtranslate(img2, [y 0]));
    error = 0.98 - corr;
    error_mat = [error_mat , error];
    if (error < 0.01)
        break;
    end
end
toc
corr
y
figure;
plot(error_mat);
title('error minimazation for y shift');
xlabel('iteration');
ylabel('error');

img2 = imtranslate(img2,[x y]); %------- final aligned image

%-------------------plotting images
figure;
subplot(1,3,1);
imshow(img1)
title('Image01')
subplot(1,3,2)
imshow(data(:,:,2))
title('Image02')
subplot(1,3,3)
imshow(img2)
title('Aligned Image')
%% --------------------------------------- Section03 loading files
clear all; close all; clc;
[tone01,Fs] = audioread('DialedSequence_NoNoise.wav');
[tone02,Fs] = audioread('DialedSequence_SNR00dB.wav');
[tone03,Fs] = audioread('DialedSequence_SNR10dB.wav');
[tone04,Fs] = audioread('DialedSequence_SNR20dB.wav');
[tone05,Fs] = audioread('DialedSequence_SNR30dB.wav');
tones = {tone01 , tone02 , tone03 , tone04 , tone05};
filters = load('Section03_Filters.mat'); % loading designed filters into workspace
filters = struct2cell(filters);

%% ------------------------------------ Plotting desired plots
% plotting frequency response of filters
for i = 1:8;
    filt = filters{i};
    [h w] = freqz(filt);
    f = w/2/pi * Fs;
    subplot(2,4,i)
    semilogy(f, abs(h));
    xlim([0 4000])
    xlabel("frequency(Hz)")
    g = abs(h);
    MaxPassF = f(g == max(g));
    title("maximum f_{pass} = " + num2str(MaxPassF) + "Hz");
end

figure;
% plotting filtered signasl along original signals
for i = 1:5;
    tone = tones{i};
    [filtered_signal , time] = DTMF_Filter(tone, filters);
    subplot(2,5,i) , plot(time,tone)
%     subplot(2,5,i) , SemiBandFFT(tone, Fs, pi);
    title("tone " + num2str(i))
    xlabel("time")
    ylabel("amplitude")
    subplot(2,5,i+5) , plot(time,filtered_signal)
%     subplot(2,5,i+5) , SemiBandFFT(filtered_signal, Fs, pi);
    title("filtered tone " + num2str(i))
    xlabel("time")
    ylabel("amplitude")
end

%% --------------------------------- Processing
clear all; close all; clc;
[tone01,Fs] = audioread('DialedSequence_NoNoise.wav');
[tone02,Fs] = audioread('DialedSequence_SNR00dB.wav');
[tone03,Fs] = audioread('DialedSequence_SNR10dB.wav');
[tone04,Fs] = audioread('DialedSequence_SNR20dB.wav');
[tone05,Fs] = audioread('DialedSequence_SNR30dB.wav');
tones = {tone01 , tone02 , tone03 , tone04 , tone05};
filters = load('Section03_Filters.mat'); % loading designed filters into workspace
filters = struct2cell(filters);

output = {};
for i = 1:5;
    tone = tones{i};
    [filtered_signal , time] = DTMF_Filter(tone, filters);
    output{i} = filtered_signal;
end


for tone = 1:5
    epoch = epocher(output{tone}, Fs, tone);
    number = [];
    for i = 1:16
        number = [number , tone2number(epoch{i}, Fs)];
    end
    display("Tone " + num2str(tone) + " output: " + number)
end

% tone = tones{1};
% [filtered_signal , time] = DTMF_Filter(tone, filters);
% plot(time, tone01)

%% -----------------------------------Functions
function number = tone2number(signal, Fs)
    [Fourier , fspace] = SemiBandFFT(signal, Fs, pi);
    max1 = max(Fourier);
    max2 = max(Fourier(Fourier<max(Fourier)));
    Index1 = find(Fourier == max1);
    Index2 = find(Fourier == max2);
    f1 = fspace(Index1);
    f2 = fspace(Index2);
    if (f1 > f2)
        swap = f1;
        f1 = f2;
        f2 = swap;
    end
    freq_low = [697, 770, 852, 941];
    freq_high = [1209, 1336, 1477, 1633];
    for i = 1:4
        if (f1 > freq_low(i) - 15) && (f1 < freq_low(i) + 15)
            Padx = i;
            for j = 1:4
                if (f2 > freq_high(j) - 15) && (f2 < freq_high(j) + 15)
                    Pady = j;
                end
            end
        end
    end
    Pad = ['1','2','3','A' ; '4','5','6','B' ; '7','8','9','C' ; '*','0','#','D'];
    number = Pad(Padx, Pady);
end

function epoch = epocher(signal, Fs, threshold_index)
    param = [0.1 0.65 0.15 0.1 0.1];
    threshold_param = param(threshold_index);
    L = round(Fs * 0.02); % 0.02s --> number of indexes
    interval_num = round(length(signal) / L);
    temp = signal(1:L);
    energy = [sum(temp .^ 2)];
    for i = 1:interval_num-2
        Index1 = i * L;
        Index2 = Index1 + L;
        temp = signal(Index1 : Index2);
        e = sum(temp .^ 2);
        energy = [energy , e];
    end
    i = 1;
    avg = mean(energy);
    threshold = threshold_param * avg;
    epoch = {};
    epoch_cnt = 0;
    while (i <= length(energy))
        if (energy(i) >= threshold)
            temp = [];
            epoch_cnt = epoch_cnt + 1;
            j = i;
            while (energy(j) >= threshold)
                Index1 = (j-1) * L;
                Index2 = j * L;
                sig = signal(Index1 : Index2);
                temp = [temp ; sig];
                j = j + 1;
            end
            epoch{epoch_cnt} = temp;
            i = j;
        end
        i = i + 1;
    end
end

function [filtered_signal , time] = DTMF_Filter(tone, filters)
    time = linspace(0, length(tone)/8192,length(tone));
    filtered_signal = 0;
    for i = 1:8
        h = filters{i};
        temp = filter(h, 1, tone);
        filtered_signal = filtered_signal + temp;
    end
end


function [Mag_X , fspace] = SemiBandFFT(InputSignal, Fs, f)
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
%     plot(fspace,Mag_X);
%     title("Fourier Transform");
%     xlabel("f(Hz)");
%     ylabel("Amplitude");
end













