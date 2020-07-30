%-------------------------------------EEG Question01
% loading subject data
clear all; close all; clc;
data = load('Subject1.mat');
data = data.SubjectData;

%finding Fs
t = data(1,:);
Ts = t(2) - t(1);
Fs = 1/Ts;


%-------------------------------------Question03
f_filter = pi;
for i = 1:8
    electrode_data = data(i+1,:);
    subplot(2,4,i);
    SemiBandFFT(electrode_data, Fs, f_filter);
    title("Channel " + num2str(i));
end

%-------------------------------------Question04
electrode01 = data(2,:);
power = [];
for freq = 0.1:0.1:70
    power = [power, bandpower(electrode01,Fs,[0 freq])];
end
freq = 0.1:0.1:70;
figure, plot(freq,power, 'LineWidth', 2);
title("Accumulative Power");
xlabel("f(Hz)");
ylabel("power");

%% -----------------------------------Question05
% loading subject data
clear all; close all; clc;
data = load('Subject1.mat');
data = data.SubjectData;

%finding Fs
t = data(1,:);
Ts = t(2) - t(1);
Fs = 1/Ts;

f_filter = pi;

signal = data(2,:);
subplot(2,1,1);
SemiBandFFT(signal, Fs, f_filter); % plotting frequency domain
title("channel 1");

subplot(2,1,2);
output = EEG_filter(signal, Fs); % filtering the signal
SemiBandFFT(output, Fs, f_filter); % plotting frequency domain
title("filtered channel 1");

%% ------------------------------------Question06
clear all; close all; clc;
% loading subject data
clear all; close all; clc;
data = load('Subject1.mat');
data = data.SubjectData;

%finding Fs
t = data(1,:);
Ts = t(2) - t(1);
Fs = 1/Ts;
L = length(t);

% filtering 256Hz signal:
filtered_signal = [];
for i = 2:9
    temp = data(i,:);
    output = EEG_filter(temp,Fs);
    filtered_signal = [filtered_signal ; output];
end

% Reducing Fs to 1/3 Fs
reduced_data = [];
temp = data(1,:);
j = 1:3:L;
for k = j
    reduced_data = [reduced_data, temp(k)];
end

for i = 1:8
    x = [];
    temp = filtered_signal(i,:);
    for k = j
        x = [x, temp(k)];
    end
    reduced_data = [reduced_data ; x]; % adding data columns
end
 
% plotting reduced signals
Fs = 256/3;
for i = 2:9
    signal = reduced_data(i,:);
    subplot(2,4,i-1);
    SemiBandFFT(signal,Fs,pi);
    title("Reduced Channel" + num2str(i-1));
end
save('reduced_data.mat','reduced_data'); % saving reduced data to use later


%% ---------------------------------------- Question07 epoching and 1000ms window
clear all; close all; clc;
% loading subject data
clear all; close all; clc;
data = load('Subject1.mat');
data = data.SubjectData;

%loading reduced Fs data
reduced_data = load('reduced_data.mat');
reduced_data = reduced_data.reduced_data;

%finding StimuliOnset vector,, indexes without repeatition 
stimuli = data(10,:);
StimuliOnset = find(stimuli);
index = 1:4:10800;
temp = [];
for i = index
    temp = [temp, StimuliOnset(i)];
end
StimuliOnset = temp;
StimuliOnset = round(StimuliOnset/3);

epoch = epoching(reduced_data, 200, 800, StimuliOnset); %this function is found in function section
save('epoch.mat','epoch');

%% -----------------------------------Question08 band energy
clear all; close all; clc;
epoch = load('epoch.mat');
epoch = epoch.epoch;
Fs = 256/3;
% SemiBandFFT(epoch(:,1,5),Fs,pi);

y = freqband(epoch, 2.5, 10, Fs);
Energy = zeros(8,2700);
for ch = 1:8;
    for trial = 1:2700
        Energy(ch,trial) =  sum(y(:,trial,ch) .* y(:,trial,ch));
    end
end
plot(Energy(1,:))
title("Channel1 Energy per trial")
xlabel("trial");
ylabel("energy");
sum(Energy(1,:))

%% --------------------------------Functions
function y = freqband(x, passband1, passband2, Fs)
h = BPF(86,2,15,Fs);
y = zeros(86, 2700, 8);
tic
for ch = 1:8
    for trial = 1:2700
        y(:,trial,ch) = filter(h,1,x(:,trial,ch));
    end
end
toc
end

function epoch = epoching(InputSignal, BackwardSamples, ForwardSamples, StimuliOnset)
Fs = 256/3; %signal is down sampled
Ts = 1/Fs;
a1 = round((BackwardSamples/1000)/Ts); % data points to extract before stimuli
a2 = round((ForwardSamples/1000)/Ts); % data points to extract after stimuli

% building time-trial matrix
t = InputSignal(1,:); % time portion of input data
time_trial = [];
for i = StimuliOnset
    index1 = i - a1;
    index2 = i + a2;
    time_trial = [time_trial ; t(index1 : index2)];
end
epoch = zeros(a1+a2+1,length(StimuliOnset),8);
%filling epoch matrix with data
tic
for ch = 2:9 % channel index
    channel = InputSignal(ch,:);
    for trial = 1:length(StimuliOnset) % trial index
        for i = 1:a1+a2+1
            index = find(t == time_trial(trial,i));
            epoch(i,trial,ch-1) = channel(index);
        end
    end
end
toc
end

function filtered_signal = EEG_filter(signal, Fs) % bandpass filter 1 to 39.5 Hz
avg = mean(signal);
filtered_signal = signal - avg;
filtered_signal = bandpass(filtered_signal,[1 39.5], Fs);
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