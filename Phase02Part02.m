%% question1
clear;clc
data1 = load('SubjectData1.mat');
data2 = load('SubjectData2.mat');
data3 = load('SubjectData3.mat');
data4 = load('SubjectData4.mat');
data5 = load('SubjectData5.mat');
data6 = load('SubjectData6.mat');
data7 = load('SubjectData7.mat');
data8 = load('SubjectData8.mat');
data9 = load('SubjectData9.mat');

data1.testindexvalue(1,:) = find(data1.test(10,:));
data1.testindexvalue(2,:) = data1.test(10,data1.testindexvalue(1,:));

data2.testindexvalue(1,:) = find(data2.test(10,:));
data2.testindexvalue(2,:) = data2.test(10,data2.testindexvalue(1,:));

data3.testindexvalue(1,:) = find(data3.test(10,:));
data3.testindexvalue(2,:) = data3.test(10,data3.testindexvalue(1,:));

data4.testindexvalue(1,:) = find(data4.test(10,:));
data4.testindexvalue(2,:) = data4.test(10,data4.testindexvalue(1,:));

data5.testindexvalue(1,:) = find(data5.test(10,:));
data5.testindexvalue(2,:) = data5.test(10,data5.testindexvalue(1,:));

data6.testindexvalue(1,:) = find(data6.test(10,:));
data6.testindexvalue(2,:) = data6.test(10,data6.testindexvalue(1,:));

data7.testindexvalue(1,:) = find(data7.test(10,:));
data7.testindexvalue(2,:) = data7.test(10,data7.testindexvalue(1,:));

data8.testindexvalue(1,:) = find(data8.test(10,:));
data8.testindexvalue(2,:) = data8.test(10,data8.testindexvalue(1,:));

data9.testindexvalue(1,:) = find(data9.test(10,:));
data9.testindexvalue(2,:) = data9.test(10,data9.testindexvalue(1,:));

%% question2

data1.trainindexvalue(1,:) = find(data1.train(10,:));
data1.trainindexvalue(2,:) = data1.train(10,data1.trainindexvalue(1,:));
data1.target(1,:) = find(data1.train(11,:));
data1.target(2,:) = data1.train(10,data1.target(1,:));



data3.trainindexvalue(1,:) = find(data3.train(10,:));
data3.trainindexvalue(2,:) = data3.train(10,data3.trainindexvalue(1,:));
data3.target(1,:) = find(data3.train(11,:));
data3.target(2,:) = data3.train(10,data3.target(1,:));

%% Question 3 

clear;clc
data1 = load('SubjectData1.mat');
data2 = load('SubjectData2.mat');
data3 = load('SubjectData3.mat');
data4 = load('SubjectData4.mat');
data5 = load('SubjectData5.mat');
data6 = load('SubjectData6.mat');
data7 = load('SubjectData7.mat');
data8 = load('SubjectData8.mat');
data9 = load('SubjectData9.mat');


data1 = IndexExtraction(data1);
data2 = IndexExtraction(data2);
data3 = IndexExtraction(data3);
data4 = IndexExtraction(data4);
data5 = IndexExtraction(data5);
data6 = IndexExtraction(data6);
data7 = IndexExtraction(data7);
data8 = IndexExtraction(data8);
data9 = IndexExtraction(data9);


data1.StimuliTarget = data1.Time(1,find(data1.Time(2,:)));
data1.StimuliTarget = data1.StimuliTarget(1,1:4:end);
data1.StimuliNonTarget = data1.Time(1,find(data1.Time(2,:)==0));
data1.StimuliNonTarget = data1.StimuliNonTarget(1,1:4:end);

data2.StimuliTarget = data2.Time(1,find(data2.Time(2,:)));
data2.StimuliTarget = data2.StimuliTarget(1,1:4:end);
data2.StimuliNonTarget = data2.Time(1,find(data2.Time(2,:)==0));
data2.StimuliNonTarget = data2.StimuliNonTarget(1,1:4:end);

data3.StimuliTarget = data3.Time(1,find(data3.Time(2,:)));
data3.StimuliTarget = data3.StimuliTarget(1,1:4:end);
data3.StimuliNonTarget = data3.Time(1,find(data3.Time(2,:)==0));
data3.StimuliNonTarget = data3.StimuliNonTarget(1,1:4:end);

data4.StimuliTarget = data4.Time(1,find(data4.Time(2,:)));
data4.StimuliTarget = data4.StimuliTarget(1,1:4:end);
data4.StimuliNonTarget = data4.Time(1,find(data4.Time(2,:)==0));
data4.StimuliNonTarget = data4.StimuliNonTarget(1,1:4:end);

data5.StimuliTarget = data5.Time(1,find(data5.Time(2,:)));
data5.StimuliTarget = data5.StimuliTarget(1,1:4:end);
data5.StimuliNonTarget = data5.Time(1,find(data5.Time(2,:)==0));
data5.StimuliNonTarget = data5.StimuliNonTarget(1,1:4:end);

data6.StimuliTarget = data6.Time(1,find(data6.Time(2,:)));
data6.StimuliTarget = data6.StimuliTarget(1,1:4:end);
data6.StimuliNonTarget = data6.Time(1,find(data6.Time(2,:)==0));
data6.StimuliNonTarget = data6.StimuliNonTarget(1,1:4:end);

data7.StimuliTarget = data7.Time(1,find(data7.Time(2,:)));
data7.StimuliTarget = data7.StimuliTarget(1,1:4:end);
data7.StimuliNonTarget = data7.Time(1,find(data7.Time(2,:)==0));
data7.StimuliNonTarget = data7.StimuliNonTarget(1,1:4:end);

data8.StimuliTarget = data8.Time(1,find(data8.Time(2,:)));
data8.StimuliTarget = data8.StimuliTarget(1,1:4:end);
data8.StimuliNonTarget = data8.Time(1,find(data8.Time(2,:)==0));
data8.StimuliNonTarget = data8.StimuliNonTarget(1,1:4:end);

data9.StimuliTarget = data9.Time(1,find(data9.Time(2,:)));
data9.StimuliTarget = data9.StimuliTarget(1,1:4:end);
data9.StimuliNonTarget = data9.Time(1,find(data9.Time(2,:)==0));
data9.StimuliNonTarget = data9.StimuliNonTarget(1,1:4:end);

%% Filtering & Epoching =================================================
%finding Fs
clc;
t = data1.train(1,:); % Data1
Ts = t(2) - t(1);
Fs = 1/Ts;
load 'BandPass.mat';

tic % filterign all the data
filtered_signal1 = filtering(data1.train,h);
filtered_signal2 = filtering(data2.train,h);
filtered_signal3 = filtering(data3.train,h);
filtered_signal4 = filtering(data4.train,h);
filtered_signal5 = filtering(data5.train,h);
filtered_signal6 = filtering(data6.train,h);
filtered_signal7 = filtering(data7.train,h);
filtered_signal8 = filtering(data8.train,h);
filtered_signal9 = filtering(data9.train,h);
toc
% tic
% % Epoching of Target Data[1...9]
% EpochData01Target = epoching(filtered_signal1, 200, 800, data1.StimuliTarget);
% EpochData02Target = epoching(filtered_signal2, 200, 800, data2.StimuliTarget); 
% EpochData03Target = epoching(filtered_signal3, 200, 800, data3.StimuliTarget); 
% EpochData04Target = epoching(filtered_signal4, 200, 800, data4.StimuliTarget); 
% EpochData05Target = epoching(filtered_signal5, 200, 800, data5.StimuliTarget); 
% EpochData06Target = epoching(filtered_signal6, 200, 800, data6.StimuliTarget); 
% EpochData07Target = epoching(filtered_signal7, 200, 800, data7.StimuliTarget); 
% EpochData08Target = epoching(filtered_signal8, 200, 800, data8.StimuliTarget); 
% EpochData09Target = epoching(filtered_signal9, 200, 800, data9.StimuliTarget); 
% toc

% tic
% % Epoching of NonTarget Data[1...9]
% EpochData01NonTarget = epoching(filtered_signal1, 200, 800, data1.StimuliNonTarget);
% EpochData02NonTarget = epoching(filtered_signal2, 200, 800, data2.StimuliNonTarget); 
% EpochData03NonTarget = epoching(filtered_signal3, 200, 800, data3.StimuliNonTarget); 
% EpochData04NonTarget = epoching(filtered_signal4, 200, 800, data4.StimuliNonTarget); 
% EpochData05NonTarget = epoching(filtered_signal5, 200, 800, data5.StimuliNonTarget); 
% EpochData06NonTarget = epoching(filtered_signal6, 200, 800, data6.StimuliNonTarget); 
% EpochData07NonTarget = epoching(filtered_signal7, 200, 800, data7.StimuliNonTarget); 
% EpochData08NonTarget = epoching(filtered_signal8, 200, 800, data8.StimuliNonTarget); 
% EpochData09NonTarget = epoching(filtered_signal9, 200, 800, data9.StimuliNonTarget); 
% toc

%% Test Data Epoching =================================================
clear all; clc; close all;
data1 = load('SubjectData1.mat');
data2 = load('SubjectData2.mat');
data3 = load('SubjectData3.mat');
data4 = load('SubjectData4.mat');
data5 = load('SubjectData5.mat');
data6 = load('SubjectData6.mat');
data7 = load('SubjectData7.mat');
data8 = load('SubjectData8.mat');
data9 = load('SubjectData9.mat');
data1 = data1.test;
data2 = data2.test;
data3 = data3.test;
data4 = data4.test;
data5 = data5.test;
data6 = data6.test;
data7 = data7.test;
data8 = data8.test;
data9 = data9.test;

t = data1(1,:); % Data1
Ts = t(2) - t(1);
Fs = 1/Ts;
load 'BandPass.mat';

tic % filterign all the data
filtered_signal1 = filtering(data1,h);
filtered_signal2 = filtering(data2,h);
filtered_signal3 = filtering(data3,h);
filtered_signal4 = filtering(data4,h);
filtered_signal5 = filtering(data5,h);
filtered_signal6 = filtering(data6,h);
filtered_signal7 = filtering(data7,h);
filtered_signal8 = filtering(data8,h);
filtered_signal9 = filtering(data9,h);
toc
tic
StimuliOnset1 = find_stimuli(data1);
StimuliOnset2 = find_stimuli(data2);
StimuliOnset3 = find_stimuli(data3);
StimuliOnset4 = find_stimuli(data4);
StimuliOnset5 = find_stimuli(data5);
StimuliOnset6 = find_stimuli(data6);
StimuliOnset7 = find_stimuli(data7);
StimuliOnset8 = find_stimuli(data8);
StimuliOnset9 = find_stimuli(data9);
toc
tic
TestEpochData1 = epoching(filtered_signal1, 200, 800, StimuliOnset1);
save('TestEpochData1.mat','TestEpochData1');
TestEpochData2 = epoching(filtered_signal2, 200, 800, StimuliOnset2);
save('TestEpochData2.mat','TestEpochData2');
TestEpochData3 = epoching(filtered_signal3, 200, 800, StimuliOnset3);
save('TestEpochData3.mat','TestEpochData3');
TestEpochData4 = epoching(filtered_signal4, 200, 800, StimuliOnset4);
save('TestEpochData4.mat','TestEpochData4');
TestEpochData5 = epoching(filtered_signal5, 200, 800, StimuliOnset5);
save('TestEpochData5.mat','TestEpochData5');
TestEpochData6 = epoching(filtered_signal6, 200, 800, StimuliOnset6);
save('TestEpochData6.mat','TestEpochData6');
TestEpochData7 = epoching(filtered_signal7, 200, 800, StimuliOnset7);
save('TestEpochData7.mat','TestEpochData7');
TestEpochData8 = epoching(filtered_signal8, 200, 800, StimuliOnset8);
save('TestEpochData8.mat','TestEpochData8');
TestEpochData9 = epoching(filtered_signal9, 200, 800, StimuliOnset9);
save('TestEpochData9.mat','TestEpochData9');
toc


%% -----------Finctions-------------------------------------------------
% finding test data stimuli on set
function StimuliOnset = find_stimuli(data)
    stimuli = data(10,:);
    StimuliOnset = find(stimuli);
    index = 1:4:length(StimuliOnset);
    temp = [];
    for i = index
        temp = [temp, StimuliOnset(i)];
    end
    StimuliOnset = temp;
end

%============================Filtering Functions=========================
function filtered_signal = filtering(data,h)
    filtered_signal = [];
    for i = 2:9
        temp = data(i,:);
        output = EEG_filter(temp, h);
        filtered_signal = [filtered_signal ; output];
    end
    filtered_signal = [data(1,:) ; filtered_signal];
end
function filtered_signal = EEG_filter(signal, h) % filtering
    avg = mean(signal);
    filtered_signal = signal - avg;
    filtered_signal = zphasefilter(h, signal);
end
function filtered_signal = zphasefilter(h,x) % filtering with respect to gd
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
function gd = groupdelay(h,N) % finding gd
    n = 0:1:length(h)-1;
    h2 = n .* h;
    gd = real(fft(h2,N) ./ fft(h,N));
    gd = gd(abs(gd) < 100000000);
    gd = gd(~isnan(gd));
    gd = gd(~isinf(gd));
end
%=======================================================================


function newdata = IndexExtraction(data)
%     data.Time(1,:) = find(data.test(10,:));
    data.Time(1,:) = find(data.train(10,:));
    data.Time(2,:) = data.train(11,data.Time(1,:));
    newdata=data;
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

function epoch = epoching(InputSignal, BackwardSamples, ForwardSamples, StimuliOnset)
Fs = 256; %signal is down sampled
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



