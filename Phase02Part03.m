%% Loading Data ======================================================
clear all; clc; close all;
load NonTargetEpochs.mat
load TargetEpochs.mat

%% =====================Freq Feature Extraction ===========================
clc
tic
% Frequency Band Energy
% energy1 = freq_feature(EpochData01NonTarget, 'band_energy'); 
% energy2 = freq_feature(EpochData01Target, 'band_energy'); 
% % Median Frequency
% med_freq1 = freq_feature(EpochData01NonTarget, 'median_frequency');
% med_freq2 = freq_feature(EpochData01Target, 'median_frequency');
% 
% 
% % % Frequency Mean
% mean_freq1 = freq_feature(EpochData01NonTarget, 'mean_frequency');
% mean_freq2 = freq_feature(EpochData01Target, 'mean_frequency');
% 
% % % Frequency Median
% med_freq1 = freq_feature(EpochData01NonTarget, 'median_frequency');
% med_freq2 = freq_feature(EpochData01Target, 'median_frequency');
% 
% % Beta Band Power
% beta_power1 = freq_feature(EpochData01NonTarget, 'beta_power');
% beta_power2 = freq_feature(EpochData01Target, 'beta_power');
% 
% % Alpha Band Power
% alpha_power1 = freq_feature(EpochData01NonTarget, 'alpha_power');
% alpha_power2 = freq_feature(EpochData01Target, 'alpha_power');

% Cosine Transform
cosine_transform1 = freq_feature(EpochData01NonTarget, 'cosine_transform');
cosine_transform2 = freq_feature(EpochData01Target, 'cosine_transform');

toc

%% =====================Time Feature Extraction ===========================

%///////-----variance for each 8channles--------\\\\\
   var_ch2 = FeatureVarCh (EpochData01Target);
   var_ch1 = FeatureVarCh (EpochData01NonTarget);
    %///////-----variance for trials--------\\\\\
   var_trial2 = FeatureVarTrial (EpochData01Target);
   var_trial1 = FeatureVarTrial (EpochData01NonTarget);
    %///////-----average for each 8channles--------\\\\\
   avg_ch2 = FeatureAvgCh (EpochData01Target);
   avg_ch1 = FeatureAvgCh (EpochData01NonTarget);
    %///////-----average for trials--------\\\\\
   avg_trial2 = FeatureAvgTrial (EpochData01Target);
   avg_trial1 = FeatureAvgTrial (EpochData01NonTarget);
    %///////-----Correlation mat--------\\\\\
%     corr2 = FeatureCorr(EpochData01Target);
%     corr1 = FeatureCorr(EpochData01NonTarget);


%% =====================Frequency-Time Feature Extraction ===========================
clc
tic
wavelet_transform1 = wavelet(EpochData01NonTarget);
wavelet_transform2 = wavelet(EpochData01Target);
toc
%% ===================== Wavelet Transform for test data ================
clc;
tic
wavelet_transformTest1 = wavelet(TestEpochData1);
cosine_transformTest1 = freq_feature(TestEpochData1, 'cosine_transform');

wavelet_transformTest2 = wavelet(TestEpochData2);
cosine_transformTest2 = freq_feature(TestEpochData2, 'cosine_transform');

wavelet_transformTest3 = wavelet(TestEpochData3);
cosine_transformTest3 = freq_feature(TestEpochData3, 'cosine_transform');

wavelet_transformTest4 = wavelet(TestEpochData4);
cosine_transformTest4 = freq_feature(TestEpochData4, 'cosine_transform');

wavelet_transformTest5 = wavelet(TestEpochData5);
cosine_transformTest5 = freq_feature(TestEpochData5, 'cosine_transform');

wavelet_transformTest6 = wavelet(TestEpochData6);
cosine_transformTest6 = freq_feature(TestEpochData6, 'cosine_transform');

wavelet_transformTest7 = wavelet(TestEpochData7);
cosine_transformTest7 = freq_feature(TestEpochData7, 'cosine_transform');

wavelet_transformTest8 = wavelet(TestEpochData8);
cosine_transformTest8 = freq_feature(TestEpochData8, 'cosine_transform');

wavelet_transformTest9 = wavelet(TestEpochData9);
cosine_transformTest9 = freq_feature(TestEpochData9, 'cosine_transform');
toc

%% Anova

for i=1:8
%    AnovaMeanFreq(i) = Anova1(mean_freq1(:,i), mean_freq2(:,i));
%    AnovaMedFreq(i) = Anova1(med_freq1(:,i), med_freq2(:,i));
%    AnovaBetaPower(i) = Anova1(beta_power1(:,i), beta_power2(:,i));
%    AnovaEnergy(i) = Anova1(energy1(:,i), energy2(:,i));
%    AnovaAlphaPower(i) = Anova1(alpha_power1(:,i), alpha_power2(:,i));
   for j=1:50
        AnovaCT(i,j) = Anova1 (cosine_transform1(j,:,i),cosine_transform2(j,:,i)); 
   end
   for j=1:16
        AnovaWaveletTransform(i,j) = Anova1 (wavelet_transform1(j,:,i),wavelet_transform2(j,:,i)); 
   end
   for j=1:257
      AnovaTimeSignal(i,j) = Anova1 (EpochData01NonTarget(j,:,i) , EpochData01Target(j,:,i)); 
   end
%    AnovaAvgch(i) = Anova1(avg_ch1(:,i), avg_ch2(:,i));
%    AnovaVarch(i) = Anova1(var_ch1(:,i), var_ch2(:,i));
end
% AnovaAvgTrial = Anova1(avg_trial1' , avg_trial2');
% AnovaVarTrial = Anova1(var_trial1' , var_trial2');

%% Finding 
data = load('data3.mat');
data = data.data3;
trial_index = find(ytest == 1);
StimuliOnset = find_stimuli(data.test);
target_index = StimuliOnset(trial_index);
target_val = data.test(10,target_index)';
s1 = target_val(1:30,1);
x1 = mode(s1);
s1 = s1(s1~=x1);
y1 = mode(s1);


%% ======================== Functions ==================================
function out = wavelet(data)
    trial_num = size(data, 2);
    channel_num = size(data, 3);
    out = zeros(16, trial_num, channel_num);
    for trial = 1:trial_num
        for ch = 1:channel_num
            signal = data(:,trial,ch);
            [c, l] = wavedec(signal, 5, 'db2');
            % A5------------------------
            A5 = c(1 : l(1));
            A5_mean = mean(A5);
            A5_max = max(A5);
            A5_min = min(A5);
            A5_std = std(A5);
            % D5-----------------------
            D5 = c(l(1)+1 : l(1)+l(2));
            D5_mean = mean(D5);
            D5_max = max(D5);
            D5_min = min(D5);
            D5_std = std(D5);
            % D4------------------------
            D4 = c(l(1)+l(2)+1 : l(1)+l(2)+l(3));
            D4_mean = mean(D4);
            D4_max = max(D4);
            D4_min = min(D4);
            D4_std = std(D4);
            % D3---------------------------
            D3 = c(l(1)+l(2)+l(3)+1 : l(1)+l(2)+l(3)+l(4));
            D3_mean = mean(D3);
            D3_max = max(D3);
            D3_min = min(D3);
            D3_std = std(D3);
            
            out(1,trial,ch) = A5_mean;
            out(2,trial,ch) = A5_max;
            out(3,trial,ch) = A5_min;
            out(4,trial,ch) = A5_std;
            out(5,trial,ch) = D5_mean;
            out(6,trial,ch) = D5_max;
            out(7,trial,ch) = D5_min;
            out(8,trial,ch) = D5_std;
            out(9,trial,ch) = D4_mean;
            out(10,trial,ch) = D4_max;
            out(11,trial,ch) = D4_min;
            out(12,trial,ch) = D4_std;
            out(13,trial,ch) = D3_mean;
            out(14,trial,ch) = D3_max;
            out(15,trial,ch) = D3_min;
            out(16,trial,ch) = D3_std;
        end
    end
end


function out = freq_feature(data, feature)
    Fs = 256;
    trial_num = size(data, 2);
    channel_num = size(data, 3);
    out = zeros(trial_num,channel_num);
    if (feature == "cosine_transform")
         out = zeros(50,trial_num,channel_num);
     end
    for trial = 1:trial_num
        for ch = 1:channel_num
            if (feature == "mean_frequency")
                signal = data(:,trial,ch);
                mean_freq = meanfreq(signal, Fs);
                out(trial,ch) = mean_freq;
            end
            if (feature == "band_energy")
                signal = data(:,trial,ch);
                N = length(signal);
                signal_fft = fft(signal);
                temp = (1/(2*pi*N)) * abs(signal_fft) .^ 2;
                out(trial,ch) = sum(temp); 
            end
            if (feature == "median_frequency")
                signal = data(:,trial,ch);
                med_freq = medfreq(signal, Fs);
                out(trial,ch) = med_freq;
            end
            if(feature == "beta_power")
                signal = data(:,trial,ch);
                power_temp = bandpower(signal, Fs, [13 30]);
                out(trial,ch) = power_temp;
            end
            if(feature == "alpha_power")
                signal = data(:,trial,ch);
                power_temp = bandpower(signal, Fs, [8 13]);
                out(trial,ch) = power_temp;
            end
            if (feature == "cosine_transform")
                signal = data(:,trial,ch);
                y = dct(signal);
                y = y(1:50);
                out(:,trial,ch) = y;
                
                
            end
        end
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

function PSD(InputSignal, Fs, f)
    X = fft(InputSignal);
    L = length(InputSignal);
    X = X(1:floor(L/2)); % half of the data
    Mag_X = abs(X)/L;
    Mag_X(1:end-1) = Mag_X(1:end-1) * 2;
    Mag_X = (1/(2*pi)) * Mag_X .^ 2;
    fspace = linspace(0,pi,floor(L/2));
    fspace = fspace * Fs/(2*pi);
    if (f<pi)
        CutOff = floor(f/pi * floor(L/2));
        fspace = fspace(1:CutOff);
        Mag_X = Mag_X(1:CutOff);
    end
    plot(fspace,log10(Mag_X));
    title("Fourier Transform");
    xlabel("f(Hz)");
    ylabel("Amplitude");
end
function variance = FeatureVarCh (epoch)

[i j k] = size(epoch);
    for trial = 1:j
       for ch = 1:k
            variance(trial,ch) = var(epoch(:,trial,ch));
       end
    end

end

function avg = FeatureAvgCh (epoch)

[i j k] = size(epoch);
    for trial = 1:j
       for ch = 1:k
            avg(trial,ch) = sum(epoch(:,trial,ch)/i);
       end
    end

end

function variance = FeatureVarTrial (epoch)

    [i j k] = size(epoch);
    for trial = 1:j
       a = reshape(epoch(:,trial,:) , [i*k 1] );
       variance(trial) = var(a);
    end
end

function avg = FeatureAvgTrial (epoch)

    [i j k] = size(epoch);
    for trial = 1:j
       a = reshape(epoch(:,trial,:) , [i*k 1] );
       avg(trial) = sum(a)/(i*k);
    end
end

function cor = FeatureCorr (epoch)
    [i j k] = size(epoch);
    for trial = 1:j
       for x = 1:k
          for y = 1:k
             cor(trial,x,y) = corr(epoch(:,trial,x),epoch(:,trial,y)); 
          end
       end
    end
end



function j=Anova1(data1 , data2)

a1 = size(data1 , 1);
a2 = size(data2 , 1);
u1 = sum(data1)/a1;
u2 = sum(data2)/a2;
u0 = (a1*u1 + a2*u2)/(a1 + a2);
c1 = var(data1);
c2 = var (data2);
j = (abs(u0 - u1) + abs(u0 -u2) )/(c1 + c2);

end


