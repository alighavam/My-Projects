clear all; close all; clc;
%------------------------------ Question01
data = load('64channeldata.mat');
data = data.data;
Fs = 600;

% signal = signal - avg
for ch = 1:63
    for trial = 1:44
        data(ch,:,trial) = data(ch,:,trial) - mean(data(ch,:,trial));
    end
end

for i = 1:6
    subplot(2,3,i);
    SemiBandFFT(data(54,:,7*i),Fs,pi);
%     plot(data(54,:,7*i))
    title("channel54 trial " + num2str(i*7));
%     xlim([0 80])
end

%% ---------------------------- Question02
clear all; close all; clc;
data = load('64channeldata.mat');
data = data.data;
Fs = 600;
Fs2 = 120;

index = 1:5:1800;
reduced_64channeldata = zeros(63,360,44);
for ch = 1:63
    for trial = 1:44
        reduced_64channeldata(ch,:,trial) = data(ch,index,trial);
    end
end
save('reduced_64channeldata.mat','reduced_64channeldata');
% comparing datas to check if there is any mistakes
for i = 1:5
    subplot(2,5,i)
    SemiBandFFT(data(54,:,i*8),Fs,pi);
%     plot(data(54,:,i*8))
    title("Channel54 Trial " + num2str(i*8));
    xlim([0 60]);
end
for i = 1:5
    subplot(2,5,i+5)
    SemiBandFFT(reduced_64channeldata(54,:,i*8),Fs2,pi);
%     plot(reduced_64channeldata(54,:,i*8))
    title("Reduced Data Channel54 Trial " + num2str(i*8));
end

%% ----------------------------------------- Question03
clear all; close all; clc;
data = load('reduced_64channeldata.mat');
data = data.reduced_64channeldata;
correlation_mat = zeros(63,63);

% connecting the trials
full_signal = [];
for ch = 1:63
    temp = [];
    for trial = 1:44
        temp = [temp, data(ch,:,trial)];
    end
    full_signal = [full_signal; temp];
end

% finding correlation matrix
tic
for i = 1:63
    for j = 1:63
        numerator = sum(full_signal(i,:) .* full_signal(j,:));
%         denominator = sqrt(sum(full_signal(i,:) .* full_signal(i,:)) * sum(full_signal(j,:) .* full_signal(j,:)));
        denominator = sqrt(sum(full_signal(i,:) .^2) * sum(full_signal(j,:) .^2));
        correlation_mat(i,j) = numerator / denominator;
    end
end
toc

%---------------------------------------------------------------------------------
Cluster = CorrelationCluster(correlation_mat, 0.3, 2);
%--------------------------------------------------------------------------------

% modifying Cluster
decoy = 999;
[row col] = find(Cluster(1,:) == decoy);
modified_cluster = Cluster;
modified_cluster(:,col) = [];
[row col] = find(modified_cluster == 1);
clus = zeros(63,length(modified_cluster(1,:)));
[GC,GR] = groupcounts(col);
k = 1;
for i = 1:length(GR)
    clus(1:GC(i),i) = row(k:k+GC(i)-1);
    k = k+GC(i);
end

%EEGPlot(clus); %The function we used from another group

%% ---------------------------------------------- Question04
clear all; close all; clc;
data = load('epoch.mat');
data = data.epoch;
data = data(2:end,:);
correlation_mat = zeros(8,8);

% connecting the trials
% full_signal = [];
% for ch = 1:8
%     temp = [];
%     for trial = 1:2700
%         temp = [temp, data(:,trial,ch)];
%     end
%     full_signal = [full_signal; temp];
% end

% finding correlation matrix
full_signal=data;
tic
for i = 1:8
    for j = 1:8
        numerator = sum(full_signal(i,:) .* full_signal(j,:));
        denominator = sqrt(sum(full_signal(i,:) .* full_signal(i,:)) * sum(full_signal(j,:) .* full_signal(j,:)));
        correlation_mat(i,j) = numerator / denominator;
    end
end
toc

%---------------------------------------------------------------------------------
Cluster = CorrelationCluster_for8channel(correlation_mat, 0.2, 1);
%--------------------------------------------------------------------------------

% modifying Cluster
decoy = 999;
[row col] = find(Cluster(1,:) == decoy);
modified_cluster = Cluster;
modified_cluster(:,col) = [];
[row col] = find(modified_cluster == 1);
clus = zeros(8,length(modified_cluster(1,:)));
[GC,GR] = groupcounts(col);
k = 1;
for i = 1:length(GR)
    clus(1:GC(i),i) = row(k:k+GC(i)-1);
    k = k+GC(i);
end

%% -------------------------------- Functions
function Cluster = CorrelationCluster_for8channel(InputCorrMat, DistanceMeasure, method);
    distance = ones(8,8) - abs(InputCorrMat);
    distance = distance+ 5*tril(ones(8,8));
    Cluster = zeros(8,8);
    Cluster = Cluster + eye(8,8);
    decoy = 999 * ones(8,1);
    MinDist = min(min(distance));
    while(MinDist < DistanceMeasure)
    [row col] = find(distance == MinDist);
    if (method == 1)
        distance_update = dist1(Cluster, InputCorrMat, distance, row, col, 8);
    end
    if (method == 2)
        distance_update = dist2(Cluster, InputCorrMat, distance, row, col, 8);
    end
    Cluster(:,row) = Cluster(:,row) + Cluster(:,col);
    Cluster(:,col) = decoy;
    distance = distance_update;
    MinDist = min(min(distance));
%         MinDist = 1; %----------
%         Cluster = distance; %---------
    end
end

function Cluster = CorrelationCluster(InputCorrMat, DistanceMeasure, method);
    distance = ones(63,63) - abs(InputCorrMat);
    distance = distance+ 5*tril(ones(63,63));
    Cluster = zeros(63,63);
    Cluster = Cluster + eye(63,63);
    decoy = 999 * ones(63,1);
    MinDist = min(min(distance));
    while(MinDist < DistanceMeasure)
        [row, col] = find(distance == MinDist);
        if (method == 1)
            distance_update = dist1(Cluster, InputCorrMat, distance, row, col, 63);
        end
        if (method == 2)
            distance_update = dist2(Cluster, InputCorrMat, distance, row, col, 63);
        end
        Cluster(:,row) = Cluster(:,row) + Cluster(:,col);
        Cluster(:,col) = decoy;
        distance = distance_update;
        MinDist = min(min(distance));
%         MinDist = 1; %----------
%         Cluster = distance; %---------
    end
end

function distance_update = dist1(Cluster, CorrMat, distance, row, col, dim)
    temp_dist = distance;
    temp_dist(:,col) = temp_dist(:,col) + 100;
    temp_dist(col,:) = temp_dist(col,:) + 100;
    
    A = length(find(Cluster(:,row) ~= 0));
    B = length(find(Cluster(:,col) ~= 0));
    % updating distance(:,row)
    for i = 1:row-1
        dist_row_i = distance(i,row);
        dist_col_i = distance(i,col);
        du = (A * dist_row_i + B * dist_col_i) / (A + B);
        temp_dist(i,row) = du;
    end
    
    % updating distance(row,:)
    if (row ~= dim)
        for i = row+1:col-1
            dist_row_i = distance(row,i);
            dist_col_i = distance(i,col);
            du = (A * dist_row_i + B * dist_col_i) / (A + B);
            temp_dist(row,i) = du;
        end
        for i = col+1:dim
            dist_row_i = distance(row,i);
            dist_col_i = distance(col,i);
            du = (A * dist_row_i + B * dist_col_i) / (A + B);
            temp_dist(row,i) = du;            
        end
    end
    distance_update = temp_dist;
end

function distance_update = dist2(Cluster, CorrMat, distance, row, col, dim)
    temp_dist = distance;
    temp_dist(:,col) = temp_dist(:,col) + 100;
    temp_dist(col,:) = temp_dist(col,:) + 100;
    % updating distance(:,row)
    for i = 1:row-1
        dist_row_i = distance(i,row);
        dist_col_i = distance(i,col);
        du = (dist_row_i + dist_col_i) / 2;
        temp_dist(i,row) = du;
    end
    
    % updating distance(row,:)
    if (row ~= dim)
        for i = row+1:col-1
            dist_row_i = distance(row,i);
            dist_col_i = distance(i,col);
            du = (dist_row_i + dist_col_i) / 2;
            temp_dist(row,i) = du;
        end
        for i = col+1:dim
            dist_row_i = distance(row,i);
            dist_col_i = distance(col,i);
            du = (dist_row_i + dist_col_i) / 2;
            temp_dist(row,i) = du;            
        end
    end
    distance_update = temp_dist;
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


