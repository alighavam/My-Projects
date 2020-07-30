%% 
clear x y test ytest mdl
clear all
close all
%% --------------target part-----------------------
clc
clear x y test ytest mdl
data1 = EpochData05Target;
data2 = EpochData05NonTarget;
tic
x = data1(:,:,2)';
x = [x;x;x];
y = ones(3*size(data1,2),1);
x_temp = data2(:,:,2)';
x = [x ; x_temp];
y = [y ; -1 * ones(size(data2,2),1)];
mdl = fitcsvm(x,y);
toc

%% Finding Indexes
clc;

test = TestEpochData5(:,:,1)';

pred = predict(mdl,test);

length(find(pred == 1))


%% Finding Word
data5 = load('SubjectData5.mat');
index = find(pred == 1); % Target Stimuli Number
data = data5.test;
words_temp = data(10,:);
StimuliOnset1 = find_stimuli(data);
target_stimuli = StimuliOnset1(index);
words = words_temp(target_stimuli)';

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


