%------------------------------------------------------------ Section 1
%Introduction to MATLAB
% Q1.1 Plots
clear all; close all; clc;

%chart 1
t1 = -3:0.01:3;
y2 = 2*sinc(4*t1);

figure
subplot(3,2,[1 3])
for x=-2:2 %blue vertical lines
    plot([x x],[-2 2],'b','LineWidth',2)
    hold on
end
for x=-3:3 %blue lines --> y = 2x 
    p1 = plot([x x+1],[-2 2],'b','LineWidth',2);
    hold on
end
hold on
p2 = plot(t1,y2,'--g','LineWidth',2);
ylim([-3.5 2.5])
xlim([-3 3])
grid minor
xticks([-3:1:3])
yticks([-3:1:3])
ylabel('$y(V)$','Interpreter','LaTex')
title('$y_i(t)$ for  $i$ in $\{1,2,3\}$ on the same figure','Interpreter','LaTex')
for x=-3:3 %blue lines --> y = 2x 
    hold on
    y =(-1)^x;
    y1 = [y y y y y y y y y y y y y y y y y y];
    x2 = linspace(x,x+1,18);
    p3 = plot(x2,y1,'.r','LineWidth',1);
end
legend([p1 p2 p3],'y1(t)','y2(t)','y3(t)')

%chart 2
subplot(3,2,2)
n=-10:1:20;
h = ((1/2).^n).*(n>=0) + ((2/3).^n).*(n>=-1);
stem(n,h,'b','LineWidth',2)
grid on
grid minor
xlabel('$n$','Interpreter','LaTex');
ylabel('$h[n]$','Interpreter','LaTex');
title('$h[n] = (\frac{1}{2})^nu[n]+(\frac{2}{3})^nu[n+1] $','Interpreter','LaTex')

%chart 3
syms xs
subplot(3,2,5)
x = [-1:0.1:3];
y4 = x.^3 - 4*x.^2 + 4*x + 1;
plot(x,y4)
xlim([-1 3])
ylim([-8 4])
title('$y4(x)$','Interpreter','LaTex')
eq = xs^3 - 4*xs^2 + 4*xs + 1 == 0;
root = vpasolve(eq,xs);
root = root(1);
t1 = text(root,0,'$\leftarrow x^4-4x^2+4x+1=0$','Interpreter','LaTex')
set(t1,'FontSize',15)
%chart 4
subplot(3,2,[4 6],'Position',[0.625 0.11 0.225 0.525])
theta = [0:0.01:2*pi];
x = 4 * cos(theta) + 8;
y = 4 * sin(theta) + 6;
plot(x,y,'LineWidth',1.5)
grid on
hold on
ellipse(6,6,2,1);
ellipse(10,6,2,1);
ellipse(8,4,1,2);
ellipse(8,8,1,2);
xlim([4.01 12]);
ylim([1.5 10.5]);

%% Q1.2 Matrix Calculations
clear all; close all; clc;
% part 1
matrix = ones(50,50); % 50x50 ones matrix
%---------------------------- replacing under main diagonal with [-5 4] random numbers
randMatrix = randi([-6 3],50);
randMatrix = tril(randMatrix,-1); % under triangular random matrix
matrix = matrix + randMatrix;
%---------------------------- 
%---------------------------- magic matrix replacement
magicMatrix = magic(50);
maxRowData = max(magicMatrix,[],2); % each row max
magicMatrix = (4 * magicMatrix) ./ maxRowData; % normalized to 4
magicMatrix = magicMatrix - 1; % we want to add it to ones matrix
magicMatrix = triu(magicMatrix,1); % upper triangular
matrix = magicMatrix + matrix;
%---------------------------
%--------------------------- anti diagonal to 0
matrixFlip = flip(matrix); % changin' main diagonal with anti diagonal
matrixFlip = triu(matrixFlip);
matrixFlip = tril(matrixFlip); 
matrixFlip = flip(matrixFlip); % changin' anti diagonal with main diagonal
matrix = matrix - matrixFlip;
%----------------------------
%---------------------------- main digonal to 2
matrix = matrix + eye(50);
%----------------------------
%---------------------------- replacing two random NaN
elements = randi([1 50],2,2)
matrix(elements(1),elements(3)) = NaN;
matrix(elements(2),elements(4)) = NaN;
%----------------------------
My_Struct.('matrix_including_NaN') = matrix;

% part 2
colmean = mean(matrix); % there is NaN in output
My_Struct.('col_mean_with_NaN') = colmean;
rowmean = mean(matrix,2); % there is NaN in output
My_Struct.('row_mean_with_NaN') = rowmean;
colmax = max(matrix); % it ignores the NaN
My_Struct.('col_max_with_NaN') =  colmax;
rowmax = max(matrix,[],2); % it ignores the NaN
My_Struct.('row_max_with_NaN') = rowmax;
[row, col] = find(isnan(matrix));
% replacing NaN with average of 8 neraby elements
a = row(1)-1:row(1)+1;
b = col(1)-1:col(1)+1;
avg = avger(matrix,a,b); % avger function is in functions section
matrix(row(1),col(1)) = avg;
a = row(2)-1:row(2)+1;
b = col(2)-1:col(2)+1;
avg = avger(matrix,a,b);
matrix(row(2),col(2)) = avg;
My_Struct.('matrix_without_NaN') = matrix;

%part 3
[row col] = find(matrix == 1);
[GC,GR] = groupcounts(col); % matlab 2019 function!!!
indices = find(GC > 4); % GC is the number of occurance of 1 for each col
mustRemove = GR(indices) % repeted cols with more than 4 number of occurance of 1
matrix(:,mustRemove) = [];
My_Struct.('matrix_after_removing_cols_with_more_than_4_ones') = matrix;

My_Cell = struct2cell(My_Struct); % copying struct into cell :D

save My_Struct
save My_Cell

%% Q1.3 Conway's Game of Life
% The function Conways_Game_of_Life is available in functions section
clear all; close all; clc;
col = 200;
row = 200;
City = randi([0 1],row,col); % random city
%------------------------------------- 10 cell row!!, in case u want to check
%if alogorithm is right or not use this city which is a famous test called
%10 cell row:

% City = zeros(100);    
% City(46:55,50) = 1;

%-------------------------------------
Conways_Game_of_Life(City,row,col);

%% --------------------------------------------------------------Section 2
% System Analysis Using Z-Transform
clear all; close all; clc;
%----------------------------------------------------Q2.1
% Part 1
syms n z
sympref('HeavisideAtOrigin',1);
h1(n) = ((1/2)^n)*(heaviside(n)-heaviside(n-30));
h2(n) = (2^n)*(heaviside(n)-heaviside(n-40));
h3(n) = ((1/2)^n)*(heaviside(n));
h4(n) = ((2)^n)*(heaviside(n));
n1 = [-5:1:45];
%----------------h1 plot
subplot(2,2,1)
stem(n1,h1(n1),'b','LineWidth',1.5)
grid on
grid minor
title('h1[n]','Interpreter','LaTex')
xlabel('n','Interpreter','LaTex')
%----------------h2 plot
subplot(2,2,2)
stem(n1,h2(n1),'b','LineWidth',1.5)
grid on
grid minor
title('h2[n]','Interpreter','LaTex')
xlabel('n','Interpreter','LaTex')
%---------------h3 plot
subplot(2,2,3)
stem(n1,h3(n1),'b','LineWidth',1.5)
grid on
grid minor
title('h3[n]','Interpreter','LaTex')
xlabel('n','Interpreter','LaTex')
%----------------h4 plot
subplot(2,2,4)
stem(n1,h4(n1),'b','LineWidth',1.5)
grid on
grid minor
title('h4[n]','Interpreter','LaTex')
xlabel('n','Interpreter','LaTex')

%----------------z-trans of functions
%H1
H1 = ztrans(h1);
disp("H1 is:");
pretty(simplify(H1));
%H2
H2 = ztrans(h2);
disp("H2 is:");
pretty(simplify(H2));
%H3
H3 = ztrans(h3);
disp("H3 is:");
pretty(simplify(H3));
%H4
H4 = ztrans(h4);
disp("H4 is:");
pretty(simplify(H4));

%-------------- zero pole plots
%H1
figure
subplot(2,2,1)
[A1 A2] = numden(H1); 
Z = tf(sym2poly(A1),sym2poly(A2),-1);
zero1 = zero(Z);
pole1 = pole(Z);
zplane(zero1,pole1)
title('zero-pole map h1(z)')
%H2
subplot(2,2,2)
[A1 A2] = numden(H2); 
Z = tf(sym2poly(A1),sym2poly(A2),-1);
zero2 = zero(Z);
pole2 = pole(Z);
zplane(zero2,pole2)
title('zero-pole map h2(z)')
%H3
subplot(2,2,3)
[A1 A2] = numden(H3); 
Z = tf(sym2poly(A1),sym2poly(A2),-1);
zero3 = zero(Z);
pole3 = pole(Z);
zplane(zero3,pole3)
title('zero-pole map h3(z)')
% %H2
subplot(2,2,4)
[A1 A2] = numden(H4); 
Z = tf(sym2poly(A1),sym2poly(A2),-1);
zero4 = zero(Z);
pole4 = pole(Z);
zplane(zero4,pole4)
title('zero-pole map h4(z)')

%% Part 2
clear all; close all; clc;
syms n z
x1 = tf([1],[1 -0.5],-1,'variable','z^-1');
x1_sym = 1/(1 - 0.5*z^-1);
x2 = tf([1],[1 -0.9],-1,'variable','z^-1');
x2_sym = 1/(1 - 0.9*z^-1);
x3 = tf([1],[1 -1.1],-1,'variable','z^-1');
x3_sym = 1/(1 - 1.1*z^-1);
x4 = tf([1],[1 -5],-1,'variable','z^-1');
x4_sym = 1/(1 - 5*z^-1);
x5 = tf([1],[1 -1],-1,'variable','z^-1');
x5_sym = 1/(1 - z^-1);
x6 = tf([1],[1 -2 1],-1,'variable','z^-1');
x6_sym = 1/(1 - z^-1)^2;
%-------------zero-pole map
figure
%x1
subplot(2,3,1)
zero1 = zero(x1);
pole1 = pole(x1);
zplane(zero1,pole1)
title('zero-pole map x1(z)')
%x2
subplot(2,3,2)
zero2 = zero(x2);
pole2 = pole(x2);
zplane(zero2,pole2)
title('zero-pole map x2(z)')
%x3
subplot(2,3,3)
zero3 = zero(x3);
pole3 = pole(x3);
zplane(zero3,pole3)
title('zero-pole map x3(z)')
%x4
subplot(2,3,4)
zero4 = zero(x4);
pole4 = pole(x4);
zplane(zero4,pole4)
title('zero-pole map x4(z)')
%x5
subplot(2,3,5)
zero5 = zero(x5);
pole5 = pole(x5);
zplane(zero5,pole5)
title('zero-pole map x5(z)')
%x6
subplot(2,3,6)
zero6 = zero(x6);
pole6 = pole(x6);
zplane(zero6,pole6)
title('zero-pole map x6(z)')

%------------- impulse response
figure
n1 = 0:30;
%x1
subplot(2,3,1)
x(n) = iztrans(x1_sym);
stem(n1,x(n1),'LineWidth',1.5);
xlabel('n','Interpreter','LaTex')
title('x1[n]')
%x2
subplot(2,3,2)
x(n) = iztrans(x2_sym);
stem(n1,x(n1),'LineWidth',1.5);
xlabel('n','Interpreter','LaTex')
title('x2[n]')
%x3
subplot(2,3,3)
x(n) = iztrans(x3_sym);
stem(n1,x(n1),'LineWidth',1.5);
xlabel('n','Interpreter','LaTex')
title('x3[n]')
%x4
subplot(2,3,4)
x(n) = iztrans(x4_sym);
stem(n1,x(n1),'LineWidth',1.5);
xlabel('n','Interpreter','LaTex')
title('x4[n]')
%x5
subplot(2,3,5)
x(n) = iztrans(x5_sym);
stem(n1,x(n1),'LineWidth',1.5);
xlabel('n','Interpreter','LaTex')
title('x5[n]')
ylim([0 1.5])
%x6
subplot(2,3,6)
x(n) = iztrans(x6_sym);
stem(n1,x(n1),'LineWidth',1.5);
xlabel('n','Interpreter','LaTex')
title('x6[n]')

%% Part 3
clear all; close all; clc;
syms n z
x7 = tf([1],[1 -sqrt(2) 1],-1,'variable','z^-1');
x7_sym = 1/(1 - sqrt(2)*z^-1 + z^-2);
x8 = tf([1],[1 -1 2],-1,'variable','z^-1');
x8_sym = 1/(1 - z^-1 + 2*z^-2);
x9 = tf([1],[2 -1 1],-1,'variable','z^-1');
x9_sym = 1/(2 - z^-1 + z^-2);

%-------------- zero pole map
figure
%x7
subplot(2,3,1)
zero7 = zero(x7);
pole7 = pole(x7);
zplane(zero7,pole7)
title('zero-pole map x7(z)')
%x8
subplot(2,3,2)
zero8 = zero(x8);
pole8 = pole(x8);
zplane(zero8,pole8)
title('zero-pole map x8(z)')
%x9
subplot(2,3,3)
zero9 = zero(x9);
pole9 = pole(x9);
zplane(zero9,pole9)
title('zero-pole map x9(z)')

%-------------- impulse response
n1 = 0:30;
%x7
subplot(2,3,4)
x(n) = iztrans(x7_sym);
stem(n1,x(n1),'LineWidth',1.5);
xlabel('n','Interpreter','LaTex')
title('x7[n]')
%x2
subplot(2,3,5)
x(n) = iztrans(x8_sym);
stem(n1,x(n1),'LineWidth',1.5);
xlabel('n','Interpreter','LaTex')
title('x8[n]')
%x3
subplot(2,3,6)
x(n) = iztrans(x9_sym);
stem(n1,x(n1),'LineWidth',1.5);
xlabel('n','Interpreter','LaTex')
title('x9[n]')

%% ---------------------------------Q2.2 propeties of z transform
% Part 1
clear all; close all; clc;
syms n z
sympref('HeavisideAtOrigin',1);
x(n) = cos(n * pi/6) *  heaviside(n);
zt = simplify(ztrans(x));% without simplify() its not a sym expression
[A1 A2] = numden(zt);
Z = tf(sym2poly(A1),sym2poly(A2),-1)
zeros = zero(Z);
poles = pole(Z);
zplane(zeros,poles)
title('Zero-Pole Map of X')

%% Part 2
clear all; close all; clc;
syms n z
sympref('HeavisideAtOrigin',1);
x(n) = cos(n * pi/6) *  heaviside(n);
zt(z) = simplify(ztrans(x));% without simplify() its not a sym expression
x2(z) = zt(2*z)
% zero pole map
[A1 A2] = numden(x2)
Z = tf(sym2poly(A1),sym2poly(A2),-1)
zeros = zero(Z);
poles = pole(Z);
zplane(zeros,poles)
title('Zero-Pole Map of X1')

%x1[n] plot
figure
subplot(1,2,1)
n1 = 0:40;
x2(n) = iztrans(x2(z));
stem(n1,x2(n1),'LineWidth',1.5)
title('x1[n]')
xlabel('n')

%x[n] plot
subplot(1,2,2)
stem(n1,x(n1),'LineWidth',1.5)
title('x[n]')
xlabel('n')

%% Part 3
clear all; close all; clc;
syms n z
sympref('HeavisideAtOrigin',1);
x(n) = cos(n * pi/6) *  heaviside(n);
zt(z) = simplify(ztrans(x));% without simplify() its not a sym expression
x2(z) = zt(z^3)
% zero pole map
[A1 A2] = numden(x2)
Z = tf(sym2poly(A1),sym2poly(A2),-1)
zeros = zero(Z);
poles = pole(Z);
zplane(zeros,poles)
title('Zero-Pole Map of X2')

%x2[n] plot
figure
subplot(1,2,1)
n1 = 0:40;
x2(n) = iztrans(x2(z));
stem(n1,x2(n1),'LineWidth',1.5)
title('x2[n]')
xlabel('n')

%x[n] plot
subplot(1,2,2)
stem(n1,x(n1),'LineWidth',1.5)
title('x[n]')
xlabel('n')

%% ---------------------------------Q2.3 The Inverse Z-Transform
% Part 1
clear all; clc; close all;
syms z n
h1 = tf([1 -1],[1 -1 0.5],1,'variable','z^-1');
h2 = tf([0 1],[2 -sqrt(3) 0.5],-1,'variable','z^-1');
%zero pole plot
subplot(1,2,1)
zero1 = zero(h1);
pole1 = pole(h1);
zplane(zero1,pole1)
title('Zero-Pole Map of H1')
subplot(1,2,2)
zero2 = zero(h2);
pole2 = pole(h2);
zplane(zero2,pole2)
title('Zero-Pole Map of H2')

%% Part 2
clear all; close all; clc;
syms n z
[r1 p1 k1] = residuez([1 -1],[1 -1 0.5])
[r2 p2 k2] = residuez([0 1],[2 -sqrt(3) 0.5])

%% Part 3
clear all; close all; clc;
syms z n
sympref('HeavisideAtOrigin',1);
h1_part2(n) = (0.5+0.5i)^(n+1) * heaviside(n) + (0.5-0.5i)^(n+1) * heaviside(n);
h2_part2(n) = (-i) * (sqrt(3)/4 + 0.25i)^n * heaviside(n) + i * (sqrt(3)/4 - 0.25i)^n * heaviside(n);

h1(z) = (1 - z^-1)/(1 - z^-1 + 0.5*z^-2);
h2(z) = (z^-1)/(2 - sqrt(3)*z^-1 + 0.5*z^-2);
h1_inv(n) = iztrans(h1);
h2_inv(n) = iztrans(h2);

% plot
subplot(2,2,1)
n1 = 0:40;
stem(n1,h1_part2(n1),'LineWidth',1.5)
xlabel('n')
title('manually calculated h1[n]')
subplot(2,2,2)
stem(n1,h1_inv(n1),'LineWidth',1.5)
xlabel('n')
title('h1[n] using iztrans')
subplot(2,2,3)
stem(n1,h2_part2(n1),'LineWidth',1.5)
xlabel('n')
title('manually calculated h2[n]')
subplot(2,2,4)
stem(n1,h2_inv(n1),'LineWidth',1.5)
xlabel('n')
title('h2[n] using iztrans')

%% ----------------------------Q2.4 Differential Equation
clear all; close all; clc;
% Part 1
sympref('HeavisideAtOrigin',1);
syms n z
% hz(z) = (2 - z^-1)/(1 - 0.7*z^-1 + 0.49*z^-2);
% h_iz(n) = iztrans(hz);
[r p k] = residuez([2 -1],[1 -0.7 0.49])
h(n) = (1 + 0.2474i) * (0.35 + 0.6062i)^n * heaviside(n) + (1 - 0.2474i) * (0.35 - 0.6062i)^n * heaviside(n);

%plot
n1 = 0:40;
figure
stem(n1,h(n1),'LineWidth',1.5)
title('manually calculated h[n]')
xlabel('n')

%% Part 2
% finding poles
clear all; close all; clc;

Z = tf([2 -1],[1 -0.7 0.49],-1,'variable','z^-1')
poles = pole(Z)

%solving equation:
a = [1 1 ; 0.35+0.6062i 0.35-0.6062i]^-1 * [2 ; 0.4]    

%finding h[n] and plotting alongside with manually calculated h[n]
syms n z
sympref('HeavisideAtOrigin',1);
h_zt(z) = (1 + 0.2474i) / (1 - (0.35 + 0.6062i) * z^-1) + (1 - 0.2474i) / (1 - (0.35 - 0.6062i) * z^-1);
h_n(n) = iztrans(h_zt);
subplot(1,2,1)
n1 = 0:40;
stem(n1,h_n(n1),'LineWidth',1.5)
title('h[n] found in part 2')
xlabel('n')

subplot(1,2,2)
h_part1(n) = (1 + 0.2474i) * (0.35 + 0.6062i)^n * heaviside(n) + (1 - 0.2474i) * (0.35 - 0.6062i)^n * heaviside(n);
stem(n1,h_part1(n1),'LineWidth',1.5)
title('h[n] from part 1')
xlabel('n')

%% Part 3
clear all; clc; close all;
syms n z
x = [1 zeros(1,40)];
b = [2 -1];
a = [1 -0.7 0.49];
h = filter(b,a,x);
n1 = 0:40
stem(n1,h,'LineWidth',1.5)
title('h[n] using filter() function')
xlabel('n')

% difference between filter and iztrans()
figure
h_zt(z) = (2 - z^-1)/(1 - 0.7*z^-1 + 0.49*z^-2);
x = [zeros(1,5) 1 zeros(1,40)];
h = filter(b,a,x);
n1 = -5:40;
subplot(1,2,1)
stem(n1,h,'LineWidth',1.5)
title('h[n] using filter() function')
grid on
grid minor
xlabel('n')

subplot(1,2,2)
iz(n) = iztrans(h_zt);
stem(n1,iz(n1),'LineWidth',1.5)
title('h[n] using iztrans() function')
xlabel('n')
grid on
grid minor

%% ------------Part 4(optinal*)
% Part 2*
clear all; close all; clc;
syms z n
Z = tf([2 -1],[1 -1 -0.5 0.5],-1,'variable','z^-1')
h_zt = (2 - z^-1) / (1 - z^-1 - 0.5*z^-2 + 0.5*z^-3);
poles = pole(Z)

%solving equation:
a = [1 1 1; -0.7071 1 0.7071;(-0.7071)^2 1 (0.7071)^2]^-1 * [2 ; 1 ; 2]    

%plot
sympref('HeavisideAtOrigin',1);
h(n) = (0.7071 * (-0.7071)^n + 2 - 0.7071 * (0.7071)^n) * heaviside(n);
n1 = 0:40;
subplot(1,2,1)
stem(n1,h(n1),'LineWidth',1.5)
title('h(n) part 2')
xlabel('n')

subplot(1,2,2)
h_niz(n) = iztrans(h_zt);
stem(n1,h_niz(n1),'LineWidth',1.5)
title('h(n) from iztrans()')
xlabel('n')

%% Part 3*
clear all; clc; close all;
syms n z
x = [1 zeros(1,40)];
b = [2 -1];
a = [1 -1 -0.5 0.5];
h = filter(b,a,x);
n1 = 0:40
stem(n1,h,'LineWidth',1.5)
title('h[n] using filter() function')
xlabel('n')










%%




%% functions
function ellipse(xc,yc,xr,yr)
    theta = 0:0.01:2*pi;
    x = xc + xr * cos(theta);
    y = yc + yr * sin(theta);
    plot(x,y,'LineWidth',1.5)
    hold on
end

function avg = avger(matrix,a,b) % finds the mean of nearby elements
    % preventing bound limit error
    avgMat = matrix;
    appendMat = zeros(1,50);
    avgMat = [appendMat;avgMat;appendMat];
    appendMat = zeros(52,1);
    avgMat = [appendMat avgMat appendMat];
    
    avgMat = avgMat(a+1,b+1);
    avgMat(2,2) = 0;
    avg = sum(avgMat,'all') / 8;
end

function Conways_Game_of_Life(City,row,col)
    h = figure;
    % first I add zeros around City to prevent bound limits exceeding
    nextGeneration = zeros(row,col);
    while(ishandle(h))
        appendCity = zeros(1,col);
        appCity = [appendCity ; City ; appendCity];
        appendCity = zeros(row+2,1);
        appCity = [appendCity appCity appendCity];
        for i = 2:col+1
            for j = 2:row+1
                condition = appCity(i,j); % 1 = alive &&& 0 = dead
                a = [i-1:1:i+1];
                b = [j-1:1:j+1];
                decider = appCity(a,b);
                decider(2,2) = 0;
                aroundLiving = sum(decider,'all');
                if (condition == 1) % if cell is alive
                    if ((aroundLiving < 2) || (aroundLiving > 3))
                        condition = 0;
                    end
                else % if cell is dead
                    if (aroundLiving == 3)
                        condition = 1;
                    end
                end
                nextGeneration(i-1,j-1) = condition;   
            end
        end
        imagesc(nextGeneration)
        colormap(gray);
        pause(0.01)
        City = nextGeneration;
    end
end




