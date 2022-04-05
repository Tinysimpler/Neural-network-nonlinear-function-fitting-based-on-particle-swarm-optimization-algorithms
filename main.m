% 清空环境变量
clc
clear
% 生成训练数据，数量100
x1 = linspace(1,100,100);
x2 = linspace(1,100,100);
X = [x1;x2];
Y = zeros( 100,100);
for row = 1 : 1 : 100    
    for col = 1 : 1 : 100        
        Y( row,col) = sin(10*x1(row))-x2(col).^3+(x1(row).^2) .* x2(col);
    end
end

% 生成检验数据，数量100
xt1 = linspace(1,100,100);
xt2 = linspace(1,100,100);
XT = [xt1;xt2];
Y2 = zeros( 100,100);
for row = 1 : 1 : 100
    for col = 1 : 1 : 100
        Y2( row,col) = sin(10*xt1(row))-xt2(col).^3+(xt1(row).^2) .* xt2(col); 
    end
end
% 对样本输入X输出Y作归一化处理，数据范围限制在[-1,1]，归一化数据结构保存在ps
[Data_target,ps_output] = mapminmax(Y,-1,1);
[Data_input,ps_input] = mapminmax(X,-1,1);
% 对检验数据做归一化处理
Data_test = mapminmax('apply',XT,ps_input);

%节点个数
inputnum=size(Data_input,1);       % 输入层神经元个数 
outputnum=size(Data_target,1);     % 输出层神经元个数
hiddennum=10;
% 创建网络；
net1 = newff(Data_input,Data_target,hiddennum);
net2 = newff(Data_input,Data_target,hiddennum);
net3 = newff(Data_input,Data_target,hiddennum);
%节点总数 2*5 + 5 + 5 + 1 = 21 
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

%% 粒子群算法求权值和阈值
%粒子群算法参数设置
N = 20;
c1 = 2;
c2 = 2;
w = 0.6;
M = 100;
D = numsum;
x = zeros(1,D);
% 调用粒子群算法函数
[xm1,fv1,Pbest1] = NNPSO(x,hiddennum,net1,Data_input,Data_target,N,w,c1,c2,M,D);
[xm2,fv2,Pbest2] = NNSAPSO(x,hiddennum,net2,Data_input,Data_target,N,w,c1,c2,M,D);
[xm3,fv3,Pbest3] = NNCSAPSO(x,hiddennum,net3,Data_input,Data_target,N,w,c1,c2,M,D);
% [xm3,fv3,Pbest3] = NNCSAPSO2(x,hiddennum,net3,Data_input,Data_target,N,w,c1,c2,M,D);

%% 把最优初始阀值权值赋予网络预测
% 用粒子群算法优化的BP网络进行值预测
w1_1=xm1(1:inputnum*hiddennum);
B1_1=xm1(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2_1=xm1(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2_1=xm1(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net1.iw{1,1}=reshape(w1_1,hiddennum,inputnum);
net1.lw{2,1}=reshape(w2_1,outputnum,hiddennum);
net1.b{1}=reshape(B1_1,hiddennum,1);
net1.b{2}=reshape(B2_1,outputnum,1);

% % 用模拟退火粒子群算法优化的BP网络进行值预测
w1_2=xm2(1:inputnum*hiddennum);
B1_2=xm2(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2_2=xm2(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2_2=xm2(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net2.iw{1,1}=reshape(w1_2,hiddennum,inputnum);
net2.lw{2,1}=reshape(w2_2,outputnum,hiddennum);
net2.b{1}=reshape(B1_2,hiddennum,1);
net2.b{2}=reshape(B2_2,outputnum,1);

% 用混沌模拟退火粒子群算法优化的BP网络进行值预测
w1_3=xm3(1:inputnum*hiddennum);
B1_3=xm3(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2_3=xm3(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2_3=xm3(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net3.iw{1,1}=reshape(w1_3,hiddennum,inputnum);
net3.lw{2,1}=reshape(w2_3,outputnum,hiddennum);
net3.b{1}=reshape(B1_3,hiddennum,1);
net3.b{2}=reshape(B2_3,outputnum,1);

%% BP网络训练
%粒子群网络进化参数
net1.trainParam.epochs=100;
net1.trainParam.lr = 0.1;
net1.trainParam.goal=1e-3; % 训练目标误差
% 
%模拟退火粒子群网络进化参数
net2.trainParam.epochs=100;
net2.trainParam.lr=0.1;
net2.trainParam.goal=1e-6;

%混沌模拟退火粒子群网络进化参数
net3.trainParam.epochs=100;
net3.trainParam.lr=0.1;
net3.trainParam.goal=1e-3;

% 训练网络
net1 = train(net1,Data_input,Data_target); % 粒子群
net2 = train(net2,Data_input,Data_target); % 模拟退火粒子群
net3 = train(net3,Data_input,Data_target); % 混沌模拟退火粒子群

%% 仿真测试
test_sim1 = sim(net1,Data_test); % 粒子群
test_sim2 = sim(net2,Data_test); % 模拟退火粒子群
test_sim3 = sim(net3,Data_test); % 混沌模拟退火粒子群

% 输出数据反归一化，Test_sim为测试数据通过神经网络的预测输出值
Test_sim1 = mapminmax('reverse',test_sim1,ps_output); % 粒子群
Test_sim2 = mapminmax('reverse',test_sim2,ps_output); % 模拟退火粒子群
Test_sim3 = mapminmax('reverse',test_sim3,ps_output); % 混沌模拟退火粒子群

%% 算法结果分析 
figure(1)
t = 1:M;
plot(t,Pbest1,'b',t,Pbest2,'g',t,Pbest3,'r');
title('算法收敛过程');
xlabel('进化代数');
ylabel('最小均方误差值（MSE值）');
legend('基本粒子群算法','模拟退火粒子群算法','混沌模拟退火粒子群算法');

figure(2)
mesh(xt1,xt2,Y2);
title('函数实际图形');

%% 拟合图形对比输出

%基础粒子群对比输出
figure(3)
subplot(1,2,1)
mesh(xt1,xt2,Y2);
title('函数实际图形');
xlabel('X1取值');ylabel('X2取值');zlabel('非线性函数输出值');

subplot(1,2,2)
mesh(xt1,xt2,Test_sim1);
title('基础粒子群算法拟合图形');
xlabel('X1取值');ylabel('X2取值');zlabel('非线性函数输出值');

% % 模拟退火粒子群对比输出
figure(4)
subplot(1,2,1)
mesh(xt1,xt2,Y2);
title('函数实际图形');
xlabel('X1取值');ylabel('X2取值');zlabel('非线性函数输出值');

subplot(1,2,2)
mesh(xt1,xt2,Test_sim2);
title('模拟退火粒子群算法拟合图形');
xlabel('X1取值');ylabel('X2取值');zlabel('非线性函数输出值');

%混沌模拟退火粒子群拟合图形单独输出
figure(5)
subplot(1,2,1)
mesh(xt1,xt2,Y2);
title('函数实际图形');
xlabel('X1取值');ylabel('X2取值');zlabel('非线性函数输出值');

subplot(1,2,2)
mesh(xt1,xt2,Test_sim3);
title('混沌模拟退火粒子群算法拟合图形');
xlabel('X轴');ylabel('Y轴');zlabel('Z轴');