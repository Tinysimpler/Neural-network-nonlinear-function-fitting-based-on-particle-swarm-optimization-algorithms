function error = nnfitness(x,hiddennum,net,Data_input,Data_target)
%该函数用来计算适应度值
%x          input     待优化参数（权值、阈值）集合
%inputnum   input     输入层节点数
%outputnum  input     隐含层节点数
%net        input     网络
%Data_input input     训练输入数据
%Data_targetinput     训练输出数据
%error      output    个体适应度值

inputnum = size(Data_input,1);
outputnum = size(Data_target,1);

%提取
w1=x(1:inputnum*hiddennum);%取到输入层与隐含层连接的权值
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);%隐含层神经元阈值
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);%取到隐含层与输出层连接的权值
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);%输出层神经元阈值 

%网络进化参数
% net.trainParam.epochs=20;%迭代次数
% net.trainParam.lr=0.1;%学习率
% net.trainParam.goal=0.001;%最小目标值误差
% net.trainParam.show=100;
% net.trainParam.showWindow=0;

%网络权值赋值
net.iw{1,1}=reshape(w1,hiddennum,inputnum);%将w1由1行inputnum*hiddennum列转为hiddennum行inputnum列的二维矩阵
net.lw{2,1}=reshape(w2,outputnum,hiddennum);%更改矩阵的保存格式
net.b{1}=reshape(B1,hiddennum,1);%1行hiddennum列，为隐含层的神经元阈值
net.b{2}=reshape(B2,outputnum,1);

% 训练网络
% net = train(net,Data_input,Data_target);

%网络仿真预测
an=sim(net,Data_input);
% error=sum((abs(test_sim-Data_target)).^2);%粒子对应的适应度值，即训练的最小均方误差值和
% 求平均值,即化为1行100列
test_sim = zeros(1,outputnum);
for i = 1:outputnum
    test_sim (i)= sum(an(:,i))/outputnum; 
end
error1=sum((abs(test_sim-Data_target)).^2);%粒子对应的适应度值
% 求平均值，即化为一个具体的适应度值
error = sum(error1)/outputnum;



