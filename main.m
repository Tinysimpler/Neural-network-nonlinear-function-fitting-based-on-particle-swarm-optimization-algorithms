% ��ջ�������
clc
clear
% ����ѵ�����ݣ�����100
x1 = linspace(1,100,100);
x2 = linspace(1,100,100);
X = [x1;x2];
Y = zeros( 100,100);
for row = 1 : 1 : 100    
    for col = 1 : 1 : 100        
        Y( row,col) = sin(10*x1(row))-x2(col).^3+(x1(row).^2) .* x2(col);
    end
end

% ���ɼ������ݣ�����100
xt1 = linspace(1,100,100);
xt2 = linspace(1,100,100);
XT = [xt1;xt2];
Y2 = zeros( 100,100);
for row = 1 : 1 : 100
    for col = 1 : 1 : 100
        Y2( row,col) = sin(10*xt1(row))-xt2(col).^3+(xt1(row).^2) .* xt2(col); 
    end
end
% ����������X���Y����һ���������ݷ�Χ������[-1,1]����һ�����ݽṹ������ps
[Data_target,ps_output] = mapminmax(Y,-1,1);
[Data_input,ps_input] = mapminmax(X,-1,1);
% �Լ�����������һ������
Data_test = mapminmax('apply',XT,ps_input);

%�ڵ����
inputnum=size(Data_input,1);       % �������Ԫ���� 
outputnum=size(Data_target,1);     % �������Ԫ����
hiddennum=10;
% �������磻
net1 = newff(Data_input,Data_target,hiddennum);
net2 = newff(Data_input,Data_target,hiddennum);
net3 = newff(Data_input,Data_target,hiddennum);
%�ڵ����� 2*5 + 5 + 5 + 1 = 21 
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

%% ����Ⱥ�㷨��Ȩֵ����ֵ
%����Ⱥ�㷨��������
N = 20;
c1 = 2;
c2 = 2;
w = 0.6;
M = 100;
D = numsum;
x = zeros(1,D);
% ��������Ⱥ�㷨����
[xm1,fv1,Pbest1] = NNPSO(x,hiddennum,net1,Data_input,Data_target,N,w,c1,c2,M,D);
[xm2,fv2,Pbest2] = NNSAPSO(x,hiddennum,net2,Data_input,Data_target,N,w,c1,c2,M,D);
[xm3,fv3,Pbest3] = NNCSAPSO(x,hiddennum,net3,Data_input,Data_target,N,w,c1,c2,M,D);
% [xm3,fv3,Pbest3] = NNCSAPSO2(x,hiddennum,net3,Data_input,Data_target,N,w,c1,c2,M,D);

%% �����ų�ʼ��ֵȨֵ��������Ԥ��
% ������Ⱥ�㷨�Ż���BP�������ֵԤ��
w1_1=xm1(1:inputnum*hiddennum);
B1_1=xm1(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2_1=xm1(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2_1=xm1(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net1.iw{1,1}=reshape(w1_1,hiddennum,inputnum);
net1.lw{2,1}=reshape(w2_1,outputnum,hiddennum);
net1.b{1}=reshape(B1_1,hiddennum,1);
net1.b{2}=reshape(B2_1,outputnum,1);

% % ��ģ���˻�����Ⱥ�㷨�Ż���BP�������ֵԤ��
w1_2=xm2(1:inputnum*hiddennum);
B1_2=xm2(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2_2=xm2(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2_2=xm2(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net2.iw{1,1}=reshape(w1_2,hiddennum,inputnum);
net2.lw{2,1}=reshape(w2_2,outputnum,hiddennum);
net2.b{1}=reshape(B1_2,hiddennum,1);
net2.b{2}=reshape(B2_2,outputnum,1);

% �û���ģ���˻�����Ⱥ�㷨�Ż���BP�������ֵԤ��
w1_3=xm3(1:inputnum*hiddennum);
B1_3=xm3(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2_3=xm3(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2_3=xm3(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net3.iw{1,1}=reshape(w1_3,hiddennum,inputnum);
net3.lw{2,1}=reshape(w2_3,outputnum,hiddennum);
net3.b{1}=reshape(B1_3,hiddennum,1);
net3.b{2}=reshape(B2_3,outputnum,1);

%% BP����ѵ��
%����Ⱥ�����������
net1.trainParam.epochs=100;
net1.trainParam.lr = 0.1;
net1.trainParam.goal=1e-3; % ѵ��Ŀ�����
% 
%ģ���˻�����Ⱥ�����������
net2.trainParam.epochs=100;
net2.trainParam.lr=0.1;
net2.trainParam.goal=1e-6;

%����ģ���˻�����Ⱥ�����������
net3.trainParam.epochs=100;
net3.trainParam.lr=0.1;
net3.trainParam.goal=1e-3;

% ѵ������
net1 = train(net1,Data_input,Data_target); % ����Ⱥ
net2 = train(net2,Data_input,Data_target); % ģ���˻�����Ⱥ
net3 = train(net3,Data_input,Data_target); % ����ģ���˻�����Ⱥ

%% �������
test_sim1 = sim(net1,Data_test); % ����Ⱥ
test_sim2 = sim(net2,Data_test); % ģ���˻�����Ⱥ
test_sim3 = sim(net3,Data_test); % ����ģ���˻�����Ⱥ

% ������ݷ���һ����Test_simΪ��������ͨ���������Ԥ�����ֵ
Test_sim1 = mapminmax('reverse',test_sim1,ps_output); % ����Ⱥ
Test_sim2 = mapminmax('reverse',test_sim2,ps_output); % ģ���˻�����Ⱥ
Test_sim3 = mapminmax('reverse',test_sim3,ps_output); % ����ģ���˻�����Ⱥ

%% �㷨������� 
figure(1)
t = 1:M;
plot(t,Pbest1,'b',t,Pbest2,'g',t,Pbest3,'r');
title('�㷨��������');
xlabel('��������');
ylabel('��С�������ֵ��MSEֵ��');
legend('��������Ⱥ�㷨','ģ���˻�����Ⱥ�㷨','����ģ���˻�����Ⱥ�㷨');

figure(2)
mesh(xt1,xt2,Y2);
title('����ʵ��ͼ��');

%% ���ͼ�ζԱ����

%��������Ⱥ�Ա����
figure(3)
subplot(1,2,1)
mesh(xt1,xt2,Y2);
title('����ʵ��ͼ��');
xlabel('X1ȡֵ');ylabel('X2ȡֵ');zlabel('�����Ժ������ֵ');

subplot(1,2,2)
mesh(xt1,xt2,Test_sim1);
title('��������Ⱥ�㷨���ͼ��');
xlabel('X1ȡֵ');ylabel('X2ȡֵ');zlabel('�����Ժ������ֵ');

% % ģ���˻�����Ⱥ�Ա����
figure(4)
subplot(1,2,1)
mesh(xt1,xt2,Y2);
title('����ʵ��ͼ��');
xlabel('X1ȡֵ');ylabel('X2ȡֵ');zlabel('�����Ժ������ֵ');

subplot(1,2,2)
mesh(xt1,xt2,Test_sim2);
title('ģ���˻�����Ⱥ�㷨���ͼ��');
xlabel('X1ȡֵ');ylabel('X2ȡֵ');zlabel('�����Ժ������ֵ');

%����ģ���˻�����Ⱥ���ͼ�ε������
figure(5)
subplot(1,2,1)
mesh(xt1,xt2,Y2);
title('����ʵ��ͼ��');
xlabel('X1ȡֵ');ylabel('X2ȡֵ');zlabel('�����Ժ������ֵ');

subplot(1,2,2)
mesh(xt1,xt2,Test_sim3);
title('����ģ���˻�����Ⱥ�㷨���ͼ��');
xlabel('X��');ylabel('Y��');zlabel('Z��');