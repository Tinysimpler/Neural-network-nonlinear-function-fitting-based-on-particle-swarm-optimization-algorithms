function error = nnfitness(x,hiddennum,net,Data_input,Data_target)
%�ú�������������Ӧ��ֵ
%x          input     ���Ż�������Ȩֵ����ֵ������
%inputnum   input     �����ڵ���
%outputnum  input     ������ڵ���
%net        input     ����
%Data_input input     ѵ����������
%Data_targetinput     ѵ���������
%error      output    ������Ӧ��ֵ

inputnum = size(Data_input,1);
outputnum = size(Data_target,1);

%��ȡ
w1=x(1:inputnum*hiddennum);%ȡ������������������ӵ�Ȩֵ
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);%��������Ԫ��ֵ
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);%ȡ������������������ӵ�Ȩֵ
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);%�������Ԫ��ֵ 

%�����������
% net.trainParam.epochs=20;%��������
% net.trainParam.lr=0.1;%ѧϰ��
% net.trainParam.goal=0.001;%��СĿ��ֵ���
% net.trainParam.show=100;
% net.trainParam.showWindow=0;

%����Ȩֵ��ֵ
net.iw{1,1}=reshape(w1,hiddennum,inputnum);%��w1��1��inputnum*hiddennum��תΪhiddennum��inputnum�еĶ�ά����
net.lw{2,1}=reshape(w2,outputnum,hiddennum);%���ľ���ı����ʽ
net.b{1}=reshape(B1,hiddennum,1);%1��hiddennum�У�Ϊ���������Ԫ��ֵ
net.b{2}=reshape(B2,outputnum,1);

% ѵ������
% net = train(net,Data_input,Data_target);

%�������Ԥ��
an=sim(net,Data_input);
% error=sum((abs(test_sim-Data_target)).^2);%���Ӷ�Ӧ����Ӧ��ֵ����ѵ������С�������ֵ��
% ��ƽ��ֵ,����Ϊ1��100��
test_sim = zeros(1,outputnum);
for i = 1:outputnum
    test_sim (i)= sum(an(:,i))/outputnum; 
end
error1=sum((abs(test_sim-Data_target)).^2);%���Ӷ�Ӧ����Ӧ��ֵ
% ��ƽ��ֵ������Ϊһ���������Ӧ��ֵ
error = sum(error1)/outputnum;



