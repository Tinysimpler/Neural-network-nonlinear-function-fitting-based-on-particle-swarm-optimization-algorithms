function [xm,fv,Pbest] = NNCSAPSO(x,hiddennum,net,Data_input,Data_target,N,w,c1,c2,M,D)

% �û��繫ʽ��������Ⱥ�㷨�е����������r���ں��㷨

xmax = 1*ones(1,D); % �������λ��
xmin = - xmax;      % ������Сλ��
vmax = 0.1*ones(1,D);% ��������ٶ�
vmin = - vmax;       % ������С�ٶ�
wmax = 0.9;
wmin = 0.4;
%% ģ���˻��������
T = 1000;
alpha = 0.98;
%% ��ʼ������λ��&�ٶ�
for i = 1:N
    for j = 1:D
        x(i,j)=rand;
        v(i,j)=rand;
    end
end
%% ��ʼ����Ӧ��ֵ
for i = 1:N
    p(i)=nnfitness(x(i,:),hiddennum,net,Data_input,Data_target);
    y(i,:)=x(i,:);
end
pg = x(N,:); % ȫ������λ��
for i = 1:(N-1)
    if nnfitness(x(i,:),hiddennum,net,Data_input,Data_target) < nnfitness(pg,hiddennum,net,Data_input,Data_target)
        pg=x(i,:);
    end
end
r = rand;
%% ����Ѱ��
for t=1:M
    for i = 1:N
%         w = wmax - (t * (wmax - wmin))/M;
        % ���绯����Ⱥ����
        r = 4 * r * (1-r);
        w = 4 * w * (1-w);
        % �ٶȸ���
        v(i,:)=w * v(i,:) + c1 * r * (y(i,:)-x(i,:)) + c2 * (1-r) * (pg-x(i,:));
        
        % �ٶȷ�Χ����
        if v(i,:) > vmax
            v(i,:) = vmax;
        end
        if v(i,:) < vmin
            v(i,:) = vmin;
        end
        % λ�ø���
        x(i,:)=x(i,:)+v(i,:);
        % λ�÷�Χ����
        if x(i,:) > xmax
            x(i,:) = xmax;
        end
        if x(i,:) < xmin
            x(i,:) = xmin;
        end
        delta = nnfitness(x(i,:),hiddennum,net,Data_input,Data_target) - p(i);
        if delta < 0 || exp(-delta/T) > rand
            p(i)=nnfitness(x(i,:),hiddennum,net,Data_input,Data_target);
            y(i,:)=x(i,:);
        end
        if p(i) < nnfitness(pg,hiddennum,net,Data_input,Data_target)
            pg=y(i,:);
            T = T * alpha;
        end
    end
    Pbest(t)=nnfitness(pg,hiddennum,net,Data_input,Data_target); % ����ÿһ�ε�����������Ӧ��ֵ
end
xm = pg; % ����λ��
fv = nnfitness(pg,hiddennum,net,Data_input,Data_target);% ����λ�ö�Ӧ��������Ӧ��ֵ�����������Ž⣨��Сֵ��