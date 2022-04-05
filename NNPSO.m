function [xm,fv,Pbest] = NNPSO(x,hiddennum,net,Data_input,Data_target,N,w,c1,c2,M,D)
xmax = 1*ones(1,D);
xmin = - xmax;
vmax = 0.1*ones(1,D);
vmin = - vmax;

% ��ʼ������λ��&�ٶ�
for i = 1:N
    for j = 1:D
        x(i,j)=rand;
        v(i,j)=rand;
    end
end

% ��ʼ����Ӧ��ֵ
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

%% ����Ѱ��
for t=1:M
    for i = 1:N
        % �ٶȸ���
        v(i,:)=w * v(i,:) + c1 * rand * (y(i,:)-x(i,:)) + c2 * rand * (pg-x(i,:));
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
        if nnfitness(x(i,:),hiddennum,net,Data_input,Data_target) < p(i)
            p(i)=nnfitness(x(i,:),hiddennum,net,Data_input,Data_target);
            y(i,:)=x(i,:);
        end
        if p(i) < nnfitness(pg,hiddennum,net,Data_input,Data_target)
            pg=y(i,:);
        end
    end
    Pbest(t)=nnfitness(pg,hiddennum,net,Data_input,Data_target); % ����ÿһ�ε�����������Ӧ��ֵ
end
xm = pg; % ����λ��
fv = nnfitness(pg,hiddennum,net,Data_input,Data_target);% ����λ�ö�Ӧ��������Ӧ��ֵ�����������Ž⣨��Сֵ��