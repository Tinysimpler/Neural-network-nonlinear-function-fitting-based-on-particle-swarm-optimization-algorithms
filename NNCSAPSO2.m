function [xm,fv,Pbest] = NNCSAPSO2(x,hiddennum,net,Data_input,Data_target,N,w,c1,c2,M,D)

% �û���ֲ������ķ�ʽ���㷨�ں�
%% ����Ⱥ��������
xmax = 1*ones(1,D);
xmin = - xmax;
vmax = 0.1*ones(1,D);
vmin = - vmax;
wmax = 0.9;
wmin = 0.4;
%% ����ֲ�����������
MaxC = 8; 
%% ģ���˻��������
T = 1000;
alpha = 0.98;
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
r = rand;
%% ����Ѱ��
for t=1:M
    %% ģ���˻��㷨������λ��
    f_gb = nnfitness(pg,hiddennum,net,Data_input,Data_target);
    for i = 1:N  % ��ǰ�¶��¸������ӵ���Ӧ��
        Tfitness(i) = exp(- (p(i) - f_gb)/T);
    end
    SumTfitness = sum(Tfitness);
    Tfitness  = Tfitness./SumTfitness;
    for i = 1:N
        ComFitness(i)=sum(Tfitness(1:i));
        if rand <= ComFitness(i) % ���̶ķ�ʽȷ��ȫ�����ŵ�ĳ�����ֵ
            pg = x(i,:);
            break;
        end
    end
   %% ����Ⱥ�㷨������ʼ
    for i = 1:N
        w = wmax - (t * (wmax - wmin))/M;
        % �ٶȸ���
        v(i,:)=w * v(i,:) + c1 * r * (y(i,:)-x(i,:)) + c2 * r * (pg-x(i,:));

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
        f_p(i) = nnfitness(x(i,:),hiddennum,net,Data_input,Data_target);
    end
    
    %% ����ֲ�����
    [sort_f_p,index] = sort(f_p);
    Nbest = floor(N * 0.2);
    for n = 1:Nbest
        Tmp_x = x(index(n),:);   
        for k = 1:MaxC
            for i = 1:D
                Cx(i) = (Tmp_x(1,i) - xmin(i)) / (Tmp_x(1,i) - xmax(i));
                Cx(i) = 4 * Cx(i) * (1 - Cx(i));
                Tmp_x(1,i) = Tmp_x(1,i) + Cx(i) * (xmax(i) - xmin(i));
            end
            fcs = nnfitness(Tmp_x,hiddennum,net,Data_input,Data_target);
            if fcs < sort_f_p(n) % �Ի���������ľ��߱�����������
                x(index(n),:) = Tmp_x;
                break;
            end
        end
        x(index(n),:) = Tmp_x;
    end
    for s = 1:D % ������������
        xmin(s) = max(xmin(s),pg(s) - rand * (xmax(s) - xmin(s)) );
        xmax(s) = min(xmax(s),pg(s) + rand * (xmax(s) - xmin(s)) );
    end
    x(1:Nbest,:) = x(index(1:Nbest),:);
    for i = (Nbest + 1):N % �������ʣ��80%΢��
        for j = D
            x(i,j) = xmin(j) + rand * (xmax(j) - xmin(j)); % �����ʼ��λ��
            v(i,j) = rand; % �����ʼ���ٶ�
        end
    end
   %% ���¸�������λ�� Ⱥ������λ��
    for i = 1:N
        if nnfitness(x(i,:),hiddennum,net,Data_input,Data_target) < p(i)
            p(i) = nnfitness(x(i,:),hiddennum,net,Data_input,Data_target);
            y(i,:) = x(i,:);
        end
        if p(i) < nnfitness(pg,hiddennum,net,Data_input,Data_target)
            pg = y(i,:);
        end
    end
    T = T * alpha; % ģ���˻��²���
    Pbest(t)=nnfitness(pg,hiddennum,net,Data_input,Data_target); % ����ÿһ�ε�����������Ӧ��ֵ
end
xm = pg; % ����λ��
fv = nnfitness(pg,hiddennum,net,Data_input,Data_target);% ����λ�ö�Ӧ��������Ӧ��ֵ�����������Ž⣨��Сֵ��