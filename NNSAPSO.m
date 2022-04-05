function [xm,fv,Pbest] = NNSAPSO(x,hiddennum,net,Data_input,Data_target,N,w,c1,c2,M,D)
xmax = 1*ones(1,D); % 粒子最大位置
xmin = - xmax;      % 粒子最小位置
vmax = 0.1*ones(1,D);% 粒子最大速度
vmin = - vmax;       % 粒子最小速度
wmax = 0.9;
wmin = 0.4;
%% 模拟退火参数设置
T = 1000;
alpha = 0.98;
%% 初始化粒子位移&速度
for i = 1:N
    for j = 1:D
        x(i,j)=rand;
        v(i,j)=rand;
    end
end
%% 初始化适应度值
for i = 1:N
    p(i)=nnfitness(x(i,:),hiddennum,net,Data_input,Data_target);
    y(i,:)=x(i,:);
end
pg = x(N,:); % 全局最优位置
for i = 1:(N-1)
    if nnfitness(x(i,:),hiddennum,net,Data_input,Data_target) < nnfitness(pg,hiddennum,net,Data_input,Data_target)
        pg=x(i,:);
    end
end

%% 迭代寻优
for t=1:M
    for i = 1:N
        w = wmax - (t * (wmax - wmin))/M;
        % 速度更新
        v(i,:)=w * v(i,:) + c1 * rand * (y(i,:)-x(i,:)) + c2 * rand * (pg-x(i,:));
        % 速度范围限制
        if v(i,:) > vmax
            v(i,:) = vmax;
        end
        if v(i,:) < vmin
            v(i,:) = vmin;
        end
        % 位置更新
        x(i,:)=x(i,:)+v(i,:);
        % 位置范围限制
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
    Pbest(t)=nnfitness(pg,hiddennum,net,Data_input,Data_target); % 保留每一次迭代的最优适应度值
end
xm = pg; % 最优位置
fv = nnfitness(pg,hiddennum,net,Data_input,Data_target);% 最优位置对应的最优适应度值，即函数最优解（最小值）