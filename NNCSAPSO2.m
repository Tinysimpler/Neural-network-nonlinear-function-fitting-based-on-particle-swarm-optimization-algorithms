function [xm,fv,Pbest] = NNCSAPSO2(x,hiddennum,net,Data_input,Data_target,N,w,c1,c2,M,D)

% 用混沌局部搜索的方式做算法融合
%% 粒子群参数设置
xmax = 1*ones(1,D);
xmin = - xmax;
vmax = 0.1*ones(1,D);
vmin = - vmax;
wmax = 0.9;
wmin = 0.4;
%% 混沌局部最大迭代步数
MaxC = 8; 
%% 模拟退火参数设置
T = 1000;
alpha = 0.98;
% 初始化粒子位移&速度
for i = 1:N
    for j = 1:D
        x(i,j)=rand;
        v(i,j)=rand;
    end
end
% 初始化适应度值
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
r = rand;
%% 迭代寻优
for t=1:M
    %% 模拟退火算法求最优位置
    f_gb = nnfitness(pg,hiddennum,net,Data_input,Data_target);
    for i = 1:N  % 当前温度下各个粒子的适应度
        Tfitness(i) = exp(- (p(i) - f_gb)/T);
    end
    SumTfitness = sum(Tfitness);
    Tfitness  = Tfitness./SumTfitness;
    for i = 1:N
        ComFitness(i)=sum(Tfitness(1:i));
        if rand <= ComFitness(i) % 轮盘赌方式确定全局最优的某个替代值
            pg = x(i,:);
            break;
        end
    end
   %% 粒子群算法迭代开始
    for i = 1:N
        w = wmax - (t * (wmax - wmin))/M;
        % 速度更新
        v(i,:)=w * v(i,:) + c1 * r * (y(i,:)-x(i,:)) + c2 * r * (pg-x(i,:));

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
        f_p(i) = nnfitness(x(i,:),hiddennum,net,Data_input,Data_target);
    end
    
    %% 混沌局部搜索
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
            if fcs < sort_f_p(n) % 对混沌搜索后的决策变量进行评估
                x(index(n),:) = Tmp_x;
                break;
            end
        end
        x(index(n),:) = Tmp_x;
    end
    for s = 1:D % 收缩搜索区域
        xmin(s) = max(xmin(s),pg(s) - rand * (xmax(s) - xmin(s)) );
        xmax(s) = min(xmax(s),pg(s) + rand * (xmax(s) - xmin(s)) );
    end
    x(1:Nbest,:) = x(index(1:Nbest),:);
    for i = (Nbest + 1):N % 随机产生剩余80%微粒
        for j = D
            x(i,j) = xmin(j) + rand * (xmax(j) - xmin(j)); % 随机初始化位置
            v(i,j) = rand; % 随机初始化速度
        end
    end
   %% 更新个体最优位置 群体最优位置
    for i = 1:N
        if nnfitness(x(i,:),hiddennum,net,Data_input,Data_target) < p(i)
            p(i) = nnfitness(x(i,:),hiddennum,net,Data_input,Data_target);
            y(i,:) = x(i,:);
        end
        if p(i) < nnfitness(pg,hiddennum,net,Data_input,Data_target)
            pg = y(i,:);
        end
    end
    T = T * alpha; % 模拟退火降温操作
    Pbest(t)=nnfitness(pg,hiddennum,net,Data_input,Data_target); % 保留每一次迭代的最优适应度值
end
xm = pg; % 最优位置
fv = nnfitness(pg,hiddennum,net,Data_input,Data_target);% 最优位置对应的最优适应度值，即函数最优解（最小值）