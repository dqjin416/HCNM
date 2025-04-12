function [Yseq,cluster_num,huatu,II,tmp1]=MSNMcluster(X)


[m,n]=size(X);
% X_norm = normalize(X,'scale');
X_norm=X;
sig_times=30;%以最小值为步长，达到最大值的步数（ceil为取离此数最近的大于或等于的最大整数，）
huatu=cell(sig_times,1);
II=cell(sig_times,1);
cluster_times=sig_times;
sigm1step=(1-0.3)/sig_times;%
Yseq=cell(cluster_times,1);%记录每次分类结果
cluster_num=zeros(cluster_times,1);%记录每次聚类中类的数量
sigm=[0.01:sigm1step:1];
tmp1=cell(sig_times,1);
for i=1:cluster_times
    
    X=X_norm;
    sigm1=sigm(i);%宽度步长递增

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%最大值优先   
    Wm=dogdistm(X,sigm1);%计算DoG滤波后各点间相互作用矩阵
   
    tmp=sum(Wm);%各个点接受的层向相互作用和
   
    [~,I] = sort(tmp,'descend');%按滤波后波峰值降序排列，I为对应X的序号
    tmp1{i}=tmp;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%随机选取
%     I=randperm(m);
    YT=zeros(m,1);%存储数据原始排序下的类标号 
    XL=X(I(1),:);%用于聚类的初始点
    YL=1;%记录已分类数据的类标号，与XL对应
    XL2=X(I(1),:);%用于聚类的初始点
    YL2=1;%记录已分类数据的类标号，与XL对应
    seq=1:m;
    for j=1:m          
        [W0]=MSNMtrain(YL);        
        Ij=I(j);
        xt=X(Ij,:);
        ud=MSNMtest(xt,XL,YL,sigm1,W0);
        [udm,im]=max(ud);  
        ud_greater0=length(ud(ud>0));      
        if ud_greater0==0      
            mc=setdiff(seq,YL);
            YT(Ij)=mc(1);
               tmp=[YL;YT(Ij)]; 
                YL=tmp;
                tmp=[XL;xt];
                XL=tmp;
        end 

        if ud_greater0==1
            YT(Ij)=im;
               tmp=[YL;YT(Ij)]; 
                YL=tmp;
                tmp=[XL;xt];
                XL=tmp;
        end
         if ud_greater0>1 
            YT(Ij)=im;
            [~,index]=sort(ud,'descend');           
           Wm2=dogdistm(normalize([(XL(YL==im,:));xt]),1);  
            if sum(Wm2(end,:))>0
                tmp=[YL;YT(Ij)]; 
                YL=tmp;
                tmp=[XL;xt];
                XL=tmp;
            end
         end
        tmp=[YL2;YT(Ij)]; 
        YL2=tmp;
        tmp=[XL2;xt];
        XL2=tmp;
    end

    Yseq{i}=YT;
    cluster_num(i)=numel(unique(YL2));
    cluster_num_data=cell(cluster_num(i),1);
    for ii=1:cluster_num(i)
        cluster_num_data{ii}=XL2(YL2==ii,:);
    end
   II{i}=I;
    huatu{i}=cluster_num_data;
%     disp(['第',num2str(i),'轮类个数：',num2str(cluster_num(i)),'个']);
   

end