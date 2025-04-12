function [VSS,VSS_Y,sig_times]=VSSCon(XL,YL)

[YL1,cluster_num,huatu,II]=MSNMcluster(XL);%Y是存储各次聚类结果的细胞数组，cluster_num是各次聚类中类的个数
% save YL1 YL1
cluster_numpos=cluster_num(cluster_num>0);
Num(:,1)=unique(cluster_numpos);
[n_bin,~]=size(Num);
[Num(:,2),~] = histc(cluster_numpos,Num(:,1));

[m,~]=size(XL);
clusteraccuracy;

% [~,III]=max(bianhua);
% sig_times=III-1;
% III=2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DYL=zeros(size(YL,1));
for i=1:size(YL,1)
    for j=1:size(YL,1)
        if YL(i)==YL(j)
            DYL(i,j)=1;
        end
    end
end


DYL11=zeros(size(YL,1));
DYL1=cell(size(YL1,1),1);
for i=1:size(YL1,1)
    for j=1:size(YL,1)
        for k=1:size(YL,1)
            if YL1{i}(j)==YL1{i}(k)
                DYL11(j,k)=1;
                DYL1{i}=DYL11;
            end
        end
    end
end


D1=cell(size(YL1,1),1);
parfor i=1:size(YL1,1)
    D1{i}=DYL-DYL1{i};
end
D2=D1;
parfor i=1:size(YL1,1)
    for j=1:size(YL,1)
        for k=1:size(YL,1)
            if D2{i}(j,k)==-1
                D2{i}(j,k)=0;
            end
        end
    end
end
D3=D1;
parfor i=1:size(YL1,1)
    for j=1:size(YL,1)
        for k=1:size(YL,1)
            if D3{i}(j,k)==1
                D3{i}(j,k)=0;
            end
        end
    end
end
D4=D3;
parfor i=1:size(YL1,1)
    for j=1:size(YL,1)
        for k=1:size(YL,1)
            if D4{i}(j,k)==-1
                D4{i}(j,k)=1;
            end
        end
    end
end

[idx]=kmeans(XL,max(YL));%聚类个数为40
% k_acc=zeros(size(YL,1),1);
idx1=zeros(size(YL,1));
for i=1:size(YL,1)
    for j=1:size(YL,1)
        if idx(i)==idx(j)
            idx1(i,j)=1;
        end
    end
end
k_acc=DYL-idx1;
k_acc1=(k_acc~=0);
ml=size(YL,1);
kmeans_acc=(ml*ml-sum(k_acc1(:)))/(ml*ml);
lambda=kmeans_acc;


% D_NEW=cell(size(YL1,1),1);
% for i=1:size(YL1,1)
%     D_NEW{i}=D2{i}-lambda*D4{i};
% end

D_NEW=zeros(size(YL1,1),1);
for i=1:size(YL1,1)
    D2_NEW=(D2{i}~=0);
    D4_NEW=(D4{i}~=0);
    D_NEW(i)=sum(D2_NEW(:))-lambda*sum(D4_NEW(:));
end
% [D_MIN,NEW_sig]=min(D_NEW);
[D_MIN,NEW_sig]=min(D_NEW>0);
% cluster_num(NEW_sig);
III=NEW_sig;
cluster_num(III)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% VSS=zeros(mc,size(XL,2));%Select two VSS for each category
% VSS_Y=zeros(mc,1);%the class label of VSS
mc = max(YL);
VSS=[];
VSS_Y=[];
for i=1:mc
    LYL=find(YL==i);%Find the samples belonging to class i in Original training set.
    SXL=XL(LYL,:); 
    [SYL,cluster_num,huatu,~]=MSNMcluster(SXL);%Y是存储各次聚类结果的细胞数组，cluster_num是各次聚类中类的个数
    VSS_I_J=zeros(cluster_num(III),size(SXL,2));
     VSS_Y_I_J=zeros(1,cluster_num(III));
     
    for j=1:cluster_num(III)  
        SYL1=find(SYL{III}==j);
        SXL1=SXL(SYL1,:);
        VSSi_1=mean(SXL1,1);
        VSS_I_J(j,:)=VSSi_1;
        VSS_Y_I_J(j)=i;   
%     VSSi_1=mean(SXL1,1);
%     VSS(i,:)=VSSi_1;
%     VSS_Y(i)=i;   
    end
    VSS=[VSS;VSS_I_J];
    VSS_Y=[VSS_Y,VSS_Y_I_J];
end
    
end
