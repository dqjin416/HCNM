%凝聚算法准确率
[rr,~]=size(cluster_numpos);

%求出每次聚类的结果（m个数据分被属于哪一类）
failrate=zeros(rr,1);
bianhua=zeros(rr,1);
%标准类标矩阵MY

CYD1=zeros(m);
%     CY=YL;
    CY=YL1{1};
    for z=1:m
        for k=1:z
            if CY(z)==CY(k)
                CYD1(z,k)=1;
            end
        end
    end
for i=2:rr%凝聚第i层的类标号CY
    CYD=zeros(m);
    CY=YL1{i};
%     CY=YL;
    for z=1:m
        for k=1:z
            if CY(z)==CY(k)
                CYD(z,k)=1;
            end
        end
    end
    cha=CYD~=CYD1;
    v=diag(cha);
    b=cha+diag(v)*(-1)+cha';  
    bh=sum(sum(b~=0))/(m*m); 
    bianhua(i)=bh;
    CYD1=CYD;
end
daicaicishu=1:1:rr;
disp('迭代次数=  各轮类个数=  各轮较上一轮变化=')
disp([daicaicishu',cluster_numpos,bianhua])
