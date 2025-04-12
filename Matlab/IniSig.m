function [sigm,sigm1max,sigm1min,XT,XL]=IniSig(XL,XT,L)
[~,n]=size(XL);%ml为标号数据数量，n为数据维数
sigm=zeros(n,1);
sigm1min=zeros(n,1);
sigm1max=zeros(n,1);
    for j=1:n
        DistXT=squareform(pdist(XT(:,j)));
        DistmpL=DistXT(L,:);        
        sigm(j)=max(max(max(DistmpL)),eps);
        sigm1max(j)=max(max(max(DistmpL)),eps);
        sigm1min(j)=max(min(mean(DistmpL)),eps);
    end
    sigm_xl=repmat(sigm',[size(XL,1),1]);
    sigm_xt=repmat(sigm',[size(XT,1),1]);
    XL=XL./sigm_xl;
    XT=XT./sigm_xt;
    sigm=mean(sigm);