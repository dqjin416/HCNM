function [L,ratioL]=Sample(X_15shot,Y_15shot,kshot,type)
% 获得m(表示m个样本)
[m,~]=size(X_15shot);
mc=max(Y_15shot);
P=(1:m)';
if strcmp(type,'balance')
%     mlk=floor(floor(m*kshot)/mc);
    mlk=kshot;
    ml=mc*mlk;
    L=zeros(ml,1);
    for k=1:mc
        Pk=P(Y_15shot==k);
        [mk,~]=size(Pk);
        Ltmp=randperm(mk,mlk);
        Lc=Pk(Ltmp);
        L((k-1)*mlk+1:k*mlk)=Lc;
    end
    ratioL=ml/m;
else
    for k=1:mc        
        Pk=P(Y_15shot==k);
        [mk,~]=size(Pk);
        mlk=max(floor(mk*kshot),1);
        Ltmp=randperm(mk,mlk);
        Lc=Pk(Ltmp);
        L((k-1)*mlk+1:k*mlk)=Lc;
    end
    L(L==0)=[];
    [Lr,~]=size(L);
    ratioL=Lr/m;
end