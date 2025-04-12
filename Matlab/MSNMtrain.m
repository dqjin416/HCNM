function W0=MSNMtrain(YL)
mc=max(YL);
[ml,~]=size(YL);
W0=zeros(ml,mc);
for j=1:mc
    for i=1:mc
    for jj=1:ml
        if YL(jj)==i
            W0(jj,i)=1;
        end
    end
    end
end
end