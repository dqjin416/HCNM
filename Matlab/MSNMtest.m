function [v,u]=MSNMtest(xt,XL,YL,sigm1,w0)

% nway,kshot下ml=5*5=25,n=向量维度
[ml,n]=size(XL);
% 最大标号，nway下为mc=5
mc=max(YL);

% 维度(26,n)
XT=[XL;xt];
ml=ml+1;
% 维度(25+1,1)
sm=zeros(ml,1);
sm(ml)=1;
tmp=w0;
% nway下维度为(26,5)
w0=zeros(ml,mc);
w0(1:ml-1,:)=tmp;

% 计算距离矩阵(26,26)
Wm=dogdistm(XT,sigm1);
% (26,26)*(26,1)=(26,26)
u=thetae(Wm*thetart(sm-0));
% v=thetart((w0'*thetart(u)));
% (5,26)*(26,26)报错
 v=thetart(w0'*thetart(u-0));