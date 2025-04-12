%SVMd对比算法
    Training=KNN_XL;
   
    [XT_SVM_NUM,~]=size(XT_KNN);
    Group=YL_err;
    cph_SVMmodel=fitcecoc(Training,Group);
    tic
    [cph_label,cph_score]=predict(cph_SVMmodel,XT_KNN);
    test_time_svm(z)=toc;
    k=0;
    for i=1:XT_SVM_NUM
        if cph_label(i)==YT(i)
            k=k+1;
        end
    end
    svm_acc(z)=k/XT_SVM_NUM;
           disp('svm');
           disp(k/XT_SVM_NUM);
%     Yresult=zeros(size(YD));
% %      for i=1:mc %mc个分类需要做mc-1次二分类 或mc个分类
%         %将训练集类标号分成分为i类和~=i类两类
%        YDi=zeros(size(YDtmp));
%         YDi(YLtmp==1)=1;
% %         [mi,~]=size(YDi)
%         X =XD;
%         y = YDi;
% SVMModel = fitcsvm(X,y);
% classOrder = SVMModel.ClassNames;
        
        %SVM二分类
        % SVMModel = fitcsvm(Training,Group);
       % svmStruct = svmtrain(Training,YLi);
%         sr= classificationSVM(svmStruct,Sample); %二分类结果为sr
% %         %提取出已经分好的第i类，利用指针储存第i类数据的位置
%          Yresult(sr==1)=i;
% %          YLtmp=YLtmp(YLtmp~=1);
% %          XDtmp=XDtmp(YLtmp~=1);
% % 
% %         XXD=[XD,sr,(1:sa)'];%将新的类标号与数据集相匹配并标号
% %         XXD=find(XXD(n+1)==i);% 找到第i类数据
% %         sxdx=XXD(:,n+2);%存储第i类数据的序号
% %         svmr=zeros(m,1);
% %         svmr(sxdx)=i;%存储分类结果
% %         
% %         %提取新的数据集
% %         tsdx=setdiff([1:sa]',sxdx);% 找到非i类数据的序号
% %         SP=[X,YD,(1:m)'];%原始数据集
% %         Sample=SP(tsxd,n);%提取出的需要再次进行二分类的数据集
% %         [ta,tb]=size(Training);
% %         Training=[Training,Group,(1:ta)'];
% %         sx=find(Training(n+1)==i);%找到Training中的第i类数据
% %         Training(sx)=[];%删除Training中第i类数据
% %         Training=Training(:,n);%设置新的训练集
% %         Group=Training(:,n+1);%设置新的训练集类标
% %         i=i+1;
% % %     end 
% %     result = abs(svmr-YD);
% %     [cor(sii,1),~]=size(result(result==0));