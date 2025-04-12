% %用于对比的knn
    %未标准化
    Mdl = fitcknn(KNN_XL, YL, 'NumNeighbors',1);
    tic
    c = predict(Mdl,XT_KNN);
    knn_acc(z)=mean(c==YT);
    test_time_knn(z)=toc;
    disp(['The KNN accuracy =',num2str(knn_acc(z))])
% %标准化
% Mdl = fitcknn(Training, Group, 'NumNeighbors',1,'Standardize',1);
% c = predict(Mdl,Sample);
% result=abs(c-YD);


% K=3;
% trainData = KNN_XL;
% trainClass = YL_err;
% c=zeros(size(XT_KNN,1),1);
% for jj=1:size(XT_KNN,1)
%  
% testData = XT_KNN(jj,:);
% 
% 
% %计算训练数据集与测试数据之间的欧氏距离dist
% dist=zeros(size(trainData,1),1);
% for ii=1:size(trainData,1)
%     dist(ii,:)=norm(trainData(ii,:)-testData);
% end
% %将dist从小到大进行排序
% [~,I]=sort(dist,1);   
% %将训练数据对应的类别与训练数据排序结果对应
% trainClass=trainClass(I);
% %确定前K个点所在类别的出现频率
% classNum=length(unique(trainClass));%取集合中的单值元素的个数
% labels=zeros(1,classNum);
% for iii=1:K
%     jjj=trainClass(iii);
%     labels(jjj)=labels(jjj)+1;
% end
% %返回前K个点中出现频率最高的类别作为测试数据的预测分类
% [~,idx]=max(labels);
% c(jj)=idx;
% end
%  knn_acc(z)=mean(c==YT);
%  disp(['The KNN accuracy =',num2str(knn_acc(z))])