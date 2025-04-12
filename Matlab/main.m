clc
clear

sampletype='balance';
dataset='cifar';
nways=5;
kshot=5;
sample_num=20;
times=600;
new_acc=zeros(times,1);
org_acc=zeros(times,1);
svm_acc=zeros(times,1);
knn_acc=zeros(times,1);
newPCA_acc=zeros(times,1);
orgPCA_acc=zeros(times,1);
pcanet_knn_acc=zeros(times,1);
pcanet_svm_acc=zeros(times,1);
all_time_new=zeros(times,1);
all_time_org=zeros(times,1);
test_time_new=zeros(times,1);
test_time_org=zeros(times,1);

% %%%%%%%%%%%%%%%%%Parameters of PCANET
% PCANet.NumStages = 2;
% PCANet.PatchSize = [3 3];
% PCANet.NumFilters = [4 7] ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
% PCANet.HistBlockSize = [3 7];
% PCANet.BlkOverLapRatio = 0;
% PCANet.Pyramid = [ ];
% ImgFormat = 'gray';
% PCANet;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%

for z= 1:times
    if strcmp(dataset,'cub')
        cub;
    elseif strcmp(dataset,'cifar')
        cifar;
    elseif strcmp(dataset,'imagenet')
        imagenet;
    end
    
    [L,ratioL]=Sample(instance,instance_label,kshot,sampletype);
    % [COEFF, SCORE, LATENT, TSQUARED] = pca((instance)) ;
    % instance=SCORE(:,1:10);
    %instance = tsne(instance, 'NumDimensions', 10);
    % ica = rica(instance, 10); % 使用 10 个独立成分
    % instance = transform(ica, instance);
    S_set=instance(L,:);
    S_set_label=instance_label(L,:);
    [mt,~]=size(instance);
    UL=setdiff((1:mt)',L);%未被采样的样本索引
    % [~,~,~,instance,S_set_ORG]=IniSig(S_set,instance,L);
    % sigm1=1;

    % [Labeled_X, Labeled_Y, Unlabeled_X, Unlabeled_Y]=select(S_set, S_set_label, 0.1);
    % [VSS,VSS_Y]=VSSCon_supervise(S_set,S_set_label,Labeled_X,Labeled_Y);
    [VSS,VSS_Y]=VSSCon(S_set,S_set_label);
    Q_set=instance(UL,:);
    Q_set_label=instance_label(UL,:);
    tic

    % % 使用 PCANet 提取特征
    % [PCANet_Fea_S, ~] = PCANet_train(S_set,PCANet); % 提取支持集特征
    % [PCANet_Fea_Q, ~] = PCANet_train(Q_set,PCANet); % 提取查询集特征
    % 
    % [VSSp,VSSp_Y]=VSSCon(PCANet_Fea_S,S_set_label);
    % 
    [Q_set_label_pre,test_time_org(z)]=MSNMclassifier(S_set,S_set_label,Q_set);
    [Q_set_label_pre1,test_time_new(z)]=MSNMclassifier(VSS,VSS_Y',Q_set);
    % 
    % %导入特征的knn分类
    % Q_set_knn = fitcknn(S_set, S_set_label, 'NumNeighbors', 1);
    % Q_set_label_KNN = predict(Q_set_knn, Q_set);
    % knn_acc(z) = mean(Q_set_label_KNN == Q_set_label);
    % fprintf('This is the %d th running \n',z);
    % disp(['The knnaccuracy =',num2str(knn_acc(z))])
    % 
    % %导入的svm分类
    % Q_set_svm = fitcecoc(S_set, S_set_label);  % 使用SVM进行训练
    % Q_set_label_SVM = predict(Q_set_svm, Q_set);  % 对查询集进行预测
    % svm_acc(z) = mean(Q_set_label_SVM == Q_set_label);
    % fprintf('This is the %d th running \n',z);
    % disp(['The svmaccuracy =',num2str(svm_acc(z))])
    % 
    % % 使用最近邻分类器进行分类
    % Mdl = fitcknn(PCANet_Fea_S, S_set_label, 'NumNeighbors', 1);
    % Q_set_label_pred = predict(Mdl, PCANet_Fea_Q);
    % % 计算 PCANet 分类的准确率
    % pcanet_knn_acc(z) = mean(Q_set_label_pred == Q_set_label);
    % fprintf('This is the %d th running \n',z);
    % disp(['The pca_knn accuracy =',num2str(pcanet_knn_acc(z))])
    % 
    % % 使用 SVM 分类器进行分类
    % SVMModel = fitcecoc(PCANet_Fea_S, S_set_label);  % 使用SVM进行训练
    % Qsvm_set_label_pred = predict(SVMModel, PCANet_Fea_Q);  % 对查询集进行预测
    % % 计算 svm_pca 分类的准确率
    % pcanet_svm_acc(z) = mean(Qsvm_set_label_pred == Q_set_label);
    % fprintf('This is the %d th running \n',z);
    % disp(['The pca_svm accuracy =',num2str(pcanet_svm_acc(z))])
    % 
    % %使用我们的模型进行分类
    % [Q_set_label_prePCA,test_time_org(z)]=MSNMclassifier(PCANet_Fea_S,S_set_label,PCANet_Fea_Q);
    % [Q_set_label_pre1PCA,test_time_new(z)]=MSNMclassifier(VSSp,VSSp_Y',PCANet_Fea_Q);
    % 
    % orgPCA_acc(z)=mean(Q_set_label_prePCA==Q_set_label);
    % fprintf('This is the %d th running \n',z);
    % disp(['The orgPCA accuracy =',num2str(orgPCA_acc(z))])
    % 
    % newPCA_acc(z)=mean(Q_set_label_pre1PCA==Q_set_label);
    % fprintf('This is the %d th running \n',z);
    % disp(['The newPCA accuracy by clustering =',num2str(newPCA_acc(z))])

    all_time_org(z)=toc;    
    org_acc(z)=mean(Q_set_label_pre==Q_set_label);
    fprintf('This is the %d th running \n',z);
    disp(['The accuracy =',num2str(org_acc(z))])

    all_time_new(z)=toc;
    new_acc(z)=mean(Q_set_label_pre1==Q_set_label);
    fprintf('This is the %d th running \n',z);
    disp(['The accuracy by clustering =',num2str(new_acc(z))])

  
    % [COEFF, SCORE, LATENT, TSQUARED] = pca((instance)) ;
    % instance_redu=SCORE(:,1:10);
    % S_set=instance_redu(L,:);
    % S_set_label=instance_label(L,:);
    % Q_set=instance_redu;
    % Q_set_label=instance_label;
    % Q_set(L,:)=[];
    % Q_set_label(L,:)=[];
    % Q_set_label_pre=MSNMclassifier(S_set,S_set_label,Q_set);
    % acc(z)=mean(Q_set_label_pre==Q_set_label);
    
end
disp(['The ', num2str(nways),'way-',num2str(kshot),'shot orgacc of ',dataset,' = ',num2str(mean(org_acc))])
disp(['The ', num2str(nways),'way-',num2str(kshot),'shot newacc of ',dataset,' = ',num2str(mean(new_acc))])
disp(['The knn accuracy =',num2str(mean(knn_acc))])
disp(['The svm accuracy =',num2str(mean(svm_acc))])
disp(['The testing new time by clustering=',num2str(mean(test_time_new))])
disp(['The origin train time =',num2str(mean(all_time_org-test_time_org))])
% 创建表格保存准确性结果
results_table = table((1:times)', org_acc, new_acc,knn_acc,svm_acc,pcanet_knn_acc,pcanet_svm_acc,orgPCA_acc,newPCA_acc,test_time_new ,test_time_org, all_time_org, all_time_new, ...
    'VariableNames', {'Run', 'Original_Accuracy', 'New_Accuracy','Knn_Accuary','Svm_Accuary','pcanet_knn_Acc','pcanet_svm_Acc','orgpca_Acc','newpca_Acc','Test_Time_New', 'Test_Time_Org', 'All_Time_Org', 'All_Time_New'});

% 将表格保存为 CSV 文件
writetable(results_table, 'accuracy_results.csv');

% 显示表格
disp(results_table);