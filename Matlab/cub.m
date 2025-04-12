
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 

load("/data/cub_resnet12_features(xiazai).mat")
load('/data/cub_resnet12_lables(xiazai).mat')
features=features;
lables=labels;
% 获取所有唯一的类别
unique_labels = unique(lables);
% 随机选择 nways 个唯一类别
class = unique_labels(randperm(length(unique_labels), nways));
% class = round(unifrnd(1,50,1,nways));
while length(unique(class))~=length(class)
    class = round(unifrnd(1,50,1,nways));
end

data=zeros(size(features,1),size(features,2));
data_label=zeros(size(features,1),1);
data_label_unique=unique(class);
a=1;
for i =1:nways
    ii=data_label_unique(i);
    b=lables(lables==ii);    
    i_num=length(b);
    data(a:i_num+a-1,:)=features((lables==ii),:);
    data_label(a:i_num+a-1,:)=i;
    a=a+i_num;
end
data(data_label==0,:)=[];
data_label(data_label==0)=[];
instance=zeros(nways*sample_num,size(features,2));
instance_label=zeros(nways*sample_num,1);
c=1;
d=1;
for i=1:nways
        len=length(lables(lables==class(i)));
    class2 = round(unifrnd(1, len, 1, sample_num));
while length(unique(class2))~=length(class2)
   class2 = round(unifrnd(1, len, 1, sample_num));
end 
   xxx=data(c:len-1+c,:);
   instance(d:sample_num-1+d,:)=xxx(class2,:);
   instance_label(d:sample_num-1+d)=i;
   c=c+len;
   d=d+sample_num;
end

        
% end   

