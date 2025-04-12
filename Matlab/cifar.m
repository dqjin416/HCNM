
load('/data/cifar_resnet12_features36');
load('/data/cifar_resnet12_lables36');
features= cifar_resnet12_features;
lables= cifar_resnet12_lables;

  
class = round(unifrnd(1,20,1,nways));
while length(unique(class))~=length(class)
    class = round(unifrnd(1,20,1,nways));
end
data=zeros(600*nways,size(features,2));
data_label=zeros(600*nways,1);
instance=zeros(nways*sample_num,size(features,2));
instance_label=zeros(nways*sample_num,1);
a=1;
b=1;
for i=1:nways
    class2 = round(unifrnd(1,600,1,sample_num));
while length(unique(class2))~=length(class2)
    class2 = round(unifrnd(1,600,1,sample_num));
end
   data(a:599+a,:)=features(lables==class(i),:);
   x=data(a:599+a,:);
   instance(b:sample_num-1+b,:)=x(class2,:);
   data_label(a:599+a)=lables(lables==class(i));
   y=data_label(a:599+a);
   instance_label(b:sample_num-1+b)=y(class2,:);
   instance_label(b:sample_num-1+b)=i;
   a=a+600;
   b=b+sample_num;
end
