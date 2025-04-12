
load('/data/imag_resnet12_features50R');
load('/data/imag_resnet12_lables50R');
features =imag_resnet12_features;
labels= imag_resnet12_lables;

class = round(unifrnd(1,20,1,nways));
while length(unique(class))~=length(class)
    class = round(unifrnd(1,20,1,nways));
end
data=zeros(600*nways,size(features,2));
data_label=zeros(600*nways,1);
instance=zeros(nways*sample_num,size(features,2));
instance_label=zeros(nways*sample_num,1);
s=1;
s2=1;
for i=1:nways
    class2 = round(unifrnd(1,600,1,sample_num));
while length(unique(class2))~=length(class2)
    class2 = round(unifrnd(1,600,1,sample_num));
end
   data(s:599+s,:)=features(labels==class(i),:);
   x=data(s:599+s,:);
   instance(s2:sample_num-1+s2,:)=x(class2,:);
   data_label(s:599+s)=labels(labels==class(i));
   y=data_label(s:599+s);
   instance_label(s2:sample_num-1+s2)=y(class2,:);
   instance_label(s2:sample_num-1+s2)=i;
   s=s+600;
   s2=s2+sample_num;
end

