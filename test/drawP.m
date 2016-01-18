function [ output_args ] = drawP( num )
%DRAWP 此处显示有关此函数的摘要
%   此处显示详细说明
load('matlab.mat')
str = ['mat', num2str(i)];
eval(['m=', str,';']);
length = size(m);

for j=1:max(output(:,4))
    k=1;   
    for i=1:length(1,1)
        my(i,1) = m(i, 1);
        my(i,2) = m(i, 2);
        my(i,3) = output(i, 3);
        if( output(i,4)==j-1)
            mx{j}(1,k) = m(i, 1);
            mx{j}(2,k) = m(i, 2);
            mx{j}(3,k) = output(i, 3);
            k = k+1;
        end
    end
end
k=0;
scatter(m(:,1), m(:,2), 'b'); %original

figure
color = ['r', 'g', 'c', 'k'];
for i=1:max(output(:,4))
    %subplot(2,2,4);  
    num = size(mx{i});
    if(num(1,2)~=1)
        hold on
        scatter(mx{i}(1,:), mx{i}(2,:), color(mod(k, 4)+1));
        k = k+1;       
    end
   
end

 
end