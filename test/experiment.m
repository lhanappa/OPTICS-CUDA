
% 
% for i=1:6
%     a = 800;
%     b = 1200;
%     c = 1000;
%     a = 4*a*i;
%     b = 4*b*i;
%     c = 4*c*i;
%     creatdata(a, b, c, i);
% end

dos('run.bat','-echo')

for i=1:6
    str1 = ['output0', num2str(i), '.txt'];
    m = load(str1);
    sum(i) = check(m);
    str2 = ['output0' num2str(num)];
    save('matlab.mat', str2, '-append');
end

for i=50:50:300
    for j=1:6
        str1 = ['time_', num2str(i), '_', num2str(j),  '.txt'];
        m = load(str1);
        str2 = ['time_', num2str(i), '_', num2str(j)];
        save('matlab.mat', str2, '-append');        
        time(i,j) = sum(m);
    end
end
 save ('matlab.mat', 'sum', 'time', '-append');