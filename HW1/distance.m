function[dist] = distance(x,y)
   dist = 0;
   for i=1:size(x)
       dist = dist + ((x(i) - y(i))^2);%Accumulate
   end
   dist = sqrt(dist);%Euclidean
end
