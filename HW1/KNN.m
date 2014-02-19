clear
% Load data in to 5 vectors
file = 'iris.data'
[sepalLen,sepalWid,petalLen,petalWid,class] = textread(file,'%f,%f,%f,%f,%s');
% Normalise the data
sepalLen = normalise(sepalLen);
sepalWid = normalise(sepalWid);
petalLen = normalise(petalLen);
petalWid = normalise(petalWid);
% Turn classes in to numbers
for i =1:size(class)
   if(strcmp(class{i},'Iris-setosa'))       
      num(i) = 1;    
   end
   if(strcmp(class{i},'Iris-versicolor'))   
      num(i) = 2;    
   end
   if(strcmp(class{i},'Iris-virginica'))    
      num(i) = 3;    
   end
end
% Display 2D for sanity
figure(1)
   % X = sepal len
   subplot(4,4,1)
      scatter(sepalLen,sepalLen,30,num,'.')
      ylabel('Sepal length')
      grid on
   subplot(4,4,5)
      scatter(sepalLen,sepalWid,30,num,'.')
      ylabel('Sepal width')
      grid on
   subplot(4,4,9)
      scatter(sepalLen,petalLen,30,num,'.')
      ylabel('Petal length')
      grid on
   subplot(4,4,13)
      scatter(sepalLen,petalWid,30,num,'.')
      xlabel('Sepal length')
      ylabel('Petal Width')
      grid on
   % X = sepal wid
   subplot(4,4,6)
      scatter(sepalWid,sepalWid,30,num,'.')
      grid on
   subplot(4,4,10)
      scatter(sepalWid,petalLen,30,num,'.')
      grid on
   subplot(4,4,14)
      scatter(sepalWid,petalWid,30,num,'.')
      xlabel('Sepal width')
      grid on
   % X = petal len
   subplot(4,4,11)  
      scatter(petalLen,petalLen,30,num,'.')
      grid on
    subplot(4,4,15)  
      scatter(petalLen,petalWid,30,num,'.')
      xlabel('Petal length')
      grid on
   % X = petal wid
   subplot(4,4,16)
      scatter(petalWid,petalWid,30,num,'.')
      xlabel('Petal width')
      grid on
% Split the data set into train and test
trainSize = 125
Instances = size(class)
for i=1:Instances(1)
   Done = 0;
   while(Done == 0)
      chosen(i) = randi(Instances(1));
      Done = 1;
      for j=1:(size(chosen)-1)
         if(chosen(j) == chosen(i))
            Done = 0;
         end
      end
   end
end
for i=1:trainSize
   train(i,1) = sepalLen(chosen(i));
   train(i,2) = sepalWid(chosen(i));
   train(i,3) = petalLen(chosen(i));
   train(i,4) = petalWid(chosen(i));
   train(i,5) = num(chosen(i));        % Use numbers instead of string
end
for i=trainSize+1:Instances(1)
   test(i-trainSize,1) = sepalLen(chosen(i));
   test(i-trainSize,2) = sepalWid(chosen(i));
   test(i-trainSize,3) = petalLen(chosen(i));
   test(i-trainSize,4) = petalWid(chosen(i));
   test(i-trainSize,5) = num(chosen(i));
end
% Run the algorithm
for paraK=1:trainSize
   disp(paraK)
   score = 0;
   numTests = size(test(:,1));
   for i=1:numTests(1)
      for j=1:4   % Build vector
         testSub(j) = test(i,j);
      end
      for j=1:size(train)
         for k=1:4   % Build vector
            trainSub(k) = train(j,k);
         end
         train(j,6) = distance(testSub,trainSub);
      end
      total = 0;
      for j=1:paraK
         [value, index] = min(train(:,6));
         total = total + train(index,5);
         train(index,6) = max(train(:,6));
      end
      total = round(total./paraK);
      if(test(i,5) == total)
         score = score + 1;
      end
   end
   results(paraK,1) = paraK;
   results(paraK,2) = score./numTests(1);
end
results
figure(2)
scatter(results(:,1),results(:,2))
xlabel('K')
ylabel('Performace (%)')
grid on
