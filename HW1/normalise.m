function[xnorm] = normalise(x)
%NORMALISE Normalise a vector.
%   Normalise(X) normalises the vector X to the range -1 to +1.
%   
%   This  function is meant to be used for a 1 x n array.
%
%   Based on the normalization function provided by Steve Lord at 
%   <http://www.mathworks.co.uk/matlabcentral/newsreader/view_thread/162772>

    % Find the minimum and maximum values
    xmin = min(x(:));
    xmax = max(x(:));
    
    if(abs(xmax) >= abs(xmin))
        
        maxamplitude = abs(xmax);
        
    else
        
        maxamplitude = abs(xmin);
        
    end
    
    if(xmin == xmax)
        % Constant matrix -- I choose to warn and return a NaN matrix
        warning('normalization:constantMatrix', 'Cannot normalise a constant matrix to the range [-1, 1].');
        xnorm = nan(size(x));
    
    else
        
        xnorm = x / maxamplitude;
        
    end

end

