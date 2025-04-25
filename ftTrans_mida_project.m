function [ftNewMapped] = ftTrans_mida_project(ftNew, domainFtNew, ftUsed, domainFtUsed, projectionMatrix, param)
%   This is an adaptation of the function ftTrans_mida which takes in a
%   projection matrix that was estimated using the ftTrans_mida function
%   and applies it to a new set of data, mapping it to the same space as
%   the samples the projection matrix was estimated on.

% ftNew: New samples in all domains to be mapped. n-by-m matrix, n 
%        is the number of samples, m is the dimension of features.
% domainFtNew: Domain features of new samples. n-by-md matrix, md is the 
%	           dimension of domain features.
% ftUsed: Samples that were originally mapped to the new space. 
% domainFtUsed: Domain features of original samples. n-by-md matrix, md 
%               is the dimension of domain features.
% projectionMatrix: Projection matrix that was used to map the original
%                   samples, as given out by ftTrans_mida
% param: Struct of hyper-parameters. Here, we only need the parameters 
%        kerName and ftAugtype. These need to be the same as in the
%        original mapping.
                                            
% ftNewMapped:	New samples in the learned subspace.

% ref: Ke Yan, Lu Kou, and David Zhang, "Domain Adaptation via Maximum 
%	Independence of Domain Features," http://arxiv.org/abs/1603.04535
% Copyright 2016 Ke YAN, Tsinghua Univ. http://yanke23.com , xjed09@gmail.com

% Adapted by Rico Stecher (Neural Computation Lab, JLU Gießen), 2025

%% default parameters
kerName = 'lin'; % kernel name, see the next cell ("kernels")
ftAugType = 1; % feature augmentation, 0: no aug; 1: aug with domainFt; 
	% 2: frustratingly easy aug, only for discrete domains, see the ref

defParam % set user-defined hyper-parameters

%% kernels
nm = @(X,p)repmat(sum(X.^2,2),1,p);
linKer = @(X1,X2)X1*X2';
rbfKer = @(X1,X2)exp(-(nm(X1,size(X2,1))+nm(X2,size(X1,1))'-2*X1*X2')/2/kerSigma^2);
lapKer = @(X1,X2)exp(-pdist2(X1,X2)/kerSigma);
polyKer = @(X1,X2)(1+kerSigma*X1*X2').^2;
if strcmpi(kerName,'lin'), kerFun = linKer; % linear kernel
elseif strcmpi(kerName,'poly'), kerFun = polyKer; % polynomial kernel
elseif strcmpi(kerName,'rbf'), kerFun = rbfKer;
elseif strcmpi(kerName,'lap'), kerFun = lapKer; % Laplacian kernel
else error('unknown kernel'); end

%% project the samples

% Assuming ftNew (nNew-by-m) and domainFtNew (nNew-by-md) are given

% Feature augmentation (match ftAugType used in ftTrans_mida)
if ftAugType == 0
    ftNewAug = ftNew;
    ftUsedAug = ftUsed;
elseif ftAugType == 1
    ftNewAug = [ftNew, domainFtNew];
    ftUsedAug = [ftUsed, domainFtUsed];
elseif ftAugType == 2
    [nNew, nFt] = size(ftNew);
    nUsed = size(ftUsed,1);
    nDomain = size(domainFtNew, 2);
    ftNewAug = zeros(nNew, nFt * (nDomain + 1));
    ftUsedAug = zeros(nUsed, nFt * (nDomain + 1));
    ftNewAug(:, 1:nFt) = ftNew;
    ftUsedAug(:, 1:nFt) = ftUsed;
    for p = 1:nDomain
        ftNewAug(domainFtNew(:, p) == 1, nFt * p + 1 : nFt * (p + 1)) = ftNew(domainFtNew(:, p) == 1, :);
        ftUsedAug(domainFtUsed(:, p) == 1, nFt * p + 1 : nFt * (p + 1)) = ftUsed(domainFtUsed(:, p) == 1, :);
    end
end

% Compute kernel matrix between new data and the used training data
KxNew = kerFun(ftNewAug, ftUsedAug);

% Project new data into domain-invariant space
ftNewMapped = KxNew * projectionMatrix;

end
