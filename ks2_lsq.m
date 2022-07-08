function ks2 = ks2_lsq(D2,G2,DG1,Rpos,theta_o,psi,weight,S)

DG2= weight.*(D2.*G2);
A = [DG2;zeros(3,size(DG2,2));DG2;zeros(3,size(DG2,2));DG2];
A = reshape(A,3,[])';
% A=[repmat(DG1',3,1),A];
% b = S.*psi./Rpos.*4.*cos(theta_o);
b = weight.*(S.*psi./Rpos.*4.*cos(theta_o)-DG1);
b = reshape(b,[],1);
x_est = A\b;
x_est(x_est<0)=0;
% ks1 = x_est(1);
% ks2 = x_est(2:4);
ks2 = x_est(1:3);
end