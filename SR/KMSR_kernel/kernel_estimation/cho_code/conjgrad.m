function x=conjgrad(x,b,maxIt,tol,Ax_func,func_param,visfunc)
% conjgrad.m
%
%   Conjugate gradient optimization
%
%     written by Sunghyun Cho (sodomau@postech.ac.kr)
%
    r = b - Ax_func(x,func_param);
    p = r;
    rsold = sum(r(:).*r(:));

    for iter=1:maxIt
        Ap = Ax_func(p,func_param);
        alpha = rsold/sum(p(:).*Ap(:));
        x=x+alpha*p;
        if exist('visfunc', 'var')
            visfunc(x, iter, func_param);
        end
        r=r-alpha*Ap;
        rsnew=sum(r(:).*r(:));
        if sqrt(rsnew)<tol
            break;
        end
        p=r+rsnew/rsold*p;
        rsold=rsnew;
    end
end
