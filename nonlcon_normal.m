function [c,ceq] = nonlcon_normal(x)
c = 0;
if any(isnan(x(1:3)))
    ceq = 0;
else
    ceq = 1-sum(x(1:3).^2,'all');
end

end