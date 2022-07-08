function [c,ceq] = nonlcon_init_normal(x,initial_normal)
normal = x(1:3);
normal = normal./sqrt(sum(normal(:).^2));
c = cos(deg2rad(60))-dot(normal,initial_normal);
if any(isnan(x(1:3)))
    ceq = 0;
else
    ceq = 1-sum(x(1:3).^2,'all');
end

end