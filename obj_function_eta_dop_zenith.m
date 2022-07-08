function loss = obj_function_eta_dop_zenith(x,dop,zenith)
eta=x;
predict_dop = compute_d_dop(zenith,eta');
loss = sum((dop - predict_dop).^2,'all');


end