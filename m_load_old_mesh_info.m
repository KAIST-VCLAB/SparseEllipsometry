
[Idx,hausdorff_dist] = knnsearch(meshCurrent_old.vertices',meshCurrent.vertices');
meshCurrent.eta = meshCurrent_old.eta(:,Idx);
meshCurrent.m1 = meshCurrent_old.m1(:,Idx);
meshCurrent.m2 = meshCurrent_old.m2(:,Idx);
meshCurrent.ks1 = meshCurrent_old.ks1(:,Idx);
meshCurrent.ks2 = meshCurrent_old.ks2(:,Idx);
meshCurrent.rho = meshCurrent_old.rho(:,Idx);
meshCurrent.new_normal = meshCurrent_old.new_normal(:,Idx);