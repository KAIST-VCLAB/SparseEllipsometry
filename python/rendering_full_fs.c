#version 130

in vec3 position_frag;
in vec3 rho_frag;
in vec3 normal_frag;
in float eta_frag;
in float m1_frag;
in float m2_frag;
in float ks1_frag;
in vec3 rho1_frag;

out vec4 outColor;


uniform mat4 opencv_intrinsic;
uniform mat4 opencv_extrinsic;
uniform unsigned int H;
uniform unsigned int W;
uniform vec3 lPos;
uniform vec3 vPos;
uniform vec3 y_l;
uniform vec3 y_v;
uniform vec3 color;
uniform vec4 light_stokes;
uniform vec4 view_stokes;
uniform vec3 select_rendering;
uniform unsigned int rendering_mode;


const float PI = 3.1415926535897932384626433832795;

void compute_Fresnel_coefficients_dielectric(in float cos_theta_i, in float cos_theta_o,
    in float n1, in float n2, out float Rs, out float Rp)
{

    Rs = (n1 * cos_theta_i - n2 * cos_theta_o ) / (n1 * cos_theta_i + n2 * cos_theta_o );
    Rp = (n1 * cos_theta_o - n2 * cos_theta_i ) / (n1 * cos_theta_o + n2 * cos_theta_i );
    Rp = Rp*Rp;
    Rs = Rs*Rs;

}

void refraction_cosangle_from_cos(in float cos_theta_i,in float n1, in float n2, out float cos_theta_o)
{
    if (cos_theta_i>=1){
        cos_theta_i = 1;
    }
    cos_theta_o = (n1/n2)*sqrt(1-cos_theta_i*cos_theta_i); // sin_theta_o
    cos_theta_o = sqrt(1-cos_theta_o*cos_theta_o);
}

void compute_D_GGX_a(in float m,in float cos_theta_h,in float tan_theta_h,out float D){
    D = m*m / (PI*pow(cos_theta_h,4)*pow((m*m + tan_theta_h*tan_theta_h),2));
    // sum of D should be one? not now.
}

void compute_G_smith_a(in float m, in float tan_theta_i, in float tan_theta_o, out float G){
    G = (2/(1 + sqrt(1 + (m*m) * (tan_theta_i*tan_theta_i)))) *
        (2/(1 + sqrt(1 + (m*m) * (tan_theta_o*tan_theta_o))));
    // Geometric factor
}

void coordinate_conversion_matrix(in float alpha, in float beta, out mat4 matrix)
{
    matrix = mat4
    (1, 0, 0, 0,
    0, beta, -alpha, 0,
    0, alpha, beta, 0,
    0, 0, 0, 1);
}


void main()
{
    vec3 normal;
    vec3 v;
    vec3 l;
    vec3 h;
    vec3 x_i,y_i,z_i,x_o,y_o,z_o;
    float ni;
    float no;
    float nh;
    float hi;
    float theta_h,theta_i,theta_o;
    float n2_hi;
    float D1,G1,D2,G2;
    float n2_ni,n2_no;
    float Ts_i,Tp_i,Ts_o,Tp_o,TpoTpi,TnoTpi,TpoTni,TnoTni;
    float Rs,Rp,Rpos,Rneg,Rcross,Spos,Sneg,Scross;
    float sin_azimuth, cos_azimuth, azimuth_denom;
    float alpha_i,beta_i,alpha_o,beta_o;
    float brewster_angle;
    float specular;
    float light_distance_sq;
    vec3 diffuse,single_scattering,specular_stokes;
    mat4 coord_conv_in,coord_conv_no,diffuse_fresnel,reflection_fresnel;
    mat4 opencv_cam;
    opencv_cam = opencv_intrinsic*opencv_extrinsic;
    vec4 camera_normal;
    vec4 output_stokes;
    vec3 newColor;

    if (eta_frag<=1){
        outColor =  vec4(0.0f,0.0f,0.0f,1.0f);
        return;
    }

    if (normal_frag[0]==0||normal_frag[1]==0||normal_frag[2]==0){
        outColor =  vec4(0.0f,0.0f,0.0f,1.0f);
        return;
    }

    v = vPos-position_frag;
    v = normalize(v);
    l = lPos-position_frag;
    light_distance_sq = dot(l,l);
    l = normalize(l);
    h = normalize(v+l);

    normal = normalize(normal_frag);

    no = dot(v,normal);
    ni = dot(l,normal);

    if (rendering_mode == 1u){
        camera_normal = opencv_extrinsic*vec4(normal,0.0f)*0.5+0.5;
        outColor = vec4(camera_normal[0],1-camera_normal[1],1-camera_normal[2],1.0f);
        return;
    }
    z_i = -l;
    y_i = y_l - dot(y_l,z_i)*z_i;
    y_i = normalize(y_i);
    x_i = cross(y_i,z_i);

    z_o = v;
    y_o = y_v - dot(y_v,z_o)*z_o;
    y_o = normalize(y_o);
    x_o = cross(y_o,z_o);

    if (ni<=0 || no<=0){
        outColor =  vec4(0.0f,0.0f,0.0f,1.0f);
        return;
    }
    hi = dot(l,h);
    nh = dot(normal,h);

    if (nh>1){
        nh=1;
    }
    if (ni>1){
        ni=1;
    }
    if (no>1){
        no=1;
    }
    if (hi>1){
        hi=1;
    }

    theta_h = acos(nh);
    theta_i = acos(ni);
    theta_o = acos(no);

    if (select_rendering[0]>0){
        sin_azimuth = dot(y_o,normal);
        cos_azimuth = dot(x_o,normal);
        azimuth_denom = inversesqrt(sin_azimuth*sin_azimuth+cos_azimuth*cos_azimuth);
        sin_azimuth = sin_azimuth*azimuth_denom;
        cos_azimuth = cos_azimuth*azimuth_denom;
        alpha_o = 2*sin_azimuth*cos_azimuth;
        beta_o = 2*cos_azimuth*cos_azimuth-1;

        sin_azimuth = dot(y_i,normal);
        cos_azimuth = dot(x_i,normal);
        azimuth_denom = inversesqrt(sin_azimuth*sin_azimuth+cos_azimuth*cos_azimuth);
        sin_azimuth = sin_azimuth*azimuth_denom;
        cos_azimuth = cos_azimuth*azimuth_denom;
        alpha_i = 2*sin_azimuth*cos_azimuth;
        beta_i = 2*cos_azimuth*cos_azimuth-1;

        refraction_cosangle_from_cos(ni,1,eta_frag,n2_ni);
        compute_Fresnel_coefficients_dielectric(ni, n2_ni, 1, eta_frag, Rs,Rp);
        Ts_i = 1-Rs;
        Tp_i = 1-Rp;

        refraction_cosangle_from_cos(no,1,eta_frag,n2_no);
        compute_Fresnel_coefficients_dielectric(n2_no,no, eta_frag, 1, Rs,Rp);
        Ts_o = 1-Rs;
        Tp_o = 1-Rp;

        TpoTpi = (Ts_i+Tp_i)*(Ts_o+Tp_o)/4;
        TnoTpi = (Ts_i+Tp_i)*(Ts_o-Tp_o)/4;
        TpoTni = (Ts_i-Tp_i)*(Ts_o+Tp_o)/4;
        TnoTni = (Ts_i-Tp_i)*(Ts_o-Tp_o)/4;

        if (select_rendering[0]>0){
            coordinate_conversion_matrix(-alpha_i,-beta_i,coord_conv_in);
            coordinate_conversion_matrix(alpha_o,-beta_o,coord_conv_no);

            diffuse_fresnel = mat4
            (TpoTpi,TnoTpi, 0,0,
             TpoTni,TnoTni,0,0,
             0,0,0,0,
             0,0,0,0);

            diffuse = max(ni,0)*rho_frag*dot(view_stokes,coord_conv_no*diffuse_fresnel*coord_conv_in*light_stokes);

        }
        else{
            diffuse = vec3(0.0f,0.0f,0.0f);
        }

    }
    else{
        diffuse = vec3(0.0f,0.0f,0.0f);
        single_scattering = vec3(0.0f,0.0f,0.0f);
    }

    if (select_rendering[1]>0 || select_rendering[2]>0){
        sin_azimuth = dot(y_o,h);
        cos_azimuth = dot(x_o,h);
        azimuth_denom = inversesqrt(sin_azimuth*sin_azimuth+cos_azimuth*cos_azimuth);
        sin_azimuth = sin_azimuth*azimuth_denom;
        cos_azimuth = cos_azimuth*azimuth_denom;
        alpha_o = 2*sin_azimuth*cos_azimuth;
        beta_o = 2*cos_azimuth*cos_azimuth-1;

        sin_azimuth = dot(y_i,h);
        cos_azimuth = dot(x_i,h);
        azimuth_denom = inversesqrt(sin_azimuth*sin_azimuth+cos_azimuth*cos_azimuth);
        sin_azimuth = sin_azimuth*azimuth_denom;
        cos_azimuth = cos_azimuth*azimuth_denom;
        alpha_i = 2*sin_azimuth*cos_azimuth;
        beta_i = 2*cos_azimuth*cos_azimuth-1;
        if (select_rendering[1]>0){
            compute_D_GGX_a(m1_frag,nh,tan(theta_h),D1);
            compute_G_smith_a(m1_frag, tan(theta_i), tan(theta_o),G1);
        }
        else{
            D1 = 0;
            G1 = 0;
        }
        if (select_rendering[2]>0){
            compute_D_GGX_a(m2_frag,nh,tan(theta_h),D2);
            compute_G_smith_a(m2_frag, tan(theta_i), tan(theta_o),G2);
        }
        else{
            D2 = 0;
            G2 = 0;
        }


        refraction_cosangle_from_cos(hi,1,eta_frag,n2_hi);
        compute_Fresnel_coefficients_dielectric(hi, n2_hi, 1, eta_frag, Rs,Rp);

        Rpos = (Rs+Rp)/2;
        Rneg = (Rs-Rp)/2;
        Rcross = sqrt(Rs*Rp);

        brewster_angle = atan(eta_frag);
        if (acos(hi)<brewster_angle){
            Rcross = -Rcross;
        }

        coordinate_conversion_matrix(-alpha_i,-beta_i,coord_conv_in);
        coordinate_conversion_matrix(alpha_o,-beta_o,coord_conv_no);


        reflection_fresnel = mat4
        (Rpos,Rneg, 0,0,
         Rneg,Rpos,0,0,
         0,0,Rcross,0,
         0,0,0,Rcross);



        specular = ks1_frag*D1*G1*dot(view_stokes,coord_conv_no*reflection_fresnel*coord_conv_in*light_stokes)/4/no;

        single_scattering = rho1_frag*D2*G2*dot(view_stokes,coord_conv_no*reflection_fresnel*coord_conv_in*light_stokes)/4/no;

    }
    else{
        specular = 0.0f;
        single_scattering = vec3(0.0f,0.0f,0.0f);
    }



    newColor = select_rendering[0]*diffuse + select_rendering[1]*specular + select_rendering[2]*single_scattering;
    newColor = newColor/light_distance_sq;

    newColor = pow(newColor,vec3(1/2.2));
    outColor = vec4(newColor, 1.0f);
}

