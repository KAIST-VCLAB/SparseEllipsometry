#version 130
in vec3 position;
in vec3 normal;
in vec3 rho;
in float eta;
in float m1;
in float m2;
in float ks1;
in vec3 rho1;
out vec3 position_frag;
out vec3 rho_frag;
out vec3 normal_frag;
out float eta_frag;
out float m1_frag;
out float m2_frag;
out float ks1_frag;
out vec3 rho1_frag;

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
    // Geoemtric factor
}

void coordinate_conversion_matrix(in float alpha, in float beta, out mat4 matrix)
{
    matrix = mat4
    (1, 0, 0, 0,
    0, beta, -alpha, 0,
    0, alpha, beta, 0,
    0, 0, 0, 1);
}

vec3 hsv2rgb(vec3 c)
{
//https://stackoverflow.com/questions/15095909/from-rgb-to-hsv-in-opengl-glsl
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main()
{
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

    gl_Position = opencv_cam*vec4(position,1.0f);
    gl_Position[0] = 2*(gl_Position[0]/gl_Position[2])/W-1;
    gl_Position[1] = -2*(gl_Position[1]/gl_Position[2])/H+1;
    gl_Position[2] = gl_Position[2]/65536;

    position_frag = position;
    rho_frag = rho;
    normal_frag = normal;
    eta_frag = eta;
    m1_frag = m1;
    m2_frag = m2;
    ks1_frag = ks1;
    rho1_frag = rho1;
}
