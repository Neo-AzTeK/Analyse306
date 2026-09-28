import numpy as np
import matplotlib.pyplot as plt

def compute_response(loading_angle,parameters,scale=2.0, npoints=10000):
    #-------------------------------------------------------------------------------------------------------------------
    # input parameters
    #-------------------------------------------------------------------------------------------------------------------
    f_n_max, xi_n, d_n_fail = parameters['f_n_max'], parameters['xi_n'], parameters['d_n_fail']
    f_t_max, xi_t, d_t_fail = parameters['f_t_max'], parameters['xi_t'], parameters['d_t_fail']
    alpha_1, alpha_2 ,alpha_3 = parameters['alpha_1'], parameters['alpha_2'], parameters['alpha_3']
    if 'h_ms' in parameters.keys():
       h_ms = parameters['h_ms']
    else:   
       h_ms = 1.0
    #-------------------------------------------------------------------------------------------------------------------
    # load steps
    #-------------------------------------------------------------------------------------------------------------------
    d_n = np.linspace(0.0,scale*d_n_fail,npoints)*np.sin(np.deg2rad(loading_angle))
    d_t = np.linspace(0.0,scale*d_t_fail,npoints)*np.cos(np.deg2rad(loading_angle))
    #-------------------------------------------------------------------------------------------------------------------
    # running variables
    #-------------------------------------------------------------------------------------------------------------------
    f_n_hat = np.zeros(len(d_n))
    f_t_hat = np.zeros(len(d_n))
    f_n     = np.zeros(len(d_n))
    f_t     = np.zeros(len(d_n))
    M_m     = np.zeros(len(d_n))
    M_s     = np.zeros(len(d_n))
    teta    = np.zeros(len(d_n))
    xi      = np.zeros(len(d_n))
    eta     = np.zeros(len(d_n))
    alpha   = np.zeros(len(d_n))
    eta_max = np.zeros(len(d_n))
    eta_max_val = 0.0
    #
    for i in range(1,npoints):
        # calc teta
        teta[i] = np.arctan(d_n[i]/d_t[i])
        # calc xi
        xi[i] = 1.-(27./4.)*(2.*teta[i]/np.pi)**2. + (27./4.)*(2.*teta[i]/np.pi)**3.
        # calc alpha
        alpha[i] = alpha_1*(xi_t-eta_max_val)/xi_t+alpha_2*eta_max_val/xi_t if eta_max_val<xi_t else alpha_2*(1.-eta_max_val)/(1.-xi_t) + alpha_3*(eta_max_val-xi_t)/(1.-xi_t)
        eta[i] = np.sqrt( (d_n[i]/d_n_fail)**2.+(d_t[i]/d_t_fail)**2. )
        eta[i] = (xi[i]+((1.-xi[i])/alpha[i])) * eta[i]
        
        eta_max[i]  = np.nanmax((np.nanmax(eta_max),eta[i]))
        eta_max_val = np.nanmax((np.nanmax(eta_max),eta[i]))
        n_max = i
        if eta_max_val >= 0.9999:       
           break 
        # calc forces
        f_n_hat[i] = 1.-((xi_n-eta_max_val)/xi_n)**8 if eta_max_val<=xi_n else 1.-((eta_max_val-xi_n)/(1.-xi_n))
        f_t_hat[i] = 1.-((xi_t-eta_max_val)/xi_t)**8 if eta_max_val<=xi_t else 1.-((eta_max_val-xi_t)/(1.-xi_t))       
        f_n[i] = f_n_max * f_n_hat[i] * d_n[i] / (eta_max_val*d_n_fail)
        f_t[i] = f_t_max * f_t_hat[i] * d_t[i] / (eta_max_val*d_t_fail)
        
        # calc moments
        M_m[i] = h_ms/4.*f_t[i] if eta_max_val<xi_t else h_ms/4.*(1.+((eta_max_val-xi_t)/(1.-xi_t)))
        M_s[i] = h_ms/4.*f_t[i] if eta_max_val<xi_t else h_ms/4.*(1.-((eta_max_val-xi_t)/(1.-xi_t)))
    #
    delta = np.sqrt(d_n*d_n+d_t*d_t)
    force = np.sqrt(f_n*f_n+f_t*f_t)
    results = {'d_n':d_n[0:n_max],'f_n':f_n[0:n_max],'f_n_hat':f_n_hat[0:n_max],
               'd_t':d_t[0:n_max],'f_t':f_t[0:n_max],'f_t_hat':f_t_hat[0:n_max],
               'M_m':M_m[0:n_max],'M_s':M_s[0:n_max],
               'eta': eta_max[0:n_max],'eta_max_val':eta_max_val,'n_max':n_max,
               'delta':delta[0:n_max],'force':force[0:n_max]}
    return results

if __name__ == '__main__':
   
   parameters = {
                 'f_n_max':3650.0, 'xi_n':0.8, 'd_n_fail':4.0,
                 'f_t_max':7200.0, 'xi_t':0.8, 'd_t_fail':7.0,
                 'alpha_1':0.4, 'alpha_2':0.3, 'alpha_3':1.5,
                 'h_ms':4.0
                 } 
       
   loading_angle = 45.0
   results = compute_response(loading_angle,parameters)
   
   fig,axs = plt.subplots(1,3,figsize=(13,4))
   axs[0].plot(results['d_n'], results['f_n'],label = 'Tension')
   axs[0].plot(results['d_t'], results['f_t'],label = 'Shear')
   axs[0].grid()
   #axs[0].set_xlim([0.0,np.maximum(scale*d_n_fail,scale*d_t_fail)])
   axs[0].set_xlabel('Displacement (mm)')
   axs[0].set_ylabel('Force (N)')
   axs[0].legend()
   #
   axs[1].plot(results['delta'], results['eta'])
   axs[1].set_xlabel(r'Resultant displacement (mm)')
   axs[1].set_ylabel(r'Effective displacement $\eta$ (-)')
   axs[1].grid()
   #axs[1].set_xlim([0.0,1.0])
   #
   axs[2].plot(results['delta'], results['force'],label = 'Resultant')
   axs[2].set_xlabel(r'Resultant displacement (mm)')
   axs[2].set_ylabel('Resultant force (N)')
   #axs[2].set_xlim([0.0,np.max(delta[0:n_max])])
   axs[2].grid()
   axs[2].legend()
   plt.tight_layout()
   plt.savefig('MATER_306_response.pdf')
   plt.show()









