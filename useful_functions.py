# Functions

import numpy as np
import csv
import cmath

# Efficiencies of a feed, using farfield directivity


def compare_farfields(file1,file2,d_phi_degree,d_theta_degree):
    theta = []
    phi = []
    gains1 = []
    gains2 = []
    with open(file1) as f:
        data = f.readlines()[2:]
        for line in data:
            theta.append( float(line.split()[0]) )
            phi.append( float(line.split()[1]) )
            gains1.append( float(line.split()[2]) )
    with open(file2) as f:
        data = f.readlines()[2:]
        for line in data:
             gains2.append( float(line.split()[2]) )
    
    n=len(theta)
    d_phi=np.pi*d_phi_degree/180
    d_theta=np.pi*d_theta_degree/180
    
    total_diff=0
    total_gain=0
    
    for i_data in range(n):
        d_area=np.abs(np.sin(theta[i_data]*np.pi/180) * d_theta * d_phi)
        local_diff = np.abs(gains1[i_data]-gains2[i_data])*d_area
        local_gain = gains1[i_data]*d_area
        total_diff+=local_diff
        total_gain+=local_gain
        
    return total_diff/total_gain
    
            
def power_within_angle(filename,theta_crit,d_phi_degree,d_theta_degree):
    theta = []
    phi = []
    gains = []
    with open(filename) as f:
        data = f.readlines()[2:]
        for line in data:
            theta.append( float(line.split()[0]) )
            phi.append( float(line.split()[1]) )
            gains.append( float(line.split()[2]) )
            
            

    n = len(theta)
    d_phi=np.pi*d_phi_degree/180
    d_theta=np.pi*d_theta_degree/180
    
    front_gain = 0;
    total_gain = 0;

    
    
    # Integrate
    for i_data in range(n):
        d_area=np.abs(np.sin(theta[i_data]*np.pi/180) * d_theta * d_phi);
        local_gain = gains[i_data]*d_area
        if local_gain<0:
            print('error')
        total_gain = total_gain + local_gain
        if abs(abs(theta[i_data])-180)<=theta_crit:
            front_gain = front_gain + local_gain;


    return front_gain/total_gain;

    
    
    


def efficiencies(filename,f_over_D,d_phi_degree,d_theta_degree):
    # Determine angle
    theta_crit=np.arctan2(1,2*(f_over_D-1/(16*f_over_D)))*(180/np.pi)
    # Import data and make usable arrays
    # Data header: Theta [deg.]  Phi   [deg.]  Abs(Gain)[dB    ] ...
    # filename = ('farfield (f=0.4) [1].txt');
    theta = []
    phi = []
    gains = []
    with open(filename) as f:
        data = f.readlines()[2:]
        for line in data:
            theta.append( float(line.split()[0]) )
            phi.append( float(line.split()[1]) )
            gains.append( float(line.split()[2]) )
            
            

    n = len(theta)
    d_phi=np.pi*d_phi_degree/180
    d_theta=np.pi*d_theta_degree/180
    
    
    ## Spillover

    front_gain = 0;
    total_gain = 0;

    
    
    # Integrate
    for i_data in range(n):
        d_area=np.abs(np.sin(theta[i_data]*np.pi/180) * d_theta * d_phi);
        local_gain = gains[i_data]*d_area
        if local_gain<0:
            print('error')
        total_gain = total_gain + local_gain
        if theta[i_data]<=theta_crit:
            front_gain = front_gain + local_gain;


    s_eff = front_gain/total_gain; # spillover efficiency


    ## Illumination

    numerator = 0
    denominator = 0
    area = 0
    
    for i_data in range(n):
        
        d_area=np.abs(np.sin(theta[i_data]*np.pi/180)*np.cos(theta[i_data]*np.pi/180) * d_theta * d_phi)
        if theta[i_data]<=theta_crit:
            numerator=numerator + (gains[i_data]**0.5)*d_area
            denominator=denominator + gains[i_data]*d_area
            area = area + d_area;

    i_eff = (numerator**2)/(denominator*area)

    eff = s_eff*i_eff


    
    return np.array([i_eff, s_eff, eff])


def integrate_gains(filename,theta_crit,d_phi_degree,d_theta_degree,theta_inc,phi_inc):

    theta = []
    phi = []
    gains = []

    with open(filename) as f:
        data = f.readlines()[2:]
        for line in data:
            theta.append( float(line.split()[0]) )
            phi.append( float(line.split()[1]) )
            gains.append( float(line.split()[2]) )

    n = len(theta)
    d_phi=np.pi*d_phi_degree/180
    d_theta=np.pi*d_theta_degree/180
    
    integrated_gain=0
    area=0
    
    for i_data in range(n):
        
        
        if (np.mod(phi[i_data]+phi_inc,360)<=180 and theta[i_data]>=180-(theta_crit+theta_inc)) or (np.mod(phi[i_data]+phi_inc,360)>180 and theta[i_data]>=180-(theta_crit-theta_inc)):
            d_area=np.abs(np.sin(theta[i_data]*np.pi/180) * d_theta * d_phi)
            integrated_gain=integrated_gain + (gains[i_data])*d_area
            area = area + d_area;

    
    return integrated_gain#/area

def wide_beam_efficiency(filename,theta_wide,theta_narrow,d_phi_degree,d_theta_degree):

    theta = []
    phi = []
    gains = []

    with open(filename) as f:
        data = f.readlines()[2:]
        for line in data:
            theta.append( float(line.split()[0]) )
            phi.append( float(line.split()[1]) )
            gains.append( 1 )
            #gains.append( float(line.split()[2]) )

    n = len(theta)
    d_phi=np.pi*d_phi_degree/180
    d_theta=np.pi*d_theta_degree/180
    
    integrated_gain=0
    area=0
    
    for i_data in range(n):
        
        d_area=np.abs(np.sin(theta[i_data]*np.pi/180) * d_theta * d_phi)
        if theta[i_data]>180-theta_wide:
            integrated_gain=integrated_gain + (gains[i_data])*d_area
        if theta[i_data]>180-theta_narrow:
            area = area + d_area;

    efficiency = integrated_gain/area
    
    return efficiency#/area

def get_max_gain(filename):

    
    with open(filename) as f:
        data = f.readlines()[2:]


    max_gain = 0
    with open(filename) as f:
        data = f.readlines()[2:]
        for line in data:
            if float(line.split()[2])>max_gain:
                max_gain=float(line.split()[2])

   
    return max_gain

def get_const_phi_cut(filename,phi):
    theta = []
    gains = []
    gains_to_flip = []

    with open(filename) as f:
        data = f.readlines()[2:]
        for line in data:
            if float(line.split()[1])==phi:
                theta.append( float(line.split()[0]) )
                gains.append( 10*np.log10(float(line.split()[2])) )
            elif np.mod(float(line.split()[1])+180,360)==phi:
                theta.append( np.mod(float(line.split()[0])+180,360))
                gains_to_flip.append( 10*np.log10(float(line.split()[2])) )
    

    gains=gains+gains_to_flip[::-1]

    cut=np.array([theta,gains])
    
    return cut

def count_lines(filename,line_skip):
    n_lines=0
    with open(filename) as f:
        data = f.readlines()[line_skip:]
        for line in data:
            n_lines = n_lines + 1
    return n_lines
    

def get_e_fields(filename):
    
    n_lines=count_lines(filename,2)
    
    e_fields=np.zeros([n_lines,9])
    
    with open(filename) as f:
        i_line=0
        data = f.readlines()[2:]
        for line in data:
            for i_column in range(9):
                e_fields[i_line,i_column]=( float(line.split()[i_column]) )
            i_line=i_line+1

    
    return e_fields

def get_e_fields_on_surface(e_fields,R,Delta_R):
    e_fields_on_surface = []
    for i in range(e_fields.shape[0]):
        r=(e_fields[i,0]**2 + e_fields[i,1]**2 + (e_fields[i,2])**2)**0.5
        if np.abs(r-R)<Delta_R:# and e_fields[i,2]>0:
            e_fields_on_surface.append([e_fields[i,0],e_fields[i,1],e_fields[i,2],complex(e_fields[i,3],e_fields[i,4]),complex(e_fields[i,5],e_fields[i,6]),complex(e_fields[i,7],e_fields[i,8])])
                
    return np.transpose(e_fields_on_surface)


def freq_to_wavelength(frequency):
    wavelength = (3e8)/frequency
    return wavelength

def find_closest(phi,theta,R,e_fields_on_surface):
    # Determine x,y,z
    x=R*np.sin(theta)*np.cos(phi)
    y=R*np.sin(theta)*np.sin(phi)
    z=R*np.cos(theta)
    min_distance=1e9
    index=0
    for i_point in range(e_fields_on_surface.shape[1]):
        distance = ( (e_fields_on_surface[0,i_point]-x)**2 + (e_fields_on_surface[1,i_point]-y)**2 + (e_fields_on_surface[2,i_point]-z)**2 )**0.5
        if distance<min_distance:
            min_distance=distance
            index=i_point
    return index
    
    #for i in range(e_fields_on_surface.shape[1])
    #    dist=((x-e_fields_on_surface[0,i])**2 + (y-e_fields_on_surface[1,i])**2 + (z-e_fields_on_surface[2,i])**2 )**0.5
    

def get_differential_areas(e_fields,R,Delta_R):
    e_fields_on_surface = get_e_fields_on_surface(e_fields,R,Delta_R)
    print('Points to process:', e_fields_on_surface.shape[1])
    print('Points processed: 0... ', end='')
    differential_areas=np.zeros(e_fields_on_surface.shape[1])
    for i_point in range(e_fields_on_surface.shape[1]):
        neighbourhood_angle = find_max_neighbourhood_angle(e_fields_on_surface[0:3,i_point],e_fields_on_surface)
        #print(neighbourhood_angle)
        d_area=(2*(R/1000)*(neighbourhood_angle))**2
        differential_areas[i_point]=d_area
        if i_point%100==0 and i_point>0:
            print(i_point,end='')
            print('... ',end='')
    print(' done, returning the d_areas.')
    return differential_areas

def find_max_neighbourhood_angle(coordinates,e_fields_on_surface):
    min_angle=1e9
    real_min_angle=1e9
    
    theta=np.arctan2( np.real((coordinates[0]**2 + coordinates[1]**2 )**0.5) , np.real(coordinates[2]))
    phi=np.arctan2( np.real(coordinates[0]),np.real(coordinates[1]))
    
    
    for i_point in range(e_fields_on_surface.shape[1]):
        local_theta=np.arctan2( np.real((e_fields_on_surface[0,i_point]**2 + e_fields_on_surface[1,i_point]**2 )**0.5 ), np.real(e_fields_on_surface[2,i_point]))
        local_phi=np.arctan2( np.real(e_fields_on_surface[0,i_point]),np.real(e_fields_on_surface[1,i_point]))
        delta_phi=min( np.mod(phi-local_phi,2*np.pi),np.mod(local_phi-phi,2*np.pi) )
        delta_theta=min( np.mod(theta-local_theta,2*np.pi),np.mod(local_theta-theta,2*np.pi) )
        
        angle_distance=delta_phi**2 + delta_theta**2
        #angle_distance=((delta_phi**2)+(delta_theta**2))**0.5
        if angle_distance>0 and angle_distance<min_angle:
            min_angle=angle_distance
            argument=np.cos(local_theta)*np.cos(theta)+np.sin(local_theta)*np.sin(theta)*np.cos(delta_phi)
            if abs(argument)>1:
                argument*=0.999
            real_min_angle=np.arccos( argument )
    if real_min_angle==1e9:
        print('Error, didn\'t find a minimum distance.')
    return real_min_angle

def integrate_on_surface(R,Delta_R,e_fields,differential_areas):
    
    e_fields_on_surface = get_e_fields_on_surface(e_fields,R,Delta_R)
    
    front_area=0+0j
    total_area=0+0j
    phasor_x=0+0j
    phasor_y=0+0j
    phasor_z=0+0j
    mag_x=0+0j
    mag_y=0+0j
    mag_z=0+0j
    mag_abs=0+0j
    front_mag_abs=0+0j
    
    for i_point in range(e_fields_on_surface.shape[1]):
        d_area=np.abs(differential_areas[i_point])
        if e_fields_on_surface[2,i_point]>=0:
            phasor_x+=e_fields_on_surface[3,i_point]*d_area
            phasor_y+=e_fields_on_surface[4,i_point]*d_area
            phasor_z+=e_fields_on_surface[5,i_point]*d_area
            front_mag_abs+=( (np.abs(e_fields_on_surface[3,i_point])**2 + np.abs(e_fields_on_surface[4,i_point])**2 + np.abs(e_fields_on_surface[5,i_point])**2) )*d_area
            front_area+=d_area
        mag_x+=np.abs(e_fields_on_surface[3,i_point])*d_area
        mag_y+=np.abs(e_fields_on_surface[4,i_point])*d_area
        mag_z+=np.abs(e_fields_on_surface[5,i_point])*d_area
        mag_abs+=( (np.abs(e_fields_on_surface[3,i_point])**2 + np.abs(e_fields_on_surface[4,i_point])**2 + np.abs(e_fields_on_surface[5,i_point])**2) )*d_area
        total_area+=d_area
    
    front_phasor_mag=(np.abs(phasor_x)**2+np.abs(phasor_y)**2+np.abs(phasor_z)**2)
    return np.array([front_phasor_mag,front_mag_abs,front_area,mag_abs])

def get_txt(filename):
    with open(filename) as f:
        data = f.readlines()[2:]
        freq=[]
        value=[]   
        for line in data:
            if line.isspace():
                break
            freq.append( 1000*float(line.split()[0]) )
            value.append( float(line.split()[1]) )
    return np.array([freq,value])
    
    
def get_loss(filename):
    with open(filename) as f:
        data = f.readlines()[2:]
        freq=[]
        loss=[]   
        for line in data:
            if line.isspace():
                break
            freq.append( 1000*float(line.split()[0]) )
            loss.append( float(line.split()[1]) )
    return np.array([freq,loss])

def get_const_phi_cut(filename,phi):
    theta = []
    gains = []
    gains_to_flip = []

    with open(filename) as f:
        data = f.readlines()[2:]
        for line in data:
            if float(line.split()[1])==phi:
                theta.append( float(line.split()[0]) )
                gains.append( float(line.split()[2]) )
            elif np.mod(float(line.split()[1])+180,360)==phi:
                theta.append( np.mod(float(line.split()[0])+180,360))
                gains_to_flip.append( float(line.split()[2]) )
    

    gains=gains+gains_to_flip[::-1]

    cut=np.array([theta,gains])
    
    return cut