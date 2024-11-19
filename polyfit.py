import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from scipy.fft import fft, fftshift

import pywt

from geneticalgorithm import geneticalgorithm as ga

##### Author: Gilberto Rodriguez Prado

#####Need to add the find peaks function, optimization problems, and add comments 

def concatenate_indices(h, k, l):
    # Ensure the input arrays are numpy arrays
    h = np.array(h)
    k = np.array(k)
    l = np.array(l)
    
    # Zip the arrays together and format them
    formatted_indices = [f"({i}{j}{k})" for i, j, k in zip(h, k, l)]
    
    return formatted_indices

def read_xy_file(file_path):
    x_data = []
    y_data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Attempt to convert and parse the line
                x, y = map(float, line.strip().split())
                x_data.append(x)
                y_data.append(y)
            except ValueError:
                # If conversion fails, skip the line
                print(f"Skipping line: {line.strip()}")

    y_data = y_data/np.max(y_data)
    return np.array(x_data), np.array(y_data)

def read_xy_file2(file_path):
    x_data = []
    y_data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                x, y, _ = map(float, line.strip().split())  # Ignore the third value
                x_data.append(x)
                y_data.append(y)
            except ValueError:
                print(f"Skipping line: {line.strip()}")

    y_data = y_data/np.max(y_data)
    return np.array(x_data), np.array(y_data)

def BKG(theta,P2,Q,R):
    return P2*theta**2 + Q*theta + R

def lorentzian_function(H,x, disp):
    return 1/(2*np.pi) * (H / ((H/2)**2 + (x-disp)**2))

def gaussian_function(x, H, disp):
    return 1/H * (np.sqrt((4 * np.log(2)) / np.pi)) * np.exp(-4 * np.log(2) * ((x-disp)**2) / H**2)

def pseudoVoigt_function(A,H,eta,disp,x):
    return A*(eta*lorentzian_function(H,x,disp) + (1-eta)*gaussian_function(x, H,disp))

def Scherrer(theta,H):
    #Use radian and H, lambda_Cu, H2 is computed automatically
    lamb = 0.1540656 #nm
    return 2 * np.sqrt(np.log(2)/np.pi) * lamb/(H*np.cos(theta/2))

class FitnessFunctions:

    @staticmethod
    def model2(x, params,A_array):
        # Sample model function implementation
        # Unpack background parameters
        P2, Q, R = params[0], params[1], params[2]
        
        # Start constructing the total Voigt profile
        total = np.zeros_like(x)
        
        # Example processing; adapt as necessary
        param_index = 3
        for _ in range(len(A_array)):
            H = params[param_index]
            eta = params[param_index + 1]
            A = params[param_index + 2]
            disp = params[param_index + 3]
            total += pseudoVoigt_function(A, H, eta, disp, x)
            param_index += 4
        
        # Add background to the total profile
        total += BKG(x, P2, Q, R)
        return total

    @staticmethod
    def rwp_fitness_function(params, x, y, A_array):
        y_pred = FitnessFunctions.model2(x, params, A_array)
        return np.sqrt(np.sum((y - y_pred)**2) / np.sum(y**2))

    @staticmethod
    def huber_fitness_function(params, x, y, A_array, delta=2.0):
        y_pred = FitnessFunctions.model2(x, params, A_array)
        residual = np.abs(y - y_pred)
        return np.where(residual <= delta, 0.5 * residual ** 2, delta * (residual - 0.5 * delta)).mean()

    @staticmethod
    def rmse_fitness_function(params, x, y, A_array):
        y_pred = FitnessFunctions.model2(x, params, A_array)
        return np.sqrt(np.mean((y - y_pred) ** 2))

    @staticmethod
    def mse_fitness_function(params, x, y, A_array):
        y_pred = FitnessFunctions.model2(x, params, A_array)
        return np.mean((y - y_pred) ** 2)

    @staticmethod
    def mae_fitness_function(params, x, y, A_array):
        y_pred = FitnessFunctions.model2(x, params, A_array)
        return np.mean(np.abs(y - y_pred))

    @staticmethod
    def msle_fitness_function(params, x, y, A_array):
        y_pred = FitnessFunctions.model2(x, params, A_array)
        return np.mean((np.log1p(y) - np.log1p(y_pred)) ** 2)

    @staticmethod
    def logcosh_fitness_function(params, x, y, A_array):
        y_pred = FitnessFunctions.model2(x, params, A_array)
        return np.sum(np.log(np.cosh(y_pred - y)))

    @staticmethod
    def chi_squared_fitness(params, x, y, sigma, A_array):
        y_pred = FitnessFunctions.model2(x, params, A_array)
        return np.sum(((y - y_pred) / sigma) ** 2)
    

class PolyFit:
    def __init__(self, x_data, y_data, disp_array, fitness_model='chi_squared', wavelet_level=5, wavelet_threshold=1, wavelet='sym8',
                 perform_wavelet=False, P2_bound=[-0.000001, 0], Q_bound=[-0.000001, 0.000001], R_bound=[0, 0.3],
                 H_bound=[0, 1.5], eta_bound=[0, 1], A_multiplier=[0.75, 1.05], disp_deviation=1, max_num_iteration = 1000,
                 population_size = 20, mutation_probability = 0.15, elit_ratio = 0.2, crossover_probability = 0.5, parents_portion = 0.3,
                 crossover_type = 'uniform', max_iteration_without_improv = None):
        self.x_data = x_data
        self.y_data = y_data
        self.disp_array = disp_array
        self.sorted_indices = np.searchsorted(self.x_data, self.disp_array)
        self.A_array = y_data[self.sorted_indices]
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.wavelet_threshold = wavelet_threshold
        self.perform_wavelet = perform_wavelet
        self.P2_bound = P2_bound
        self.Q_bound = Q_bound
        self.R_bound = R_bound
        self.H_bound = H_bound
        self.eta_bound = eta_bound
        self.A_multiplier = A_multiplier
        self.disp_deviation = disp_deviation
        self.max_num_iteration = max_num_iteration
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.elit_ratio = elit_ratio
        self.crossover_probability = crossover_probability
        self.parents_portion = parents_portion
        self.crossover_type = crossover_type
        self.max_iteration_without_improv = max_iteration_without_improv

        self.bounds = self.initialize_bounds()
        self.fitness_function = self.set_fitness_function(fitness_model)
        self.model2 = FitnessFunctions.model2

    def initialize_bounds(self):
        varbound = [self.P2_bound, self.Q_bound, self.R_bound]
        for i, A in enumerate(self.A_array):
            A_bound = [A * self.A_multiplier[0], A * self.A_multiplier[1]]
            disp_bound = [self.disp_array[i] - self.disp_deviation, self.disp_array[i] + self.disp_deviation]
            varbound += [self.H_bound, self.eta_bound, A_bound, disp_bound]
        return np.array(varbound)

    def set_fitness_function(self, model):
        fitness_models = {
            'rwp': lambda params: FitnessFunctions.rwp_fitness_function(params, self.x_data, self.y_data, self.A_array),
            'huber': lambda params: FitnessFunctions.huber_fitness_function(params, self.x_data, self.y_data, self.A_array,delta=2.0),
            'rmse': lambda params: FitnessFunctions.rmse_fitness_function(params, self.x_data, self.y_data, self.A_array),
            'mse': lambda params: FitnessFunctions.mse_fitness_function(params, self.x_data, self.y_data, self.A_array),
            'mae': lambda params: FitnessFunctions.mae_fitness_function(params, self.x_data, self.y_data, self.A_array),
            'msle': lambda params: FitnessFunctions.msle_fitness_function(params, self.x_data, self.y_data, self.A_array),
            'logcosh': lambda params: FitnessFunctions.logcosh_fitness_function(params, self.x_data, self.y_data, self.A_array),
            'chi_squared': lambda params: FitnessFunctions.chi_squared_fitness(params, self.x_data, self.y_data, np.std(self.y_data), self.A_array)
        }
        return fitness_models[model]
    
    def perform_wavelet_transform(self, input_signal):
        coeffs = pywt.wavedec(input_signal, self.wavelet, level=self.wavelet_level)
        # Apply thresholding to detail coefficients
        coeffs = [coeffs[0]] + [pywt.threshold(i, self.wavelet_threshold, mode='hard') for i in coeffs[1:]]
        # Reconstruct the signal using the thresholded coefficients
        reconstructed_signal = pywt.waverec(coeffs, self.wavelet)
        return reconstructed_signal

    def genetic_algorithm_optimization(self):
        algorithm_param = {
            'max_num_iteration': self.max_num_iteration,
            'population_size': self.population_size,
            'mutation_probability': self.mutation_probability,
            'elit_ratio': self.elit_ratio,
            'crossover_probability': self.crossover_probability,
            'parents_portion': self.parents_portion,
            'crossover_type': self.crossover_type,
            'max_iteration_without_improv': self.max_iteration_without_improv
        }
        model = ga(
            function=lambda params: self.fitness_function(params),
            dimension=len(self.bounds),
            variable_type='real',
            variable_boundaries=self.bounds,
            algorithm_parameters=algorithm_param
        )
        model.run()
        return model.output_dict['variable']
    
    def return_parameters(self, best_params, radians=False):
        FWHM = []
        angles = []
        Intensity = []
        etas = []

        param_index = 3
        for _ in range((len(best_params) - 3) // 4):
            FWHM.append(best_params[param_index])
            etas.append(best_params[param_index+1])
            Intensity.append(best_params[param_index + 2])
            angles.append(best_params[param_index + 3])
            param_index += 4

        bkg_parameters = np.array([best_params[0], best_params[1], best_params[2]])
        FWHM = np.array(FWHM)
        angles = np.array(angles)
        Intensity = np.array(Intensity)
        etas = np.array(etas)
        index = np.searchsorted(self.x_data, angles)

        if radians:
            return best_params, FWHM * np.pi / 180, angles * np.pi / 180, index ,bkg_parameters, etas, Intensity 
        else:
            return best_params, FWHM, angles, index, bkg_parameters, etas, Intensity 
    
    def run_analysis(self):
        if self.perform_wavelet:
            self.y_data = self.perform_wavelet_transform(self.y_data)
        best_params = self.genetic_algorithm_optimization()
        return self.return_parameters(best_params, radians=True)
    
    def polyGraph(x_data, y_data, y_best_fit, index=None, miller_idx=None, R=0, offset=0.25):
        #May have some issues graphing Millers Indexes and angles should be used in radians

        I_aprox = y_best_fit - R
        I_reconstructed = y_data - R

        error = I_aprox - I_reconstructed - offset

        plt.figure(figsize=(10, 5))
        plt.plot(x_data, I_reconstructed, 'o', label='Datos experimentales', markersize=0.5)
        plt.plot(x_data, I_aprox, '-', label='Mejor ajuste')
        plt.plot(x_data, error, '-', label='Gráfica de error')

        # Check if all the parameters are not None and they have the same length
        if index is not None and miller_idx is not None:
            if len(index) == len(index) == len(miller_idx):
                for idx, label in zip(index, miller_idx):
                    plt.plot(x_data[idx], I_aprox[idx], "kx", label='Miller Indices' if 'Miller Indices' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.text(x_data[idx], I_aprox[idx], ' {}'.format(label), color='black', verticalalignment='bottom')
            else:
                print("Error: 'index', and 'miller_idx' must have the same length.")
        else:
            print("Warning: One or more of the parameters 'angles', 'index', 'miller_idx' are undefined or empty.")

        plt.xlabel('$2\Theta [°]$')
        plt.ylabel('Intensidad')
        plt.title('Aproximación con superposición de funciones Pseudo-Voigt')
        plt.grid()
        plt.legend()
        plt.show()
    
    def Scherrer(theta,H, lamb = 0.1540656): #nm
        #Use radians and H, lambda_Cu is by default
        Cristalite_size = 2 * np.sqrt(np.log(2)/np.pi) * lamb/(H*np.cos(theta/2)) 
        return np.mean(Cristalite_size), Cristalite_size
    
    def Halder_Wagner(angles_rads,FWHM_rads, lamb = 0.1540656):    
        y_values = (FWHM_rads/np.tan(angles_rads))**2 #*1000
        x_values = FWHM_rads/ (np.tan(angles_rads)*np.sin(angles_rads))

        x_values = x_values.reshape(-1,1)

        model_HW = LinearRegression()
        model_HW.fit(x_values,y_values)

        intercept = model_HW.intercept_
        coef = model_HW.coef_
        coef = coef[0]

        epsilon = np.sqrt(np.abs(intercept) / 16)
        size = lamb/coef

        strain = epsilon*100

        r_squared = model_HW.score(x_values, y_values)

        plt.plot(x_values,y_values,'x', label='Halder Wagner')
        plt.plot(x_values,model_HW.predict(x_values), label='Linear (Halder Wagner)')
        plt.ylabel(r'H/(tan$\theta$)$^2$')
        plt.xlabel(r'H/(tan$\theta$sin$\theta$)')
        plt.title('Aproximación lineal con Halder Wagner')
        plt.legend()
        plt.grid()
        plt.show()

        return strain, size, r_squared
    
    def Warren_Averbach(self,angles,FWHM, peaks_pair,hkl = None, ABG = None,abc = None,x_data = None, y_data = None, n = 10):
        # Choose number of coefficients for linear fit
        vect_fit_pair = np.array([0,1,2]); # For RT, 200

        peaks_pair = np.array(peaks_pair)

        #Find indx for theta_max and theta_min
        theta_min = (angles - FWHM*2.3) *180/np.pi
        theta_max = (angles + FWHM*2.3) *180/np.pi

        if x_data is not None and y_data is not None:
            idx_min = np.searchsorted(x_data, theta_min)
            idx_max = np.searchsorted(x_data, theta_max)

            peaks_y = np.empty((len(idx_max),), dtype=object)
            peaks_x = np.empty((len(idx_max),), dtype=object)

            for i in range(len(self.disp_array)):
                peaks_y[i] = list(y_data[idx_min[i] : idx_max[i]])
                peaks_x[i] = list(x_data[idx_min[i] : idx_max[i]])
                # plt.plot(peaks_x[i],peaks_y[i])

        else:
            idx_min = np.searchsorted(self.x_data, theta_min)
            idx_max = np.searchsorted(self.x_data, theta_max)

            peaks_y = np.empty((len(idx_max),), dtype=object)
            peaks_x = np.empty((len(idx_max),), dtype=object)

            for i in range(len(self.disp_array)):
                peaks_y[i] = list(self.y_data[idx_min[i] : idx_max[i]])
                peaks_x[i] = list(self.x_data[idx_min[i] : idx_max[i]])
                # plt.plot(peaks_x[i],peaks_y[i])

        #Find indx for theta_max and theta_min
        theta_min = theta_min * np.pi/180
        theta_max = theta_max * np.pi/180

        lamb = 0.1540656

        if hkl is not None and abc is not None and ABG is not None:
            h = hkl[0,:]; k = hkl[1,:]; l = hkl[2,:]
            a_r = 1/abc[0]; b_r = 1/abc[1]; c_r = 1/abc[2]
            A = ABG[0]; B = ABG[1]; G = ABG[2] #In radians

            g2 = h**2 * a_r**2 + k**2 * b_r**2 + l**2 * c_r**2
            g2 += 2*h*k*a_r*b_r*np.cos(G)
            g2 += 2*h*l*a_r*c_r*np.cos(B)
            g2 += 2*l*k*b_r*c_r*np.cos(A)

            miller_idx = concatenate_indices(h, k, l)

        else:
            print("Reciprocal distance is computed with Bragg's law")
            g2 = (2*np.sin(angles)/lamb)**2

        ### Initial calculations
        #number of fourier coefficients
        n_coeff = np.arange(n).reshape(-1, 1)
        a3 = np.abs(lamb/ (2* ( np.sin(theta_max) - np.sin(theta_min) )))
        L = n_coeff*a3 #nm

        WA = np.zeros( (n,len(self.disp_array)), dtype=complex)

        for i in range(len(self.disp_array)):
            WA[:,i] = fft(fftshift(peaks_y[i]), n=n)

        WA = np.abs(WA)
        WA = WA/np.max(WA)
        lnA = np.log(WA).conj().T

        # Create a figure and axis
        fig, ax = plt.subplots()

        legend_str = []  # initial legend
        for i in range(n):
            ax.plot(g2, lnA[:, i], '*-', linewidth=1, markersize=5, label = f"A{i}, L = {L[i,0]:.3f} nm" )
            
        # Adding text annotations
        if hkl is not None and abc is not None and ABG is not None:
            for i, label in enumerate(miller_idx):  # Assuming there is information about Miller's Index nor cell parameters
                ax.text(g2[i], lnA[i, 0], label, verticalalignment='bottom', horizontalalignment='left')
        else:
            for i in range(len(self.disp_array)):  # Assuming there is no information about Miller's Index nor cell parameters
                ax.text(g2[i], lnA[i, 0], f'Peak {i+1}', verticalalignment='bottom', horizontalalignment='left')
        

        # Set title and labels
        ax.set_title(f"Original WA")
        ax.grid()
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), shadow=True)
        ax.set_xlabel(r'$g^2 (nm^{-2})$')
        ax.set_ylabel(r'$ln(A_n)$')

        # Show plot
        plt.show()

        fig, ax = plt.subplots()

        legend_str = []  # initial legend
        for i in range(n):  # num_coeff needs to be defined
            ax.plot(g2[peaks_pair], lnA[peaks_pair, i], '*-', linewidth=1, markersize=5)
            legend_str.append(f"A{i}, L = {L[i]} nm")  # Formatting legend string

        # Adding text annotations
        if hkl is not None and abc is not None and ABG is not None:
            for i,peak in enumerate(peaks_pair):
                ax.text(g2[peaks_pair][i], lnA[peaks_pair, 0][i], miller_idx[peak],
                        verticalalignment='bottom', horizontalalignment='left')
            ax.set_title(f"{miller_idx[peaks_pair[0]]} - {miller_idx[peaks_pair[1]]}")
            
        else:
            for i in range(len(peaks_pair)):
                ax.text(g2[peaks_pair][i], lnA[peaks_pair, 0][i], f'Peak {peaks_pair[i]}',
                        verticalalignment='bottom', horizontalalignment='left')
            ax.set_title(f"Peak{peaks_pair[0]} - Peak{peaks_pair[1]}")
        
        ax.set_xlabel(r'$g^2 (nm^{-2})$')
        ax.set_ylabel(r'$ln(A_n)$')
        # Show plot
        plt.show()

        # Initialize arrays to store results
        intercept_pair = np.zeros(n)
        slope_pair = np.zeros(n)
        strain_pair = np.zeros(n)

        # Loop through each coefficient
        for i in range(n):  # Python indexing starts at 0
            # Fit a polynomial of degree 1 (linear fit)
            p = np.polyfit(g2[peaks_pair], lnA[peaks_pair, i], 1)
            
            # Store intercept and slope
            intercept_pair[i] = p[1]
            slope_pair[i] = p[0]
            
            # Calculate strain based on the slope
            if i != 0:
                strain_pair[i] = -slope_pair[i] / (2 * np.pi**2* L[i,0]**2)

        As_pair = np.exp(intercept_pair)
        As_pair = As_pair / np.max(As_pair)

        L_peak1 = L[:,peaks_pair[0]]
        L_peak2 = L[:,peaks_pair[1]]

        fit_pair1 = np.polyfit(L_peak1[vect_fit_pair], As_pair[vect_fit_pair].reshape(-1, 1),1)
        fit_pair2 = np.polyfit(L_peak2[vect_fit_pair], As_pair[vect_fit_pair].reshape(-1, 1),1)

        t_area_pair = (-fit_pair2[1]/fit_pair2[0] + -fit_pair1[1]/fit_pair1[0])/2
        #t_area_pair = -fit_pair2[1]/fit_pair2[0] 

        fig, ax = plt.subplots()

        # Plotting text for each coefficient
        for i in range(n):  # Ensure num_coeff and other variables are defined
            ax.text(L_peak1[i], As_pair[i], f"A{i}",
                    verticalalignment='bottom', horizontalalignment='left')

        for i in range(n):  # Ensure num_coeff and other variables are defined
            ax.text(L_peak2[i], As_pair[i], f"A{i}",
                    verticalalignment='bottom', horizontalalignment='left')

        
        ax.plot(L_peak1, fit_pair1[0]*L_peak1+fit_pair1[1],'--g')
        ax.plot(L_peak2, fit_pair2[0]*L_peak2+fit_pair2[1],'--b')
        ax.set_ylim(0,)
        ax.set_xlabel('L[nm]')
        ax.set_ylabel (r'$A^s[L]$')
        #Set title
        # Adding text annotations
        if hkl is not None and abc is not None and ABG is not None:
            ax.plot(L_peak1,As_pair,'*g', label= miller_idx[peaks_pair[0]])
            ax.plot(L_peak2,As_pair,'*b', label= miller_idx[peaks_pair[1]])
            ax.set_title(f"Con indices de Miller\n{miller_idx[peaks_pair[0]]} - {miller_idx[peaks_pair[1]]}\n$t_{{vol}}$ = {t_area_pair[0]:.3f} nm")
        else:
            ax.plot(L_peak1,As_pair,'*g', label= f'Peak {peaks_pair[0]}')
            ax.plot(L_peak2,As_pair,'*b', label= f'Peak {peaks_pair[1]}')
            ax.set_title(f"Con ley de Bragg\nPeak{peaks_pair[0]} - Peak{peaks_pair[1]}\n$t_{{vol}}$ = {t_area_pair[0]:.3f} nm")
                

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True)
        ax.grid()

        return strain_pair