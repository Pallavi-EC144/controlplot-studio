import numpy as np
from scipy import signal
import sympy as sp
import json
import math

class ControlEngine:
    def __init__(self):
        self.num = []
        self.den = []
        self.poles = []
        self.zeros = []
        self.gain = 1
        
    def parse_transfer_function(self, num_str, den_str):
        """Parse transfer function from string input"""
        try:
            # Parse numerator and denominator
            if num_str.strip():
                num = [float(x.strip()) for x in num_str.split(',')]
            else:
                num = [1]
            
            if den_str.strip():
                den = [float(x.strip()) for x in den_str.split(',')]
            else:
                den = [1]
            
            self.num = num
            self.den = den
            self.gain = num[0] / den[0] if den[0] != 0 else 1
            
            # Find poles and zeros
            self.poles = np.roots(den)
            self.zeros = np.roots(num)
            
            return True
        except:
            return False
    
    def parse_from_builder(self, gain, zeros, poles):
        """Parse from builder interface"""
        try:
            gain = float(gain)
            zeros = [float(z) for z in zeros if z.strip()]
            poles = [float(p) for p in poles if p.strip()]
            
            # Build polynomial from zeros
            num = [gain]
            for z in zeros:
                num = np.convolve(num, [1, -z])
            
            # Build polynomial from poles
            den = [1]
            for p in poles:
                den = np.convolve(den, [1, -p])
            
            self.num = num.tolist()
            self.den = den.tolist()
            self.poles = poles
            self.zeros = zeros
            self.gain = gain
            
            return True
        except:
            return False
    
    def get_frequency_response(self, omega_min=0.01, omega_max=10000, num_points=1000):
        """Calculate frequency response"""
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_points)
        w = omega
        
        # Calculate frequency response
        s = 1j * omega
        num_eval = np.polyval(self.num, s)
        den_eval = np.polyval(self.den, s)
        
        H = num_eval / den_eval
        
        # Magnitude in dB
        mag_db = 20 * np.log10(np.abs(H) + 1e-10)
        mag_linear = np.abs(H)
        
        # Phase in degrees
        phase_deg = np.angle(H, deg=True)
        
        # Real and imaginary parts
        real = np.real(H)
        imag = np.imag(H)
        
        return {
            'omega': omega.tolist(),
            'mag_db': mag_db.tolist(),
            'mag_linear': mag_linear.tolist(),
            'phase_deg': phase_deg.tolist(),
            'real': real.tolist(),
            'imag': imag.tolist()
        }
    
    def get_asymptotic_response(self, omega_min=0.01, omega_max=10000, num_points=1000):
        """Calculate asymptotic approximation of Bode plot"""
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_points)
        
        # Start with gain term
        mag_asym = np.full(len(omega), 20 * np.log10(abs(self.gain)))
        phase_asym = np.full(len(omega), 0.0 if self.gain > 0 else -180.0)
        
        # Add contributions from poles
        for pole in self.poles:
            if abs(pole.imag) < 1e-10:  # Real pole
                pole = pole.real
                if pole != 0:
                    # Pole at -a
                    omega_c = abs(pole)
                    idx = omega > omega_c
                    mag_asym[idx] -= 20 * np.log10(omega[idx] / omega_c)
                    phase_asym[idx] -= 90
                else:  # Pole at origin
                    mag_asym -= 20 * np.log10(omega)
                    phase_asym -= 90
        
        # Add contributions from zeros
        for zero in self.zeros:
            if abs(zero.imag) < 1e-10:  # Real zero
                zero = zero.real
                if zero != 0:
                    omega_c = abs(zero)
                    idx = omega > omega_c
                    mag_asym[idx] += 20 * np.log10(omega[idx] / omega_c)
                    phase_asym[idx] += 90
                else:  # Zero at origin
                    mag_asym += 20 * np.log10(omega)
                    phase_asym += 90
        
        return {
            'omega': omega.tolist(),
            'mag_db': mag_asym.tolist(),
            'phase_deg': phase_asym.tolist()
        }
    
    def get_pole_zero_data(self):
        """Get pole-zero information for plotting"""
        poles_data = []
        zeros_data = []
        
        for p in self.poles:
            if abs(p.imag) < 1e-10:
                poles_data.append({'real': p.real, 'imag': 0})
            else:
                poles_data.append({'real': p.real, 'imag': p.imag})
                poles_data.append({'real': p.real, 'imag': -p.imag})
        
        for z in self.zeros:
            if abs(z.imag) < 1e-10:
                zeros_data.append({'real': z.real, 'imag': 0})
            else:
                zeros_data.append({'real': z.real, 'imag': z.imag})
                zeros_data.append({'real': z.real, 'imag': -z.imag})
        
        return {'poles': poles_data, 'zeros': zeros_data}
    
    def get_stability_info(self):
        """Check system stability"""
        # Check if all poles have negative real parts
        stable = all(p.real < 0 for p in self.poles)
        
        # Calculate gain and phase margins (simplified)
        # Find frequency where phase crosses -180 degrees
        freq_response = self.get_frequency_response()
        phase = np.array(freq_response['phase_deg'])
        mag_db = np.array(freq_response['mag_db'])
        omega = np.array(freq_response['omega'])
        
        # Find phase crossover
        phase_cross_idx = np.where(np.diff(np.sign(phase + 180)))[0]
        gain_margin = None
        phase_margin = None
        
        if len(phase_cross_idx) > 0:
            idx = phase_cross_idx[0]
            if idx < len(mag_db) - 1:
                gain_margin = -mag_db[idx]
        
        # Find gain crossover (0 dB)
        gain_cross_idx = np.where(np.diff(np.sign(mag_db)))[0]
        if len(gain_cross_idx) > 0:
            idx = gain_cross_idx[0]
            if idx < len(phase) - 1:
                phase_margin = 180 + phase[idx]
        
        return {
            'stable': stable,
            'gain_margin': gain_margin,
            'phase_margin': phase_margin
        }

    def get_bode_data(self):
        """Get complete Bode plot data"""
        freq_response = self.get_frequency_response()
        asym_response = self.get_asymptotic_response()
        pole_zero = self.get_pole_zero_data()
        stability = self.get_stability_info()
        
        return {
            'frequency_response': freq_response,
            'asymptotic': asym_response,
            'pole_zero': pole_zero,
            'stability': stability
        }
