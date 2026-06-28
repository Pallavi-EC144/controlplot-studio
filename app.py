import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
import re

st.set_page_config(
    page_title="Interactive Frequency Response Explorer",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .main-header h1 { font-size: 2.5rem; margin-bottom: 5px; }
    .main-header p { opacity: 0.9; font-size: 1.1rem; }
    .derivation-box {
        background: #f8f9fe;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    .contribution-box {
        padding: 10px;
        margin: 5px 0;
        border-radius: 6px;
        border-left: 3px solid;
        background: white;
    }
    .good { color: #22c55e; font-weight: bold; }
    .warning { color: #eab308; font-weight: bold; }
    .bad { color: #ef4444; font-weight: bold; }
    .math-block {
        background: #f0f0f0;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.2rem;
    }
    .step-number {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)

class TransferFunctionAnalyzer:
    """Educational engine that derives the transfer function mathematically"""
    
    def __init__(self):
        self.num = [1]
        self.den = [1]
        self.poles = []
        self.zeros = []
        self.original_gain = 1
        self.standard_gain = 1
        self.corner_freqs = []
        self.origin_poles = 0
        self.origin_zeros = 0
        self.real_poles = []
        self.real_zeros = []
        self.complex_poles = []
        self.complex_zeros = []
        
    def parse(self, num_str, den_str):
        try:
            if num_str.strip():
                num = [float(x.strip()) for x in num_str.split(',') if x.strip()]
            else:
                num = [1]
            
            if den_str.strip():
                den = [float(x.strip()) for x in den_str.split(',') if x.strip()]
            else:
                den = [1]
            
            while len(num) > 1 and abs(num[-1]) < 1e-10:
                num.pop()
            while len(den) > 1 and abs(den[-1]) < 1e-10:
                den.pop()
            
            self.num = num
            self.den = den
            self.poles = np.roots(den)
            self.zeros = np.roots(num)
            self.original_gain = num[0] / den[0] if den[0] != 0 else 1
            
            # Classify poles and zeros
            self.origin_poles = 0
            self.origin_zeros = 0
            self.real_poles = []
            self.real_zeros = []
            self.complex_poles = []
            self.complex_zeros = []
            self.corner_freqs = []
            
            for p in self.poles:
                if abs(p.imag) < 1e-10 and abs(p.real) < 1e-10:
                    self.origin_poles += 1
                elif abs(p.imag) < 1e-10:
                    self.real_poles.append(abs(p.real))
                    self.corner_freqs.append(abs(p.real))
                else:
                    self.complex_poles.append((abs(p.real), abs(p.imag)))
                    self.corner_freqs.append(np.sqrt(p.real**2 + p.imag**2))
            
            for z in self.zeros:
                if abs(z.imag) < 1e-10 and abs(z.real) < 1e-10:
                    self.origin_zeros += 1
                elif abs(z.imag) < 1e-10:
                    self.real_zeros.append(abs(z.real))
                    self.corner_freqs.append(abs(z.real))
                else:
                    self.complex_zeros.append((abs(z.real), abs(z.imag)))
                    self.corner_freqs.append(np.sqrt(z.real**2 + z.imag**2))
            
            self.corner_freqs.sort()
            
            # Calculate standard gain K'
            self.standard_gain = self.original_gain
            for z in self.real_zeros:
                self.standard_gain *= z
            for p in self.real_poles:
                self.standard_gain /= p
            
            return True
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return False
    
    def get_derivation(self):
        """Generate complete mathematical derivation"""
        steps = []
        
        # Step 1: Original transfer function
        num_str = " + ".join([f"{c:.3f}s^{i}" for i, c in enumerate(reversed(self.num)) if abs(c) > 1e-10])
        den_str = " + ".join([f"{c:.3f}s^{i}" for i, c in enumerate(reversed(self.den)) if abs(c) > 1e-10])
        steps.append({
            'title': 'Step 1: Original Transfer Function',
            'content': f"$$H(s) = \\frac{{{num_str}}}{{{den_str}}}$$",
            'explanation': 'This is the transfer function you entered.'
        })
        
        # Step 2: Factor into poles and zeros
        poles_str = ", ".join([f"{p.real:.2f}" if abs(p.imag) < 1e-10 else f"{p.real:.2f} ± {abs(p.imag):.2f}i" for p in self.poles])
        zeros_str = ", ".join([f"{z.real:.2f}" if abs(z.imag) < 1e-10 else f"{z.real:.2f} ± {abs(z.imag):.2f}i" for z in self.zeros])
        steps.append({
            'title': 'Step 2: Poles and Zeros',
            'content': f"**Poles:** {poles_str}\n\n**Zeros:** {zeros_str}",
            'explanation': 'Poles and zeros determine the shape of the frequency response.'
        })
        
        # Step 3: Standard Bode Form
        steps.append({
            'title': 'Step 3: Standard Bode Form',
            'content': self._get_standard_form(),
            'explanation': 'Each factor is written as (1 + s/ω_c) for easy Bode construction.'
        })
        
        # Step 4: Gain
        gain_db = 20 * np.log10(abs(self.standard_gain) + 1e-10)
        steps.append({
            'title': 'Step 4: Standard Gain',
            'content': f"$$K' = {self.standard_gain:.4f} \\rightarrow {gain_db:.2f} \\text{{ dB}}$$",
            'explanation': f"The original gain {self.original_gain:.3f} has been standardized by dividing by pole frequencies and multiplying by zero frequencies."
        })
        
        # Step 5: Individual contributions
        steps.append({
            'title': 'Step 5: Individual Contributions',
            'content': self._get_contributions_table(),
            'explanation': 'Each pole and zero contributes independently to the total response.'
        })
        
        # Step 6: Magnitude equation
        steps.append({
            'title': 'Step 6: Magnitude Equation',
            'content': self._get_magnitude_equation(),
            'explanation': 'The total magnitude is the product of all individual magnitudes.'
        })
        
        # Step 7: Phase equation
        steps.append({
            'title': 'Step 7: Phase Equation',
            'content': self._get_phase_equation(),
            'explanation': 'The total phase is the sum of all individual phases.'
        })
        
        return steps
    
    def _get_standard_form(self):
        """Get standard Bode form"""
        parts = []
        if abs(self.standard_gain - 1) > 1e-10:
            parts.append(f"{self.standard_gain:.4f}")
        
        # Zero at origin
        if self.origin_zeros > 0:
            parts.append("s" * self.origin_zeros)
        
        # Real zeros
        for z in self.real_zeros:
            parts.append(f"(1 + s/{z:.2f})")
        
        # Complex zeros
        for z_real, z_imag in self.complex_zeros:
            wn = np.sqrt(z_real**2 + z_imag**2)
            zeta = z_real / wn
            parts.append(f"(1 + 2({zeta:.2f})s/{wn:.2f} + (s/{wn:.2f})²)")
        
        # Denominator
        den_parts = []
        
        # Pole at origin
        if self.origin_poles > 0:
            den_parts.append("s" * self.origin_poles)
        
        # Real poles
        for p in self.real_poles:
            den_parts.append(f"(1 + s/{p:.2f})")
        
        # Complex poles
        for p_real, p_imag in self.complex_poles:
            wn = np.sqrt(p_real**2 + p_imag**2)
            zeta = p_real / wn
            den_parts.append(f"(1 + 2({zeta:.2f})s/{wn:.2f} + (s/{wn:.2f})²)")
        
        num_str = " \\cdot ".join(parts) if parts else "1"
        den_str = " \\cdot ".join(den_parts) if den_parts else "1"
        
        return f"$$H(s) = \\frac{{{num_str}}}{{{den_str}}}$$"
    
    def _get_contributions_table(self):
        """Get table of individual contributions"""
        lines = []
        
        # Gain
        gain_db = 20 * np.log10(abs(self.standard_gain) + 1e-10)
        lines.append(f"• **Gain:** {self.standard_gain:.4f} → {gain_db:.2f} dB (constant)")
        
        # Integrators
        if self.origin_poles > 0:
            lines.append(f"• **Integrator (×{self.origin_poles}):** -20 dB/dec, -90° × {self.origin_poles}")
        
        if self.origin_zeros > 0:
            lines.append(f"• **Differentiator (×{self.origin_zeros}):** +20 dB/dec, +90° × {self.origin_zeros}")
        
        # Real poles
        for i, p in enumerate(self.real_poles):
            lines.append(f"• **Pole {i+1}:** at ω = {p:.2f} rad/s → -20 dB/dec after corner, phase from 0° to -90°")
        
        # Real zeros
        for i, z in enumerate(self.real_zeros):
            lines.append(f"• **Zero {i+1}:** at ω = {z:.2f} rad/s → +20 dB/dec after corner, phase from 0° to +90°")
        
        return "\n".join(lines)
    
    def _get_magnitude_equation(self):
        """Get magnitude equation"""
        parts = []
        parts.append(str(abs(self.standard_gain)))
        
        if self.origin_zeros > 0:
            parts.append(f"ω^{self.origin_zeros}")
        
        for z in self.real_zeros:
            parts.append(f"\\sqrt{{1 + (ω/{z:.2f})²}}")
        
        if self.origin_poles > 0:
            parts.append(f"\\frac{{1}}{{ω^{self.origin_poles}}}")
        
        for p in self.real_poles:
            parts.append(f"\\frac{{1}}{{\\sqrt{{1 + (ω/{p:.2f})²}}}}")
        
        for z_real, z_imag in self.complex_zeros:
            wn = np.sqrt(z_real**2 + z_imag**2)
            parts.append(f"\\sqrt{{(1 - (ω/{wn:.2f})²)² + (2*{z_real/wn:.2f}*ω/{wn:.2f})²}}")
        
        for p_real, p_imag in self.complex_poles:
            wn = np.sqrt(p_real**2 + p_imag**2)
            parts.append(f"\\frac{{1}}{{\\sqrt{{(1 - (ω/{wn:.2f})²)² + (2*{p_real/wn:.2f}*ω/{wn:.2f})²}}}}")
        
        eq = " \\cdot ".join(parts)
        return f"$$|H(jω)| = {eq}$$"
    
    def _get_phase_equation(self):
        """Get phase equation"""
        parts = []
        
        # Phase from standard gain
        if self.standard_gain < 0:
            parts.append("180°")
        
        # Phase from origin zeros (differentiators)
        if self.origin_zeros > 0:
            parts.append(f"+ 90° × {self.origin_zeros}")
        
        # Phase from origin poles (integrators)
        if self.origin_poles > 0:
            parts.append(f"- 90° × {self.origin_poles}")
        
        # Phase from real zeros
        for z in self.real_zeros:
            parts.append(f"+ tan⁻¹(ω/{z:.2f})")
        
        # Phase from real poles
        for p in self.real_poles:
            parts.append(f"- tan⁻¹(ω/{p:.2f})")
        
        # Phase from complex zeros
        for z_real, z_imag in self.complex_zeros:
            wn = np.sqrt(z_real**2 + z_imag**2)
            parts.append(f"+ tan⁻¹(2*{z_real/wn:.2f}*ω/{wn:.2f} / (1 - (ω/{wn:.2f})²))")
        
        # Phase from complex poles
        for p_real, p_imag in self.complex_poles:
            wn = np.sqrt(p_real**2 + p_imag**2)
            parts.append(f"- tan⁻¹(2*{p_real/wn:.2f}*ω/{wn:.2f} / (1 - (ω/{wn:.2f})²))")
        
        if not parts:
            parts.append("0°")
        
        return f"$$\phi(ω) = {' + '.join(parts)}$$"
    
    def parse_from_builder(self, gain, zeros, poles):
        try:
            gain = float(gain)
            zeros = [float(z) for z in zeros if z and z.strip()]
            poles = [float(p) for p in poles if p and p.strip()]
            
            num = [gain]
            for z in zeros:
                if z != 0:
                    num = np.convolve(num, [1, z])
                else:
                    num = np.convolve(num, [1, 0])
                    while len(num) > 1 and abs(num[0]) < 1e-10:
                        num = num[1:]
            
            den = [1]
            for p in poles:
                if p != 0:
                    den = np.convolve(den, [1, p])
                else:
                    den = np.convolve(den, [1, 0])
                    while len(den) > 1 and abs(den[0]) < 1e-10:
                        den = den[1:]
            
            num_str = ','.join([str(x) for x in num])
            den_str = ','.join([str(x) for x in den])
            
            return self.parse(num_str, den_str)
        except Exception as e:
            st.error(f"Error building: {str(e)}")
            return False
    
    def get_frequency_response(self, omega_min=0.0001, omega_max=100000, num_points=5000):
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_points)
        s = 1j * omega
        H = np.polyval(self.num, s) / np.polyval(self.den, s)
        
        mag_db = 20 * np.log10(np.abs(H) + 1e-10)
        mag_linear = np.abs(H)
        phase_rad = np.unwrap(np.angle(H))
        phase_deg = np.degrees(phase_rad)
        
        return {
            'omega': omega,
            'mag_db': mag_db,
            'mag_linear': mag_linear,
            'phase_deg': phase_deg,
            'real': np.real(H),
            'imag': np.imag(H)
        }
    
    def get_individual_contributions(self, omega_min=0.0001, omega_max=100000, num_points=1000):
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_points)
        contributions = []
        colors = ['#000000', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        # Gain contribution
        gain_mag = np.full(len(omega), 20 * np.log10(abs(self.standard_gain) + 1e-10))
        gain_phase = np.full(len(omega), 0.0 if self.standard_gain > 0 else -180.0)
        contributions.append({'name': 'Gain', 'mag': gain_mag, 'phase': gain_phase, 'color': colors[0]})
        
        # Origin poles (integrators)
        if self.origin_poles > 0:
            mag = -self.origin_poles * 20 * np.log10(omega + 1e-10)
            phase = -self.origin_poles * 90.0 * np.ones(len(omega))
            contributions.append({'name': f'Integrator (×{self.origin_poles})', 'mag': mag, 'phase': phase, 'color': colors[1]})
        
        # Origin zeros (differentiators)
        if self.origin_zeros > 0:
            mag = self.origin_zeros * 20 * np.log10(omega + 1e-10)
            phase = self.origin_zeros * 90.0 * np.ones(len(omega))
            contributions.append({'name': f'Differentiator (×{self.origin_zeros})', 'mag': mag, 'phase': phase, 'color': colors[2]})
        
        # Real poles
        for i, p in enumerate(self.real_poles):
            mag = -20 * np.log10(np.sqrt(1 + (omega/p)**2))
            phase = -np.degrees(np.arctan2(omega, p))
            contributions.append({'name': f'Pole at {p:.1f}', 'mag': mag, 'phase': phase, 'color': colors[3 + i % len(colors)]})
        
        # Real zeros
        for i, z in enumerate(self.real_zeros):
            mag = 20 * np.log10(np.sqrt(1 + (omega/z)**2))
            phase = np.degrees(np.arctan2(omega, z))
            contributions.append({'name': f'Zero at {z:.1f}', 'mag': mag, 'phase': phase, 'color': colors[4 + i % len(colors)]})
        
        return {'omega': omega, 'contributions': contributions}
    
    def calculate_margins(self):
        """Calculate margins using the CORRECT formulas"""
        fr = self.get_frequency_response()
        omega = fr['omega']
        mag_db = fr['mag_db']
        phase_deg = fr['phase_deg']
        
        gain_margin = None
        phase_margin = None
        gm_freq = None
        pm_freq = None
        
        # --- Find GAIN CROSSOVER (where |H| = 0 dB = 1) ---
        # Check for exact hits first
        exact_zero_idx = np.where(np.isclose(mag_db, 0, atol=1e-6))[0]
        
        # Then check for crossings
        gain_cross_idx = np.where(
            (mag_db[:-1] > 0) & (mag_db[1:] <= 0)
        )[0]
        gain_cross_idx2 = np.where(
            (mag_db[:-1] < 0) & (mag_db[1:] >= 0)
        )[0]
        
        all_gain_cross = np.unique(np.concatenate([gain_cross_idx, gain_cross_idx2, exact_zero_idx]))
        all_gain_cross = all_gain_cross[all_gain_cross < len(mag_db) - 1]
        
        if len(all_gain_cross) > 0:
            idx = all_gain_cross[0]
            if idx in exact_zero_idx:
                # Exact hit - use directly
                phase_margin = 180 + phase_deg[idx]
                pm_freq = omega[idx]
            else:
                # Interpolate
                x1, x2 = mag_db[idx], mag_db[idx + 1]
                y1, y2 = phase_deg[idx], phase_deg[idx + 1]
                if abs(x2 - x1) > 1e-10:
                    t = (0 - x1) / (x2 - x1)
                    if 0 <= t <= 1:
                        pm = y1 + t * (y2 - y1)
                        phase_margin = 180 + pm
                        pm_freq = omega[idx] + t * (omega[idx + 1] - omega[idx])
        
        # --- Find PHASE CROSSOVER (where phase = -180°) ---
        # Check for exact hits first
        exact_180_idx = np.where(np.isclose(phase_deg, -180, atol=1e-3))[0]
        
        # Then check for crossings
        phase_cross_idx = np.where(
            (phase_deg[:-1] > -180) & (phase_deg[1:] <= -180)
        )[0]
        phase_cross_idx2 = np.where(
            (phase_deg[:-1] < -180) & (phase_deg[1:] >= -180)
        )[0]
        
        all_phase_cross = np.unique(np.concatenate([phase_cross_idx, phase_cross_idx2, exact_180_idx]))
        all_phase_cross = all_phase_cross[all_phase_cross < len(mag_db) - 1]
        
        if len(all_phase_cross) > 0:
            idx = all_phase_cross[0]
            if idx in exact_180_idx:
                # Exact hit - use directly
                gain_margin = -mag_db[idx]
                gm_freq = omega[idx]
            else:
                # Interpolate
                x1, x2 = phase_deg[idx], phase_deg[idx + 1]
                y1, y2 = mag_db[idx], mag_db[idx + 1]
                if abs(x2 - x1) > 1e-10:
                    t = (-180 - x1) / (x2 - x1)
                    if 0 <= t <= 1:
                        gm = y1 + t * (y2 - y1)
                        gain_margin = -gm
                        gm_freq = omega[idx] + t * (omega[idx + 1] - omega[idx])
        
        # Determine stability
        stable_poles = all(p.real < 1e-10 for p in self.poles)
        marginal_poles = any(abs(p.real) < 1e-10 and abs(p.imag) > 0 for p in self.poles)
        
        # Check if margins indicate stability
        if gain_margin is not None and phase_margin is not None:
            stable = (gain_margin > 0) and (phase_margin > 0) and stable_poles
        elif gain_margin is not None:
            stable = (gain_margin > 0) and stable_poles
        elif phase_margin is not None:
            stable = (phase_margin > 0) and stable_poles
        else:
            stable = stable_poles
        
        # Special case: if phase_margin is exactly 0, it's marginally stable
        if phase_margin is not None and abs(phase_margin) < 1e-6:
            stable = False
        
        return {
            'gain_margin': gain_margin,
            'phase_margin': phase_margin,
            'gm_freq': gm_freq,
            'pm_freq': pm_freq,
            'gain_margin_dB': gain_margin if gain_margin is not None else None,
            'phase_margin_deg': phase_margin if phase_margin is not None else None,
            'stable': stable,
            'marginal': marginal_poles,
            'min_phase': np.min(phase_deg),
            'max_phase': np.max(phase_deg),
            'min_mag': np.min(mag_db),
            'max_mag': np.max(mag_db)
        }
    
    def get_pz(self):
        poles_data = []
        zeros_data = []
        for p in self.poles:
            if abs(p.imag) < 1e-10:
                poles_data.append({'real': float(p.real), 'imag': 0.0})
            else:
                poles_data.append({'real': float(p.real), 'imag': float(p.imag)})
                poles_data.append({'real': float(p.real), 'imag': float(-p.imag)})
        for z in self.zeros:
            if abs(z.imag) < 1e-10:
                zeros_data.append({'real': float(z.real), 'imag': 0.0})
            else:
                zeros_data.append({'real': float(z.real), 'imag': float(z.imag)})
                zeros_data.append({'real': float(z.real), 'imag': float(-z.imag)})
        return {'poles': poles_data, 'zeros': zeros_data}
    
    def get_bode_data(self):
        fr = self.get_frequency_response()
        margins = self.calculate_margins()
        pz = self.get_pz()
        contrib = self.get_individual_contributions()
        derivation = self.get_derivation()
        
        # Also get asymptotic using the standard gain
        omega = np.logspace(np.log10(0.0001), np.log10(100000), 1000)
        mag_asym = np.full(len(omega), 20 * np.log10(abs(self.standard_gain) + 1e-10))
        phase_asym = np.full(len(omega), 0.0 if self.standard_gain > 0 else -180.0)
        
        for p in self.poles:
            if abs(p.imag) < 1e-10:
                p_real = p.real
                if abs(p_real) > 1e-10:
                    omega_c = abs(p_real)
                    mag_asym -= 20 * np.log10(np.sqrt(1 + (omega/omega_c)**2))
                    phase_asym -= np.degrees(np.arctan2(omega, omega_c))
                else:
                    mag_asym -= 20 * np.log10(omega + 1e-10)
                    phase_asym -= 90.0
        
        for z in self.zeros:
            if abs(z.imag) < 1e-10:
                z_real = z.real
                if abs(z_real) > 1e-10:
                    omega_c = abs(z_real)
                    mag_asym += 20 * np.log10(np.sqrt(1 + (omega/omega_c)**2))
                    phase_asym += np.degrees(np.arctan2(omega, omega_c))
                else:
                    mag_asym += 20 * np.log10(omega + 1e-10)
                    phase_asym += 90.0
        
        asym = {'omega': omega, 'mag_db': mag_asym, 'phase_deg': phase_asym}
        
        return {
            'frequency_response': fr,
            'asymptotic': asym,
            'margins': margins,
            'pole_zero': pz,
            'contributions': contrib,
            'derivation': derivation,
            'standard_gain': self.standard_gain,
            'origin_poles': self.origin_poles,
            'origin_zeros': self.origin_zeros
        }

# ============================================
# Plotting Functions
# ============================================

def create_bode_plot_with_contributions(data, show_contributions=True):
    fr = data['frequency_response']
    asym = data['asymptotic']
    margins = data['margins']
    contrib = data['contributions']
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=('Magnitude Response (dB)', 'Phase Response (degrees)'))
    
    # Individual contributions
    if show_contributions:
        for c in contrib['contributions']:
            fig.add_trace(
                go.Scatter(x=contrib['omega'], y=c['mag'], mode='lines',
                          name=f'{c["name"]} (mag)', line=dict(color=c['color'], width=1.5, dash='dot'),
                          showlegend=False),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=contrib['omega'], y=c['phase'], mode='lines',
                          name=f'{c["name"]} (phase)', line=dict(color=c['color'], width=1.5, dash='dot'),
                          showlegend=False),
                row=2, col=1
            )
    
    # Actual response
    fig.add_trace(
        go.Scatter(x=fr['omega'], y=fr['mag_db'], mode='lines',
                  name='Actual Magnitude', line=dict(color='#2563eb', width=3)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=fr['omega'], y=fr['phase_deg'], mode='lines',
                  name='Actual Phase', line=dict(color='#dc2626', width=3)),
        row=2, col=1
    )
    
    # Asymptotic approximation
    fig.add_trace(
        go.Scatter(x=asym['omega'], y=asym['mag_db'], mode='lines',
                  name='Asymptotic', line=dict(color='#22c55e', width=2, dash='dash')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=asym['omega'], y=asym['phase_deg'], mode='lines',
                  name='Asymptotic Phase', line=dict(color='#f59e0b', width=2, dash='dash')),
        row=2, col=1
    )
    
    # Reference lines
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_hline(y=-180, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
    
    # Margin markers
    if margins['gm_freq'] is not None:
        fig.add_vline(x=margins['gm_freq'], line_dash="dash", line_color="green", 
                     annotation_text=f"GM: {margins['gain_margin_dB']:.2f} dB", row=1, col=1)
    if margins['pm_freq'] is not None:
        fig.add_vline(x=margins['pm_freq'], line_dash="dash", line_color="orange", 
                     annotation_text=f"PM: {margins['phase_margin_deg']:.2f}°", row=2, col=1)
    
    fig.update_xaxes(type='log', title_text='Frequency (rad/s)', gridcolor='#f0f0f0', row=2, col=1)
    fig.update_xaxes(type='log', showticklabels=False, gridcolor='#f0f0f0', row=1, col=1)
    fig.update_yaxes(title_text='Magnitude (dB)', gridcolor='#f0f0f0', row=1, col=1)
    fig.update_yaxes(title_text='Phase (degrees)', gridcolor='#f0f0f0', row=2, col=1)
    fig.update_layout(height=600, hovermode='closest', showlegend=True,
                     legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'))
    
    return fig

def create_nyquist_plot(fr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fr['real'], y=fr['imag'], mode='lines',
                  name='Nyquist Plot', line=dict(color='#7c3aed', width=2.5),
                  hovertemplate='Real: %{x:.3f}<br>Imag: %{y:.3f}<br>ω: %{text:.2f} rad/s<extra></extra>',
                  text=fr['omega']))
    fig.add_trace(go.Scatter(x=[fr['real'][0]], y=[fr['imag'][0]], mode='markers',
                  name='ω → 0', marker=dict(color='#22c55e', size=12, symbol='circle')))
    fig.add_trace(go.Scatter(x=[fr['real'][-1]], y=[fr['imag'][-1]], mode='markers',
                  name='ω → ∞', marker=dict(color='#ef4444', size=12, symbol='x')))
    fig.add_trace(go.Scatter(x=[-1], y=[0], mode='markers',
                  name='-1 point', marker=dict(color='#000', size=10, symbol='star')))
    
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                  name='Unit Circle', line=dict(color='#ccc', width=1, dash='dot'), showlegend=False))
    
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.3)
    fig.update_layout(title='Nyquist Plot', height=500, hovermode='closest',
                      xaxis=dict(title='Real', gridcolor='#f0f0f0', scaleanchor='y', scaleratio=1),
                      yaxis=dict(title='Imaginary', gridcolor='#f0f0f0'),
                      legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'))
    return fig

def create_pole_zero_plot(pz):
    fig = go.Figure()
    if pz['poles']:
        fig.add_trace(go.Scatter(x=[p['real'] for p in pz['poles']], y=[p['imag'] for p in pz['poles']],
                      mode='markers', name='Poles',
                      marker=dict(color='#ef4444', size=14, symbol='x', line=dict(width=2))))
    if pz['zeros']:
        fig.add_trace(go.Scatter(x=[z['real'] for z in pz['zeros']], y=[z['imag'] for z in pz['zeros']],
                      mode='markers', name='Zeros',
                      marker=dict(color='#2563eb', size=14, symbol='circle-open', line=dict(width=2))))
    
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                  name='Unit Circle', line=dict(color='#ccc', width=1, dash='dot'), showlegend=False))
    
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.3)
    fig.update_layout(title='Pole-Zero Map', height=500, hovermode='closest',
                      xaxis=dict(title='Real', gridcolor='#f0f0f0', scaleanchor='y', scaleratio=1),
                      yaxis=dict(title='Imaginary', gridcolor='#f0f0f0'),
                      legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'))
    return fig

# ============================================
# MAIN APP
# ============================================

st.markdown("""
    <div class="main-header">
        <h1>🎛️ Interactive Frequency Response Explorer</h1>
        <p>An educational tool that constructs Bode and Nyquist plots step by step</p>
    </div>
""", unsafe_allow_html=True)

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = TransferFunctionAnalyzer()
    st.session_state.loaded = False

with st.sidebar:
    st.header("📥 Input Methods")
    method = st.radio("Choose Method", ["Direct Entry", "Builder Interface", "Example Library"])
    st.divider()
    
    if method == "Direct Entry":
        st.subheader("Transfer Function")
        num = st.text_input("Numerator:", value="100")
        den = st.text_input("Denominator:", value="1, 30, 200")
        if st.button("🚀 Generate Plots", use_container_width=True):
            if st.session_state.analyzer.parse(num, den):
                st.session_state.loaded = True
                st.success("✅ Loaded!")
    
    elif method == "Builder Interface":
        st.subheader("Build Your System")
        gain = st.number_input("Gain:", value=100.0, step=0.1)
        
        st.write("**Zeros (enter positive for LHP):**")
        zeros_input = []
        num_zeros = st.number_input("Number of zeros:", min_value=0, max_value=5, value=0, step=1)
        for i in range(num_zeros):
            z = st.text_input(f"Zero {i+1}:", value="")
            zeros_input.append(z)
        
        st.write("**Poles (enter positive for LHP):**")
        poles_input = []
        num_poles = st.number_input("Number of poles:", min_value=0, max_value=5, value=2, step=1)
        for i in range(num_poles):
            p = st.text_input(f"Pole {i+1}:", value="10" if i == 0 else "20")
            poles_input.append(p)
        
        if st.button("🏗️ Build & Generate", use_container_width=True):
            if st.session_state.analyzer.parse_from_builder(gain, zeros_input, poles_input):
                st.session_state.loaded = True
                st.success("✅ Built!")
    
    else:
        st.subheader("Pre-built Examples")
        examples = {
            "🔷 2 Poles": {"num": "100", "den": "1, 30, 200"},
            "🔶 3 Poles": {"num": "1", "den": "1, 6, 11, 6"},
            "🔄 Integrator": {"num": "1", "den": "1, 0"},
            "🔁 Double Integrator": {"num": "1", "den": "1, 0, 0"},
            "⚡ High Gain": {"num": "1000", "den": "1, 30, 200"},
            "⚠️ RHP Zero": {"num": "1, -1", "den": "1, 3, 2"},
            "❌ Unstable": {"num": "1", "den": "1, -1, 100"},
            "🎯 PID": {"num": "100, 10, 1", "den": "1, 0"}
        }
        for name, tf in examples.items():
            if st.button(f"{name}", use_container_width=True):
                if st.session_state.analyzer.parse(tf["num"], tf["den"]):
                    st.session_state.loaded = True
                    st.success(f"✅ {name} loaded!")

if st.session_state.loaded:
    analyzer = st.session_state.analyzer
    data = analyzer.get_bode_data()
    fr = data['frequency_response']
    asym = data['asymptotic']
    margins = data['margins']
    pz = data['pole_zero']
    standard_gain = data['standard_gain']
    
    # Display transfer function
    num_str = " + ".join([f"{c:.3f}s^{i}" for i, c in enumerate(reversed(analyzer.num)) if abs(c) > 1e-10])
    den_str = " + ".join([f"{c:.3f}s^{i}" for i, c in enumerate(reversed(analyzer.den)) if abs(c) > 1e-10])
    st.latex(f"H(s) = \\frac{{{num_str}}}{{{den_str}}}")
    
    # Show standard gain info
    gain_db = 20 * np.log10(abs(standard_gain) + 1e-10)
    st.caption(f"**Standard Gain:** K' = {standard_gain:.4f} → {gain_db:.2f} dB")
    
    # Debug info in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔍 Debug Info")
        st.markdown(f"**Min Magnitude:** {margins['min_mag']:.2f} dB")
        st.markdown(f"**Max Magnitude:** {margins['max_mag']:.2f} dB")
        st.markdown(f"**Min Phase:** {margins['min_phase']:.2f}°")
        st.markdown(f"**Max Phase:** {margins['max_phase']:.2f}°")
        st.markdown(f"**Standard Gain K':** {standard_gain:.4f}")
        st.markdown(f"**Origin Poles:** {data['origin_poles']}")
        st.markdown(f"**Origin Zeros:** {data['origin_zeros']}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Bode Plot", "📝 Derivation", "🌀 Nyquist", "📍 Pole-Zero"])
    
    with tab1:
        show_contrib = st.checkbox("Show Individual Pole/Zero Contributions", value=True)
        st.plotly_chart(create_bode_plot_with_contributions(data, show_contrib), use_container_width=True)
        
        st.subheader("📊 System Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        gm = margins['gain_margin_dB']
        pm = margins['phase_margin_deg']
        
        with col1:
            if margins['marginal']:
                st.metric("Stability", "⚠️ Marginal")
            else:
                st.metric("Stability", "✅ Stable" if margins['stable'] else "❌ Unstable")
        with col2:
            st.metric("Gain Margin", f"{gm:.2f} dB" if gm is not None else "∞")
        with col3:
            st.metric("Phase Margin", f"{pm:.2f}°" if pm is not None else "∞")
        with col4:
            st.metric("GM Freq", f"{margins['gm_freq']:.2f}" if margins['gm_freq'] else "None")
        with col5:
            st.metric("PM Freq", f"{margins['pm_freq']:.2f}" if margins['pm_freq'] else "None")
        
        # Interpretation with explanation
        st.subheader("📖 Interpretation")
        
        # Show margin calculation details
        if margins['gm_freq'] is not None:
            st.markdown(f"**Gain Margin Calculation:**")
            st.markdown(f"- Phase crossover at ω = {margins['gm_freq']:.4f} rad/s")
            st.markdown(f"- Magnitude at phase crossover = {gm:.2f} dB")
            st.markdown(f"- **GM = -({gm:.2f}) = {abs(gm):.2f} dB**")
        
        if margins['pm_freq'] is not None:
            st.markdown(f"**Phase Margin Calculation:**")
            st.markdown(f"- Gain crossover at ω = {margins['pm_freq']:.4f} rad/s")
            st.markdown(f"- Phase at gain crossover = {margins['phase_margin_deg'] - 180:.2f}°")
            st.markdown(f"- **PM = 180° + ({margins['phase_margin_deg'] - 180:.2f}°) = {pm:.2f}°**")
        
        if margins['stable']:
            if pm is not None and gm is not None:
                if pm > 60 and gm > 10:
                    st.success("✅ **Very Stable** - Excellent stability margins")
                elif pm > 30 and gm > 6:
                    st.info("📊 **Adequately Stable** - Acceptable for most applications")
                else:
                    st.warning("⚠️ **Poor Stability Margins** - System may be oscillatory")
            else:
                if margins['min_phase'] > -180:
                    st.info("📊 **Stable** - Phase never reaches -180°, infinite gain margin")
                elif margins['max_mag'] < 0:
                    st.info("📊 **Stable** - Magnitude never crosses 0dB, infinite phase margin")
                else:
                    st.info("📊 **Stable** - System is stable")
        else:
            if margins['marginal']:
                st.warning("⚠️ **Marginally Stable** - System has poles on imaginary axis")
            else:
                st.error("❌ **UNSTABLE** - System has poles in right half-plane!")
    
    with tab2:
        st.subheader("📐 Mathematical Derivation")
        
        for step in data['derivation']:
            st.markdown(f"### {step['title']}")
            st.markdown(step['content'])
            st.caption(step['explanation'])
            st.markdown("---")
    
    with tab3:
        st.plotly_chart(create_nyquist_plot(fr), use_container_width=True)
        idx = st.slider("Frequency Index", 0, len(fr['omega'])-1, len(fr['omega'])//2)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ω", f"{fr['omega'][idx]:.2f} rad/s")
        col2.metric("|H|", f"{fr['mag_db'][idx]:.2f} dB")
        col3.metric("Phase", f"{fr['phase_deg'][idx]:.1f}°")
        col4.metric("H(jω)", f"{fr['real'][idx]:.3f} + {fr['imag'][idx]:.3f}i")
    
    with tab4:
        st.plotly_chart(create_pole_zero_plot(pz), use_container_width=True)
        if margins['marginal']:
            st.warning("⚠️ Marginally Stable - Poles on imaginary axis")
        elif margins['stable']:
            st.success("✅ Stable - All poles in left half-plane")
        else:
            st.error("❌ Unstable - Poles in right half-plane")

else:
    st.info("👈 Load a transfer function from the sidebar to begin!")
    
    st.markdown("""
    ### 🎯 How to use the Interactive Frequency Response Explorer
    
    1. **Choose an input method** from the sidebar
    2. **Enter your transfer function** or select an example
    3. **Explore the plots** and **mathematical derivation**
    
    ### 📚 Features
    
    - 📈 **Bode Plots** with individual pole/zero contributions
    - 📝 **Mathematical Derivation** showing every step
    - 🌀 **Nyquist Plots** with interactive frequency slider
    - 📍 **Pole-Zero Maps** with stability analysis
    - 📊 **Frequency Response Tables** with actual values
    - 🎓 **Margin Calculation** showing the actual math
    
    ### 💡 Try These Examples
    
    | System | Numerator | Denominator | Expected |
    |--------|-----------|-------------|----------|
    | 2 Poles | `100` | `1, 30, 200` | ∞ margins |
    | High Gain | `1000` | `1, 30, 200` | Finite PM |
    | Integrator | `1` | `1, 0` | PM = 90° |
    | Double Integrator | `1` | `1, 0, 0` | GM = 0 dB, PM = 0° |
    """)

if __name__ == "__main__":
    pass
