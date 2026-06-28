import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="ControlPlot Studio",
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
    .step-box {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e8e8f0;
        margin: 8px 0;
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
    </style>
""", unsafe_allow_html=True)

class ControlEngine:
    def __init__(self):
        self.num = [1]
        self.den = [1]
        self.poles = []
        self.zeros = []
        self.gain = 1
        self.standard_gain = 1  # K after standardization
        self.corner_freqs = []
        self.derivation_steps = []
        
    def parse_transfer_function(self, num_str, den_str):
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
            self.gain = num[0] / den[0] if den[0] != 0 else 1
            
            # Calculate standard form gain K
            # K = original_gain * product(|zeros|) / product(|poles|)
            # Exclude poles/zeros at origin
            self.standard_gain = self.gain
            for z in self.zeros:
                if abs(z.real) > 1e-10:
                    self.standard_gain *= abs(z.real)
            for p in self.poles:
                if abs(p.real) > 1e-10:
                    self.standard_gain /= abs(p.real)
            
            self.corner_freqs = []
            for p in self.poles:
                if abs(p.imag) < 1e-10 and abs(p.real) > 1e-10:
                    self.corner_freqs.append(abs(p.real))
            for z in self.zeros:
                if abs(z.imag) < 1e-10 and abs(z.real) > 1e-10:
                    self.corner_freqs.append(abs(z.real))
            self.corner_freqs.sort()
            
            self._generate_derivation()
            return True
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return False
    
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
            
            return self.parse_transfer_function(num_str, den_str)
        except Exception as e:
            st.error(f"Error building: {str(e)}")
            return False
    
    def _generate_derivation(self):
        """Generate step-by-step mathematical derivation"""
        steps = []
        
        # Original transfer function
        num_str = " + ".join([f"{c}s^{i}" for i, c in enumerate(reversed(self.num)) if abs(c) > 1e-10])
        den_str = " + ".join([f"{c}s^{i}" for i, c in enumerate(reversed(self.den)) if abs(c) > 1e-10])
        steps.append(f"**Original Transfer Function:**\n$$H(s) = \\frac{{{num_str}}}{{{den_str}}}$$")
        
        # Standard form
        steps.append("**Standard Bode Form:**")
        # Build standard form string
        standard_num = []
        standard_den = []
        
        for z in self.zeros:
            if abs(z.real) > 1e-10:
                standard_num.append(f"(1 + s/{abs(z.real):.2f})")
            elif abs(z.real) < 1e-10:
                standard_num.append("s")
        
        for p in self.poles:
            if abs(p.real) > 1e-10:
                standard_den.append(f"(1 + s/{abs(p.real):.2f})")
            elif abs(p.real) < 1e-10:
                standard_den.append("s")
        
        if not standard_num:
            standard_num = ["1"]
        if not standard_den:
            standard_den = ["1"]
        
        num_str = " \\cdot ".join(standard_num)
        den_str = " \\cdot ".join(standard_den)
        
        steps.append(f"$$H(s) = {self.standard_gain:.3f} \\frac{{{num_str}}}{{{den_str}}}$$")
        
        # Gain
        gain_db = 20 * np.log10(abs(self.standard_gain) + 1e-10)
        steps.append(f"**Standard Gain:** K = {self.standard_gain:.3f} → {gain_db:.2f} dB")
        
        # Poles and zeros
        poles_str = ", ".join([f"{p.real:.2f}" if abs(p.imag) < 1e-10 else f"{p.real:.2f} ± {abs(p.imag):.2f}i" for p in self.poles])
        zeros_str = ", ".join([f"{z.real:.2f}" if abs(z.imag) < 1e-10 else f"{z.real:.2f} ± {abs(z.imag):.2f}i" for z in self.zeros])
        steps.append(f"**Poles:** {poles_str}")
        steps.append(f"**Zeros:** {zeros_str}")
        
        if self.corner_freqs:
            steps.append(f"**Corner Frequencies:** {', '.join([f'{f:.2f}' for f in self.corner_freqs])} rad/s")
        
        # Individual contributions
        steps.append("**Individual Contributions:**")
        
        steps.append(f"• **Gain:** {self.standard_gain:.3f} → {gain_db:.2f} dB (constant)")
        
        for i, p in enumerate(self.poles):
            if abs(p.imag) < 1e-10 and abs(p.real) > 1e-10:
                steps.append(f"• **Pole {i+1}:** at s = {p.real:.2f} → corner at {abs(p.real):.2f} rad/s, -20 dB/dec")
            elif abs(p.imag) < 1e-10:
                steps.append(f"• **Integrator:** at s = 0 → -20 dB/dec, -90° phase")
        
        for i, z in enumerate(self.zeros):
            if abs(z.imag) < 1e-10 and abs(z.real) > 1e-10:
                steps.append(f"• **Zero {i+1}:** at s = {z.real:.2f} → corner at {abs(z.real):.2f} rad/s, +20 dB/dec")
            elif abs(z.imag) < 1e-10:
                steps.append(f"• **Differentiator:** at s = 0 → +20 dB/dec, +90° phase")
        
        # Magnitude equation
        steps.append("**Magnitude Equation (s = jω):**")
        mag_parts = []
        mag_parts.append(str(abs(self.standard_gain)))
        for z in self.zeros:
            if abs(z.real) > 1e-10:
                mag_parts.append(f"\\sqrt{{1 + (ω/{abs(z.real):.2f})^2}}")
        for p in self.poles:
            if abs(p.real) > 1e-10:
                mag_parts.append(f"\\frac{{1}}{{\\sqrt{{1 + (ω/{abs(p.real):.2f})^2}}}}")
        
        if any(abs(p.real) < 1e-10 for p in self.poles):
            mag_parts.append("\\frac{1}{ω}")
        if any(abs(z.real) < 1e-10 for z in self.zeros):
            mag_parts.append("ω")
        
        mag_eq = " \\cdot ".join(mag_parts)
        steps.append(f"$$|H(jω)| = {mag_eq}$$")
        steps.append(f"$$|H|_{{\\text{{dB}}}} = 20\\log_{{10}}(|H(jω)|)$$")
        
        # Phase equation
        steps.append("**Phase Equation:**")
        phase_parts = []
        for z in self.zeros:
            if abs(z.real) > 1e-10:
                phase_parts.append(f"+ \\tan^{{-1}}(ω/{abs(z.real):.2f})")
        for p in self.poles:
            if abs(p.real) > 1e-10:
                phase_parts.append(f"- \\tan^{{-1}}(ω/{abs(p.real):.2f})")
        if any(abs(p.real) < 1e-10 for p in self.poles):
            phase_parts.append("- 90°")
        if any(abs(z.real) < 1e-10 for z in self.zeros):
            phase_parts.append("+ 90°")
        
        if phase_parts:
            steps.append(f"$$\phi(ω) = {' + '.join(phase_parts)}$$")
        
        self.derivation_steps = steps
    
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
        
        # Standard gain contribution
        gain_mag = np.full(len(omega), 20 * np.log10(abs(self.standard_gain) + 1e-10))
        gain_phase = np.full(len(omega), 0.0 if self.standard_gain > 0 else -180.0)
        contributions.append({'name': 'Gain', 'mag': gain_mag, 'phase': gain_phase, 'color': '#000000'})
        
        # Poles
        for i, p in enumerate(self.poles):
            if abs(p.imag) < 1e-10 and abs(p.real) > 1e-10:
                mag = -20 * np.log10(np.sqrt(1 + (omega/abs(p.real))**2))
                phase = -np.degrees(np.arctan2(omega, abs(p.real)))
                contributions.append({'name': f'Pole {i+1}', 'mag': mag, 'phase': phase, 'color': f'#{hex(200 - i*30)[2:]:0>2s}0000'})
            elif abs(p.imag) < 1e-10:
                mag = -20 * np.log10(omega + 1e-10)
                phase = -90.0 * np.ones(len(omega))
                contributions.append({'name': 'Integrator', 'mag': mag, 'phase': phase, 'color': '#8B0000'})
        
        # Zeros
        for i, z in enumerate(self.zeros):
            if abs(z.imag) < 1e-10 and abs(z.real) > 1e-10:
                mag = 20 * np.log10(np.sqrt(1 + (omega/abs(z.real))**2))
                phase = np.degrees(np.arctan2(omega, abs(z.real)))
                contributions.append({'name': f'Zero {i+1}', 'mag': mag, 'phase': phase, 'color': f'#0000{hex(200 - i*30)[2:]:0>2s}'})
            elif abs(z.imag) < 1e-10:
                mag = 20 * np.log10(omega + 1e-10)
                phase = 90.0 * np.ones(len(omega))
                contributions.append({'name': 'Differentiator', 'mag': mag, 'phase': phase, 'color': '#00008B'})
        
        return {'omega': omega, 'contributions': contributions}
    
    def get_asymptotic_response(self, omega_min=0.0001, omega_max=100000, num_points=1000):
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_points)
        mag_asym = np.full(len(omega), 20 * np.log10(abs(self.standard_gain) + 1e-10))
        phase_asym = np.full(len(omega), 0.0 if self.standard_gain > 0 else -180.0)
        
        for pole in self.poles:
            if abs(pole.imag) < 1e-10:
                p_real = pole.real
                if abs(p_real) > 1e-10:
                    omega_c = abs(p_real)
                    mag_asym -= 20 * np.log10(np.sqrt(1 + (omega/omega_c)**2))
                    phase_asym -= np.degrees(np.arctan2(omega, omega_c))
                else:
                    mag_asym -= 20 * np.log10(omega + 1e-10)
                    phase_asym -= 90.0
        
        for zero in self.zeros:
            if abs(zero.imag) < 1e-10:
                z_real = zero.real
                if abs(z_real) > 1e-10:
                    omega_c = abs(z_real)
                    mag_asym += 20 * np.log10(np.sqrt(1 + (omega/omega_c)**2))
                    phase_asym += np.degrees(np.arctan2(omega, omega_c))
                else:
                    mag_asym += 20 * np.log10(omega + 1e-10)
                    phase_asym += 90.0
        
        return {'omega': omega, 'mag_db': mag_asym, 'phase_deg': phase_asym}
    
    def calculate_margins(self):
        fr = self.get_frequency_response()
        omega = fr['omega']
        mag_db = fr['mag_db']
        phase_deg = fr['phase_deg']
        
        gain_margin = None
        phase_margin = None
        gm_freq = None
        pm_freq = None
        
        # --- FIX 1: Detect EXACT hits on 0 dB ---
        # Find where magnitude crosses 0 dB (with tolerance for exact hits)
        gain_cross_idx = np.where(
            (mag_db[:-1] > 0) & (mag_db[1:] <= 0)
        )[0]
        gain_cross_idx2 = np.where(
            (mag_db[:-1] < 0) & (mag_db[1:] >= 0)
        )[0]
        
        # Check for exact hits on 0 dB
        exact_zero_idx = np.where(np.isclose(mag_db, 0, atol=1e-6))[0]
        
        # Combine all gain crossover indices
        all_gain_cross = np.unique(np.concatenate([gain_cross_idx, gain_cross_idx2, exact_zero_idx]))
        all_gain_cross = all_gain_cross[all_gain_cross < len(mag_db) - 1]
        
        # --- FIX 2: Detect EXACT hits on -180° ---
        # Find where phase crosses -180° (with tolerance for exact hits)
        phase_cross_idx = np.where(
            (phase_deg[:-1] > -180) & (phase_deg[1:] <= -180)
        )[0]
        phase_cross_idx2 = np.where(
            (phase_deg[:-1] < -180) & (phase_deg[1:] >= -180)
        )[0]
        
        # Check for exact hits on -180°
        exact_180_idx = np.where(np.isclose(phase_deg, -180, atol=1e-3))[0]
        
        # Combine all phase crossover indices
        all_phase_cross = np.unique(np.concatenate([phase_cross_idx, phase_cross_idx2, exact_180_idx]))
        all_phase_cross = all_phase_cross[all_phase_cross < len(phase_deg) - 1]
        
        # Process phase crossover (for gain margin)
        if len(all_phase_cross) > 0:
            idx = all_phase_cross[0]
            if idx < len(mag_db) - 1:
                # Check if it's an exact hit
                if idx in exact_180_idx:
                    gain_margin = -mag_db[idx]
                    gm_freq = omega[idx]
                else:
                    x1, x2 = phase_deg[idx], phase_deg[idx + 1]
                    y1, y2 = mag_db[idx], mag_db[idx + 1]
                    if abs(x2 - x1) > 1e-10:
                        t = (-180 - x1) / (x2 - x1)
                        if 0 <= t <= 1:
                            gm = y1 + t * (y2 - y1)
                            gain_margin = -gm
                            gm_freq = omega[idx] + t * (omega[idx + 1] - omega[idx])
        
        # Process gain crossover (for phase margin)
        if len(all_gain_cross) > 0:
            idx = all_gain_cross[0]
            if idx < len(phase_deg) - 1:
                # Check if it's an exact hit
                if idx in exact_zero_idx:
                    phase_margin = 180 + phase_deg[idx]
                    pm_freq = omega[idx]
                else:
                    x1, x2 = mag_db[idx], mag_db[idx + 1]
                    y1, y2 = phase_deg[idx], phase_deg[idx + 1]
                    if abs(x2 - x1) > 1e-10:
                        t = (0 - x1) / (x2 - x1)
                        if 0 <= t <= 1:
                            pm = y1 + t * (y2 - y1)
                            phase_margin = 180 + pm
                            pm_freq = omega[idx] + t * (omega[idx + 1] - omega[idx])
        
        # Stability
        stable_poles = all(p.real < 1e-10 for p in self.poles)
        marginal_poles = any(abs(p.real) < 1e-10 and abs(p.imag) > 0 for p in self.poles)
        
        if gain_margin is not None and phase_margin is not None:
            stable = (gain_margin > 0) and (phase_margin > 0) and stable_poles
        elif gain_margin is not None:
            stable = (gain_margin > 0) and stable_poles
        elif phase_margin is not None:
            stable = (phase_margin > 0) and stable_poles
        else:
            stable = stable_poles
        
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
        asym = self.get_asymptotic_response()
        margins = self.calculate_margins()
        pz = self.get_pz()
        contrib = self.get_individual_contributions()
        
        return {
            'frequency_response': fr,
            'asymptotic': asym,
            'margins': margins,
            'pole_zero': pz,
            'contributions': contrib,
            'derivation': self.derivation_steps,
            'standard_gain': self.standard_gain
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
        <h1>🎛️ ControlPlot Studio</h1>
        <p>Interactive Control System Visualization Platform</p>
    </div>
""", unsafe_allow_html=True)

if 'engine' not in st.session_state:
    st.session_state.engine = ControlEngine()
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
            if st.session_state.engine.parse_transfer_function(num, den):
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
            if st.session_state.engine.parse_from_builder(gain, zeros_input, poles_input):
                st.session_state.loaded = True
                st.success("✅ Built!")
    
    else:
        st.subheader("Pre-built Examples")
        examples = {
            "2 Poles (s+10)(s+20)": {"num": "100", "den": "1, 30, 200"},
            "3 Poles (s+1)(s+2)(s+3)": {"num": "1", "den": "1, 6, 11, 6"},
            "Integrator + 2 Poles": {"num": "10", "den": "1, 10, 0"},
            "High Gain": {"num": "1000", "den": "1, 30, 200"},
            "4th Order": {"num": "1", "den": "1, 10, 35, 50, 24"},
            "Underdamped": {"num": "1", "den": "1, 1, 100"},
            "Unstable": {"num": "1", "den": "1, -1, 100"},
            "RHP Zero": {"num": "1, -1", "den": "1, 3, 2"},
            "Double Integrator": {"num": "1", "den": "1, 0, 0"}
        }
        for name, tf in examples.items():
            if st.button(f"📂 {name}", use_container_width=True):
                if st.session_state.engine.parse_transfer_function(tf["num"], tf["den"]):
                    st.session_state.loaded = True
                    st.success(f"✅ {name} loaded!")

if st.session_state.loaded:
    engine = st.session_state.engine
    data = engine.get_bode_data()
    fr = data['frequency_response']
    asym = data['asymptotic']
    margins = data['margins']
    pz = data['pole_zero']
    standard_gain = data['standard_gain']
    
    # Display transfer function
    num_str = " + ".join([f"{c:.3f}s^{i}" for i, c in enumerate(reversed(engine.num)) if abs(c) > 1e-10])
    den_str = " + ".join([f"{c:.3f}s^{i}" for i, c in enumerate(reversed(engine.den)) if abs(c) > 1e-10])
    st.latex(f"H(s) = \\frac{{{num_str}}}{{{den_str}}}")
    
    # Show standard gain info
    st.caption(f"**Standard Gain:** K = {standard_gain:.4f} → {20 * np.log10(abs(standard_gain) + 1e-10):.2f} dB")
    
    # Debug info in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔍 Debug Info")
        st.markdown(f"**Min Magnitude:** {margins['min_mag']:.2f} dB")
        st.markdown(f"**Max Magnitude:** {margins['max_mag']:.2f} dB")
        st.markdown(f"**Min Phase:** {margins['min_phase']:.2f}°")
        st.markdown(f"**Max Phase:** {margins['max_phase']:.2f}°")
        st.markdown(f"**Standard Gain K:** {standard_gain:.4f}")
        st.markdown(f"**Standard Gain dB:** {20 * np.log10(abs(standard_gain) + 1e-10):.2f} dB")
    
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
        
        # Interpretation
        if margins['marginal']:
            st.warning("⚠️ **Marginally Stable** - System has poles on imaginary axis")
        elif margins['stable']:
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
            st.error("❌ **UNSTABLE** - System has poles in right half-plane!")
    
    with tab2:
        st.subheader("📐 Mathematical Derivation")
        for step in engine.derivation_steps:
            if step.startswith("**") and step.endswith("**"):
                st.markdown(step)
            elif step.startswith("$"):
                st.markdown(step)
            else:
                st.text(step)
        
        # Frequency table
        st.subheader("📊 Frequency Response Table")
        freq_indices = np.logspace(0, np.log10(len(fr['omega'])-1), 10, dtype=int)
        freq_indices = np.unique(freq_indices)
        table_data = []
        for idx in freq_indices:
            table_data.append([
                f"{fr['omega'][idx]:.3f}",
                f"{fr['mag_db'][idx]:.2f}",
                f"{fr['phase_deg'][idx]:.1f}",
                f"{fr['real'][idx]:.3f}",
                f"{fr['imag'][idx]:.3f}"
            ])
        
        st.table({
            "ω (rad/s)": [row[0] for row in table_data],
            "|H| (dB)": [row[1] for row in table_data],
            "Phase (°)": [row[2] for row in table_data],
            "Real": [row[3] for row in table_data],
            "Imag": [row[4] for row in table_data]
        })
    
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
    ### 🎯 How to use ControlPlot Studio
    
    1. **Choose an input method** from the sidebar
    2. **Enter your transfer function** or select an example
    3. **Explore the plots** and **mathematical derivation**
    
    ### 📚 Features
    
    - 📈 **Bode Plots** with individual pole/zero contributions
    - 📝 **Mathematical Derivation** showing every step
    - 🌀 **Nyquist Plots** with interactive frequency slider
    - 📍 **Pole-Zero Maps** with stability analysis
    - 📊 **Frequency Response Tables** with actual values
    - 🎓 **Interpretation** explaining what the numbers mean
    """)

if __name__ == "__main__":
    pass
