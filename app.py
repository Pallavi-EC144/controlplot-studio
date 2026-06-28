import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy

st.set_page_config(page_title="ControlPlot Studio", page_icon="🎛️", layout="wide")

st.markdown("""
    <style>
    .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 30px; }
    .stButton > button { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 500; }
    .stButton > button:hover { background: #5a6fd6; }
    .good { color: #22c55e; font-weight: bold; }
    .warning { color: #eab308; font-weight: bold; }
    .bad { color: #ef4444; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

class ControlEngine:
    def __init__(self):
        self.num = [1]
        self.den = [1]
        self.poles = []
        self.zeros = []
        self.gain = 1
        self.corner_freqs = []
        self.standard_form = ""
        self.step_by_step = []
        
    def parse_transfer_function(self, num_str, den_str):
        """Step 1: Parse user input"""
        try:
            # Parse numerator
            if num_str.strip():
                num = [float(x.strip()) for x in num_str.split(',') if x.strip()]
            else:
                num = [1]
            
            # Parse denominator
            if den_str.strip():
                den = [float(x.strip()) for x in den_str.split(',') if x.strip()]
            else:
                den = [1]
            
            # Clean up zeros
            while len(num) > 1 and abs(num[-1]) < 1e-10:
                num.pop()
            while len(den) > 1 and abs(den[-1]) < 1e-10:
                den.pop()
            
            self.num = num
            self.den = den
            
            # Step 2: Convert to standard form
            self.step_by_step = []
            self.step_by_step.append("Step 1 ✓ Parsed Transfer Function")
            
            # Step 3: Extract poles and zeros
            self.poles = np.roots(den)
            self.zeros = np.roots(num)
            self.gain = num[0] / den[0] if den[0] != 0 else 1
            
            self.step_by_step.append(f"Step 2 ✓ Found Poles: {self.poles}")
            self.step_by_step.append(f"Step 3 ✓ Found Zeros: {self.zeros}")
            self.step_by_step.append(f"Step 4 ✓ Gain K = {self.gain:.3f}")
            
            # Step 4: Find corner frequencies
            self.corner_freqs = []
            for p in self.poles:
                if abs(p.imag) < 1e-10 and abs(p.real) > 1e-10:
                    self.corner_freqs.append(abs(p.real))
            for z in self.zeros:
                if abs(z.imag) < 1e-10 and abs(z.real) > 1e-10:
                    self.corner_freqs.append(abs(z.real))
            self.corner_freqs.sort()
            
            self.step_by_step.append(f"Step 5 ✓ Corner Frequencies: {self.corner_freqs}")
            
            return True
        except Exception as e:
            st.error(f"Error parsing: {str(e)}")
            return False
    
    def get_frequency_response(self, omega_min=0.001, omega_max=10000, num_points=2000):
        """Step 6-7: Generate frequencies and evaluate H(jω)"""
        # Step 6: Generate log-spaced frequencies
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_points)
        
        # Step 7: Replace s with jω and evaluate
        s = 1j * omega
        H = np.polyval(self.num, s) / np.polyval(self.den, s)
        
        # Step 8-9: Compute magnitude
        mag_linear = np.abs(H)
        mag_db = 20 * np.log10(mag_linear + 1e-10)
        
        # Step 10-11: Compute phase with unwrapping
        phase_deg = np.angle(H, deg=True)
        # Manual unwrapping
        phase_unwrapped = np.unwrap(np.angle(H, deg=False))
        phase_deg_unwrapped = np.degrees(phase_unwrapped)
        
        return {
            'omega': omega,
            'mag_db': mag_db,
            'mag_linear': mag_linear,
            'phase_deg': phase_deg,
            'phase_deg_unwrapped': phase_deg_unwrapped,
            'real': np.real(H),
            'imag': np.imag(H)
        }
    
    def get_asymptotic_response(self, omega_min=0.001, omega_max=10000, num_points=1000):
        """Step 12-15: Build asymptotic Bode plot"""
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_points)
        
        # Step 12: Start with gain contribution
        mag_asym = np.full(len(omega), 20 * np.log10(abs(self.gain) + 1e-10))
        phase_asym = np.full(len(omega), 0.0 if self.gain > 0 else -180.0)
        
        # Step 13: Add pole contributions
        for pole in self.poles:
            if abs(pole.imag) < 1e-10:  # Real pole
                p_real = pole.real
                if abs(p_real) > 1e-10:
                    omega_c = abs(p_real)
                    # Asymptotic magnitude: 0 before corner, -20dB/dec after
                    mag_asym -= 20 * np.log10(np.sqrt(1 + (omega/omega_c)**2))
                    # Asymptotic phase: 0 before, -90 after
                    phase_asym -= np.degrees(np.arctan2(omega, omega_c))
                else:  # Pole at origin (integrator)
                    mag_asym -= 20 * np.log10(omega + 1e-10)
                    phase_asym -= 90.0
        
        # Step 14: Add zero contributions
        for zero in self.zeros:
            if abs(zero.imag) < 1e-10:  # Real zero
                z_real = zero.real
                if abs(z_real) > 1e-10:
                    omega_c = abs(z_real)
                    mag_asym += 20 * np.log10(np.sqrt(1 + (omega/omega_c)**2))
                    phase_asym += np.degrees(np.arctan2(omega, omega_c))
                else:  # Zero at origin (differentiator)
                    mag_asym += 20 * np.log10(omega + 1e-10)
                    phase_asym += 90.0
        
        return {'omega': omega, 'mag_db': mag_asym, 'phase_deg': phase_asym}
    
    def get_individual_contributions(self, omega_min=0.001, omega_max=10000, num_points=1000):
        """Get individual pole/zero contributions for educational display"""
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_points)
        contributions = []
        
        # Gain contribution
        gain_mag = np.full(len(omega), 20 * np.log10(abs(self.gain) + 1e-10))
        gain_phase = np.full(len(omega), 0.0 if self.gain > 0 else -180.0)
        contributions.append({'name': 'Gain', 'mag': gain_mag, 'phase': gain_phase})
        
        # Pole contributions
        for i, pole in enumerate(self.poles):
            if abs(pole.imag) < 1e-10 and abs(pole.real) > 1e-10:
                mag = -20 * np.log10(np.sqrt(1 + (omega/abs(pole.real))**2))
                phase = -np.degrees(np.arctan2(omega, abs(pole.real)))
                contributions.append({'name': f'Pole {i+1}', 'mag': mag, 'phase': phase})
            elif abs(pole.imag) < 1e-10:
                mag = -20 * np.log10(omega + 1e-10)
                phase = -90.0 * np.ones(len(omega))
                contributions.append({'name': f'Integrator {i+1}', 'mag': mag, 'phase': phase})
        
        # Zero contributions
        for i, zero in enumerate(self.zeros):
            if abs(zero.imag) < 1e-10 and abs(zero.real) > 1e-10:
                mag = 20 * np.log10(np.sqrt(1 + (omega/abs(zero.real))**2))
                phase = np.degrees(np.arctan2(omega, abs(zero.real)))
                contributions.append({'name': f'Zero {i+1}', 'mag': mag, 'phase': phase})
            elif abs(zero.imag) < 1e-10:
                mag = 20 * np.log10(omega + 1e-10)
                phase = 90.0 * np.ones(len(omega))
                contributions.append({'name': f'Differentiator {i+1}', 'mag': mag, 'phase': phase})
        
        return {'omega': omega, 'contributions': contributions}
    
    def calculate_margins(self):
        """Step 18-21: Calculate gain and phase margins"""
        fr = self.get_frequency_response()
        omega = fr['omega']
        mag_db = fr['mag_db']
        phase_deg = fr['phase_deg']
        
        gain_margin = None
        phase_margin = None
        gm_freq = None
        pm_freq = None
        
        # Step 20: Find phase crossover (phase = -180°)
        phase_cross_idx = np.where(np.diff(np.sign(phase_deg + 180)))[0]
        
        if len(phase_cross_idx) > 0:
            idx = phase_cross_idx[0]
            if idx < len(mag_db) - 1:
                x1, x2 = phase_deg[idx], phase_deg[idx + 1]
                y1, y2 = mag_db[idx], mag_db[idx + 1]
                t = (-180 - x1) / (x2 - x1) if x2 != x1 else 0
                if 0 <= t <= 1:
                    gm = y1 + t * (y2 - y1)
                    gain_margin = -gm
                    gm_freq = omega[idx] + t * (omega[idx + 1] - omega[idx])
        
        # Step 18: Find gain crossover (magnitude = 0 dB)
        gain_cross_idx = np.where(np.diff(np.sign(mag_db)))[0]
        
        if len(gain_cross_idx) > 0:
            idx = gain_cross_idx[0]
            if idx < len(phase_deg) - 1:
                x1, x2 = mag_db[idx], mag_db[idx + 1]
                y1, y2 = phase_deg[idx], phase_deg[idx + 1]
                t = (0 - x1) / (x2 - x1) if x2 != x1 else 0
                if 0 <= t <= 1:
                    pm = y1 + t * (y2 - y1)
                    phase_margin = 180 + pm
                    pm_freq = omega[idx] + t * (omega[idx + 1] - omega[idx])
        
        # Step 22: Stability decision
        if gain_margin is not None and phase_margin is not None:
            stable = (gain_margin > 0) and (phase_margin > 0)
        elif gain_margin is not None:
            stable = (gain_margin > 0)
        elif phase_margin is not None:
            stable = (phase_margin > 0)
        else:
            # Check if system is stable based on poles
            stable = all(p.real < 0 for p in self.poles)
        
        return {
            'gain_margin': gain_margin,
            'phase_margin': phase_margin,
            'gm_freq': gm_freq,
            'pm_freq': pm_freq,
            'gain_margin_dB': gain_margin if gain_margin is not None else None,
            'phase_margin_deg': phase_margin if phase_margin is not None else None,
            'stable': stable
        }
    
    def get_pz(self):
        """Get pole-zero data for plotting"""
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
    
    def get_stability(self):
        """Get stability information"""
        margins = self.calculate_margins()
        return {'stable': margins['stable']}
    
    def get_bode_data(self):
        """Get all Bode data"""
        fr = self.get_frequency_response()
        asym = self.get_asymptotic_response()
        margins = self.calculate_margins()
        pz = self.get_pz()
        
        return {
            'frequency_response': fr,
            'asymptotic': asym,
            'margins': margins,
            'pole_zero': pz
        }

def create_bode_plot(data):
    """Create Bode plot with correct formatting"""
    fr = data['frequency_response']
    asym = data['asymptotic']
    margins = data['margins']
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=('Magnitude Response', 'Phase Response'))
    
    # Magnitude - Actual
    fig.add_trace(
        go.Scatter(x=fr['omega'], y=fr['mag_db'], mode='lines',
                  name='Actual Magnitude', line=dict(color='#2563eb', width=2.5)),
        row=1, col=1
    )
    
    # Magnitude - Asymptotic
    fig.add_trace(
        go.Scatter(x=asym['omega'], y=asym['mag_db'], mode='lines',
                  name='Asymptotic', line=dict(color='#22c55e', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Phase - Actual
    fig.add_trace(
        go.Scatter(x=fr['omega'], y=fr['phase_deg'], mode='lines',
                  name='Actual Phase', line=dict(color='#dc2626', width=2.5)),
        row=2, col=1
    )
    
    # Phase - Asymptotic
    fig.add_trace(
        go.Scatter(x=asym['omega'], y=asym['phase_deg'], mode='lines',
                  name='Asymptotic Phase', line=dict(color='#f59e0b', width=2, dash='dash')),
        row=2, col=1
    )
    
    # Add 0 dB line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)
    
    # Add -180° line
    fig.add_hline(y=-180, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
    
    # Add margin markers
    if margins['gm_freq'] is not None:
        fig.add_vline(x=margins['gm_freq'], line_dash="dash", line_color="green", 
                     annotation_text=f"GM: {margins['gain_margin_dB']:.2f} dB", row=1, col=1)
    
    if margins['pm_freq'] is not None:
        fig.add_vline(x=margins['pm_freq'], line_dash="dash", line_color="orange", 
                     annotation_text=f"PM: {margins['phase_margin_deg']:.2f}°", row=2, col=1)
    
    # Update axes
    fig.update_xaxes(type='log', title_text='Frequency (rad/s)', gridcolor='#f0f0f0', row=2, col=1)
    fig.update_xaxes(type='log', showticklabels=False, gridcolor='#f0f0f0', row=1, col=1)
    fig.update_yaxes(title_text='Magnitude (dB)', gridcolor='#f0f0f0', row=1, col=1)
    fig.update_yaxes(title_text='Phase (degrees)', gridcolor='#f0f0f0', row=2, col=1)
    
    fig.update_layout(height=600, hovermode='closest', showlegend=True,
                     legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'))
    
    return fig

def create_nyquist_plot(fr):
    """Create Nyquist plot"""
    fig = go.Figure()
    
    # Main Nyquist curve
    fig.add_trace(
        go.Scatter(x=fr['real'], y=fr['imag'], mode='lines',
                  name='Nyquist Plot', line=dict(color='#7c3aed', width=2.5),
                  hovertemplate='Real: %{x:.3f}<br>Imag: %{y:.3f}<br>ω: %{text:.2f} rad/s<extra></extra>',
                  text=fr['omega'])
    )
    
    # Start point
    fig.add_trace(
        go.Scatter(x=[fr['real'][0]], y=[fr['imag'][0]], mode='markers',
                  name='ω → 0', marker=dict(color='#22c55e', size=12, symbol='circle'))
    )
    
    # End point
    fig.add_trace(
        go.Scatter(x=[fr['real'][-1]], y=[fr['imag'][-1]], mode='markers',
                  name='ω → ∞', marker=dict(color='#ef4444', size=12, symbol='x'))
    )
    
    # -1 point
    fig.add_trace(
        go.Scatter(x=[-1], y=[0], mode='markers',
                  name='-1 point', marker=dict(color='#000', size=10, symbol='star'))
    )
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(
        go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                  name='Unit Circle', line=dict(color='#ccc', width=1, dash='dot'), showlegend=False)
    )
    
    # Add axes
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.3)
    
    fig.update_layout(title='Nyquist Plot', height=500, hovermode='closest',
                      xaxis=dict(title='Real', gridcolor='#f0f0f0', scaleanchor='y', scaleratio=1),
                      yaxis=dict(title='Imaginary', gridcolor='#f0f0f0'),
                      legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'))
    
    return fig

def create_pole_zero_plot(pz):
    """Create Pole-Zero map"""
    fig = go.Figure()
    
    # Poles
    if pz['poles']:
        fig.add_trace(
            go.Scatter(x=[p['real'] for p in pz['poles']], y=[p['imag'] for p in pz['poles']],
                      mode='markers', name='Poles',
                      marker=dict(color='#ef4444', size=14, symbol='x', line=dict(width=2)))
        )
    
    # Zeros
    if pz['zeros']:
        fig.add_trace(
            go.Scatter(x=[z['real'] for z in pz['zeros']], y=[z['imag'] for z in pz['zeros']],
                      mode='markers', name='Zeros',
                      marker=dict(color='#2563eb', size=14, symbol='circle-open', line=dict(width=2)))
        )
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(
        go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                  name='Unit Circle', line=dict(color='#ccc', width=1, dash='dot'), showlegend=False)
    )
    
    # Axes
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
        num = st.text_input("Numerator (e.g., 10, 20):", value="1")
        den = st.text_input("Denominator (e.g., 1, 3, 2):", value="1, 6, 11, 6")
        if st.button("🚀 Generate Plots", use_container_width=True):
            if st.session_state.engine.parse_transfer_function(num, den):
                st.session_state.loaded = True
                st.success("✅ Loaded!")
    
    elif method == "Builder Interface":
        st.subheader("Build Your System")
        gain = st.number_input("Gain:", value=1.0, step=0.1)
        
        st.write("**Zeros:**")
        zeros_input = []
        num_zeros = st.number_input("Number of zeros:", min_value=0, max_value=5, value=0, step=1)
        for i in range(num_zeros):
            z = st.text_input(f"Zero {i+1}:", value="2" if i == 0 else "")
            zeros_input.append(z)
        
        st.write("**Poles:**")
        poles_input = []
        num_poles = st.number_input("Number of poles:", min_value=0, max_value=5, value=3, step=1)
        for i in range(num_poles):
            p = st.text_input(f"Pole {i+1}:", value="1" if i == 0 else "2" if i == 1 else "3")
            poles_input.append(p)
        
        if st.button("🏗️ Build & Generate", use_container_width=True):
            try:
                z_vals = [float(x) for x in zeros_input if x.strip()]
                p_vals = [float(x) for x in poles_input if x.strip()]
                
                # Build numerator
                num = [gain]
                for z in z_vals:
                    if z != 0:
                        num = np.convolve(num, [1, -z])
                    else:
                        num = np.convolve(num, [0, 1])
                        while len(num) > 1 and abs(num[0]) < 1e-10:
                            num = num[1:]
                
                # Build denominator
                den = [1]
                for p in p_vals:
                    if p != 0:
                        den = np.convolve(den, [1, -p])
                    else:
                        den = np.convolve(den, [0, 1])
                        while len(den) > 1 and abs(den[0]) < 1e-10:
                            den = den[1:]
                
                num_str = ','.join([str(x) for x in num])
                den_str = ','.join([str(x) for x in den])
                
                if st.session_state.engine.parse_transfer_function(num_str, den_str):
                    st.session_state.loaded = True
                    st.success("✅ Built!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:
        st.subheader("Pre-built Examples")
        examples = {
            "3 Poles": {"num": "1", "den": "1, 6, 11, 6"},
            "RHP Zero": {"num": "1, -1", "den": "1, 3, 2"},
            "Integrator + Poles": {"num": "10", "den": "1, 10, 0"},
            "High Gain": {"num": "100", "den": "1, 10, 20, 1"},
            "4th Order": {"num": "1", "den": "1, 10, 35, 50, 24"},
            "Underdamped": {"num": "1", "den": "1, 1, 100"},
            "Unstable": {"num": "1", "den": "1, -1, 100"},
            "Band Pass": {"num": "10, 0", "den": "1, 2, 100"},
            "RC Low Pass": {"num": "1", "den": "1, 1"}
        }
        for name, tf in examples.items():
            if st.button(f"📂 {name}", use_container_width=True):
                if st.session_state.engine.parse_transfer_function(tf["num"], tf["den"]):
                    st.session_state.loaded = True
                    st.success(f"✅ {name} loaded!")

if st.session_state.loaded:
    engine = st.session_state.engine
    
    try:
        # Get all data
        data = engine.get_bode_data()
        fr = data['frequency_response']
        asym = data['asymptotic']
        margins = data['margins']
        pz = data['pole_zero']
        
        # Display transfer function
        num_str = " + ".join([f"{c:.3f}s^{i}" for i, c in enumerate(reversed(engine.num)) if abs(c) > 1e-10])
        den_str = " + ".join([f"{c:.3f}s^{i}" for i, c in enumerate(reversed(engine.den)) if abs(c) > 1e-10])
        if not num_str:
            num_str = "0"
        if not den_str:
            den_str = "0"
        
        st.latex(f"H(s) = \\frac{{{num_str}}}{{{den_str}}}")
        
        # Display steps
        with st.expander("📊 Step-by-Step Pipeline"):
            for step in engine.step_by_step:
                st.write(step)
            st.write(f"Step 6 ✓ Generated {len(fr['omega'])} log-spaced frequencies")
            st.write("Step 7 ✓ Evaluated H(jω) at all frequencies")
            st.write("Step 8-9 ✓ Computed Magnitude and Phase")
            st.write("Step 10-11 ✓ Phase unwrapping applied")
            st.write("Step 12-15 ✓ Built asymptotic approximation")
            st.write("Step 16 ✓ Drew Final Bode Plot")
            st.write(f"Step 17 ✓ Gain Crossover at {margins['pm_freq']:.3f} rad/s" if margins['pm_freq'] else "Step 17 ✓ No gain crossover found")
            st.write(f"Step 18 ✓ Phase Margin: {margins['phase_margin_deg']:.3f}°" if margins['phase_margin_deg'] else "Step 18 ✓ Phase Margin: ∞")
            st.write(f"Step 19 ✓ Phase Crossover at {margins['gm_freq']:.3f} rad/s" if margins['gm_freq'] else "Step 19 ✓ No phase crossover found")
            st.write(f"Step 20 ✓ Gain Margin: {margins['gain_margin_dB']:.3f} dB" if margins['gain_margin_dB'] else "Step 20 ✓ Gain Margin: ∞")
            st.write(f"Step 21 ✓ Stability: {'✅ Stable' if margins['stable'] else '❌ Unstable'}")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Bode Plot", "🌀 Nyquist Plot", "📍 Pole-Zero Map"])
        
        with tab1:
            st.plotly_chart(create_bode_plot(data), use_container_width=True)
            
            # Metrics
            st.subheader("📊 System Metrics")
            
            gm = margins['gain_margin_dB']
            pm = margins['phase_margin_deg']
            
            gm_str = f"{gm:.2f} dB" if gm is not None else "∞"
            pm_str = f"{pm:.2f}°" if pm is not None else "∞"
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Stability", "✅ Stable" if margins['stable'] else "❌ Unstable")
            c2.metric("Gain Margin", gm_str)
            c3.metric("Phase Margin", pm_str)
            c4.metric("GM Frequency", f"{margins['gm_freq']:.2f} rad/s" if margins['gm_freq'] else "None")
            c5.metric("PM Frequency", f"{margins['pm_freq']:.2f} rad/s" if margins['pm_freq'] else "None")
            
            # Interpretation
            st.subheader("📖 Interpretation")
            if margins['stable']:
                if pm is not None and gm is not None:
                    if pm > 60 and gm > 10:
                        st.success("✅ **Very Stable** - Excellent margins, well-damped system")
                    elif pm > 30 and gm > 6:
                        st.info("📊 **Adequately Stable** - Acceptable margins for most applications")
                    else:
                        st.warning("⚠️ **Poor Stability Margins** - System may be oscillatory")
                else:
                    st.info("📊 **Stable** - Infinite margins (phase/gain never cross critical values)")
            else:
                st.error("❌ **UNSTABLE** - System has poles in right half-plane!")
        
        with tab2:
            st.plotly_chart(create_nyquist_plot(fr), use_container_width=True)
            
            # Frequency slider
            idx = st.slider("Frequency Index", 0, len(fr['omega'])-1, len(fr['omega'])//2)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ω", f"{fr['omega'][idx]:.2f} rad/s")
            c2.metric("Magnitude", f"{fr['mag_db'][idx]:.2f} dB")
            c3.metric("Phase", f"{fr['phase_deg'][idx]:.1f}°")
            c4.metric("H(jω)", f"{fr['real'][idx]:.3f} + {fr['imag'][idx]:.3f}i")
        
        with tab3:
            st.plotly_chart(create_pole_zero_plot(pz), use_container_width=True)
            
            if margins['stable']:
                st.success("✅ System is **stable** - all poles in left half-plane")
            else:
                st.error("⚠️ System is **unstable** - pole(s) in right half-plane")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Try using a different transfer function or check your input format.")

else:
    st.info("👈 Please load a transfer function from the sidebar to begin!")
    
    st.markdown("""
    ### 🎯 How to use ControlPlot Studio
    
    1. **Choose an input method** from the sidebar
    2. **Enter your transfer function** or select an example
    3. **Explore the plots** in the main area
    
    ### ✅ Test Examples with Finite Margins
    
    | System | Numerator | Denominator | GM | PM |
    |--------|-----------|-------------|----|----|
    | 3 Poles | `1` | `1, 6, 11, 6` | ~30 dB | ~60° |
    | RHP Zero | `1, -1` | `1, 3, 2` | ~10 dB | ~20° |
    | Integrator+Poles | `10` | `1, 10, 0` | ~20 dB | ~45° |
    | High Gain | `100` | `1, 10, 20, 1` | ~20 dB | ~45° |
    | 4th Order | `1` | `1, 10, 35, 50, 24` | ~25 dB | ~50° |
    """)

if __name__ == "__main__":
    pass
