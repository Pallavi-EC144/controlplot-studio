import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        
    def parse(self, num_str, den_str):
        try:
            num = [float(x.strip()) for x in num_str.split(',') if x.strip()] or [1]
            den = [float(x.strip()) for x in den_str.split(',') if x.strip()] or [1]
            self.num = num
            self.den = den
            self.gain = num[0]/den[0] if den[0] != 0 else 1
            self.poles = np.roots(den)
            self.zeros = np.roots(num)
            return True
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return False
    
    def get_freq_response(self, n=2000):
        omega = np.logspace(-3, 5, n)
        s = 1j * omega
        H = np.polyval(self.num, s) / np.polyval(self.den, s)
        return {
            'omega': omega,
            'mag_db': 20*np.log10(np.abs(H)+1e-10),
            'mag_linear': np.abs(H),
            'phase_deg': np.angle(H, deg=True),
            'real': np.real(H),
            'imag': np.imag(H)
        }
    
    def get_asymptotic(self, n=1000):
        omega = np.logspace(-3, 5, n)
        mag = np.full(n, 20*np.log10(abs(self.gain) + 1e-10))
        phase = np.full(n, 0.0 if self.gain > 0 else -180.0)
        
        for p in self.poles:
            if abs(p.imag) < 1e-10:
                p_real = p.real
                if abs(p_real) > 1e-10:
                    omega_c = abs(p_real)
                    mag -= 20 * np.log10(np.sqrt(1 + (omega/omega_c)**2))
                    phase -= np.degrees(np.arctan2(omega, omega_c))
                else:
                    mag -= 20 * np.log10(omega + 1e-10)
                    phase -= 90.0
        
        for z in self.zeros:
            if abs(z.imag) < 1e-10:
                z_real = z.real
                if abs(z_real) > 1e-10:
                    omega_c = abs(z_real)
                    mag += 20 * np.log10(np.sqrt(1 + (omega/omega_c)**2))
                    phase += np.degrees(np.arctan2(omega, omega_c))
                else:
                    mag += 20 * np.log10(omega + 1e-10)
                    phase += 90.0
        
        return {'omega': omega, 'mag_db': mag, 'phase_deg': phase}
    
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
    
    def get_stability(self):
        return {'stable': all(p.real < 0 for p in self.poles)}
    
    def calculate_margins(self):
        """Calculate gain and phase margins"""
        fr = self.get_freq_response()
        omega = fr['omega']
        mag_db = fr['mag_db']
        phase_deg = fr['phase_deg']
        mag_linear = fr['mag_linear']
        
        gain_margin = None
        phase_margin = None
        gm_freq = None
        pm_freq = None
        
        # Find phase crossover (where phase = -180°)
        phase_cross_idx = np.where(np.diff(np.sign(phase_deg + 180)))[0]
        
        if len(phase_cross_idx) > 0:
            idx = phase_cross_idx[0]
            if idx < len(mag_db) - 1:
                # Linear interpolation for more accuracy
                x1, x2 = phase_deg[idx], phase_deg[idx + 1]
                y1, y2 = mag_db[idx], mag_db[idx + 1]
                # Find where phase crosses -180
                t = (-180 - x1) / (x2 - x1)
                if 0 <= t <= 1:
                    gm = y1 + t * (y2 - y1)
                    gain_margin = -gm  # Gain margin is negative of magnitude at phase crossover
                    gm_freq = omega[idx] + t * (omega[idx + 1] - omega[idx])
        
        # Find gain crossover (where magnitude = 0 dB)
        gain_cross_idx = np.where(np.diff(np.sign(mag_db)))[0]
        
        if len(gain_cross_idx) > 0:
            idx = gain_cross_idx[0]
            if idx < len(phase_deg) - 1:
                # Linear interpolation for more accuracy
                x1, x2 = mag_db[idx], mag_db[idx + 1]
                y1, y2 = phase_deg[idx], phase_deg[idx + 1]
                # Find where magnitude crosses 0 dB
                t = (0 - x1) / (x2 - x1)
                if 0 <= t <= 1:
                    pm = y1 + t * (y2 - y1)
                    phase_margin = 180 + pm  # Phase margin
                    pm_freq = omega[idx] + t * (omega[idx + 1] - omega[idx])
        
        return {
            'gain_margin': gain_margin,
            'phase_margin': phase_margin,
            'gm_freq': gm_freq,
            'pm_freq': pm_freq,
            'gain_margin_dB': gain_margin if gain_margin is not None else None,
            'phase_margin_deg': phase_margin if phase_margin is not None else None
        }

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
        num = st.text_input("Numerator (e.g., 10, 20):", value="10, 20")
        den = st.text_input("Denominator (e.g., 1, 3, 2):", value="1, 3, 2")
        if st.button("🚀 Generate Plots", use_container_width=True):
            if st.session_state.engine.parse(num, den):
                st.session_state.loaded = True
                st.success("✅ Loaded!")
    
    elif method == "Builder Interface":
        st.subheader("Build Your System")
        gain = st.number_input("Gain:", value=10.0, step=0.1)
        
        st.write("**Zeros:**")
        zeros_input = []
        num_zeros = st.number_input("Number of zeros:", min_value=0, max_value=5, value=1, step=1)
        for i in range(num_zeros):
            z = st.text_input(f"Zero {i+1}:", value="2" if i == 0 else "")
            zeros_input.append(z)
        
        st.write("**Poles:**")
        poles_input = []
        num_poles = st.number_input("Number of poles:", min_value=0, max_value=5, value=2, step=1)
        for i in range(num_poles):
            p = st.text_input(f"Pole {i+1}:", value="1" if i == 0 else "5" if i == 1 else "")
            poles_input.append(p)
        
        if st.button("🏗️ Build & Generate", use_container_width=True):
            try:
                z_vals = [float(x) for x in zeros_input if x.strip()]
                p_vals = [float(x) for x in poles_input if x.strip()]
                
                num = [gain]
                for z in z_vals:
                    if z != 0:
                        num = np.convolve(num, [1, -z])
                    else:
                        num = np.convolve(num, [0, 1])
                        while len(num) > 1 and abs(num[0]) < 1e-10:
                            num = num[1:]
                
                den = [1]
                for p in p_vals:
                    if p != 0:
                        den = np.convolve(den, [1, -p])
                    else:
                        den = np.convolve(den, [0, 1])
                        while len(den) > 1 and abs(den[0]) < 1e-10:
                            den = den[1:]
                
                st.session_state.engine.num = num.tolist()
                st.session_state.engine.den = den.tolist()
                st.session_state.engine.poles = p_vals
                st.session_state.engine.zeros = z_vals
                st.session_state.engine.gain = gain
                st.session_state.loaded = True
                st.success("✅ Built!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:
        st.subheader("Pre-built Examples")
        examples = {
            "RC Low Pass": {"num": "1", "den": "1, 1"},
            "RC High Pass": {"num": "1, 0", "den": "1, 1"},
            "Second Order": {"num": "1", "den": "1, 1, 100"},
            "3 Poles": {"num": "1", "den": "1, 6, 11, 6"},
            "RHP Zero": {"num": "1, -1", "den": "1, 3, 2"},
            "Integrator + Poles": {"num": "10", "den": "1, 10, 0"},
            "High Gain": {"num": "100", "den": "1, 10, 20, 1"},
            "Unstable": {"num": "1", "den": "1, -1, 100"}
        }
        for name, tf in examples.items():
            if st.button(f"📂 {name}", use_container_width=True):
                if st.session_state.engine.parse(tf["num"], tf["den"]):
                    st.session_state.loaded = True
                    st.success(f"✅ {name} loaded!")

if st.session_state.loaded:
    engine = st.session_state.engine
    
    try:
        fr = engine.get_freq_response()
        asym = engine.get_asymptotic()
        pz = engine.get_pz()
        stab = engine.get_stability()
        margins = engine.calculate_margins()
        
        # Show current system
        num_str = " + ".join([f"{c}s^{i}" for i, c in enumerate(reversed(engine.num)) if abs(c) > 1e-10])
        den_str = " + ".join([f"{c}s^{i}" for i, c in enumerate(reversed(engine.den)) if abs(c) > 1e-10])
        if not num_str:
            num_str = "0"
        if not den_str:
            den_str = "0"
        st.latex(f"H(s) = \\frac{{{num_str}}}{{{den_str}}}")
        
        tab1, tab2, tab3 = st.tabs(["📈 Bode Plot", "🌀 Nyquist Plot", "📍 Pole-Zero Map"])
        
        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=('Magnitude Response', 'Phase Response'))
            
            fig.add_trace(go.Scatter(x=fr['omega'], y=fr['mag_db'], mode='lines',
                                     name='Actual Magnitude', line=dict(color='#2563eb', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=asym['omega'], y=asym['mag_db'], mode='lines',
                                     name='Asymptotic', line=dict(color='#22c55e', width=1.5, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=fr['omega'], y=fr['phase_deg'], mode='lines',
                                     name='Actual Phase', line=dict(color='#dc2626', width=2)), row=2, col=1)
            fig.add_trace(go.Scatter(x=asym['omega'], y=asym['phase_deg'], mode='lines',
                                     name='Asymptotic Phase', line=dict(color='#f59e0b', width=1.5, dash='dash')), row=2, col=1)
            
            # Add markers for margins
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
            fig.update_layout(height=600, hovermode='closest', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display margins with proper formatting
            st.subheader("📊 System Metrics")
            
            gm = margins['gain_margin_dB']
            pm = margins['phase_margin_deg']
            
            # Format margin strings
            gm_str = f"{gm:.2f} dB" if gm is not None else "∞"
            pm_str = f"{pm:.2f}°" if pm is not None else "∞"
            
            # Color code based on margins
            gm_color = "good" if gm is None or gm > 10 else "warning" if gm is None or gm > 6 else "bad"
            pm_color = "good" if pm is None or pm > 60 else "warning" if pm is None or pm > 30 else "bad"
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Stability", "✅ Stable" if stab['stable'] else "❌ Unstable")
            c2.metric("Gain Margin", gm_str)
            c3.metric("Phase Margin", pm_str)
            
            if margins['gm_freq'] is not None:
                c4.metric("GM Frequency", f"{margins['gm_freq']:.2f} rad/s")
            else:
                c4.metric("GM Frequency", "None")
            
            if margins['pm_freq'] is not None:
                c5.metric("PM Frequency", f"{margins['pm_freq']:.2f} rad/s")
            else:
                c5.metric("PM Frequency", "None")
            
            # Add interpretation
            st.subheader("📖 Interpretation")
            if stab['stable']:
                if pm is not None and gm is not None:
                    if pm > 60 and gm > 10:
                        st.success("✅ **Very Stable** - Good margins, well-damped system")
                    elif pm > 30 and gm > 6:
                        st.info("📊 **Adequately Stable** - Acceptable margins for most applications")
                    else:
                        st.warning("⚠️ **Poor Stability Margins** - System may be oscillatory")
                else:
                    st.info("📊 **Stable** - System has infinite margins (phase/gain never cross critical values)")
            else:
                st.error("❌ **UNSTABLE** - System has poles in right half-plane!")
        
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fr['real'], y=fr['imag'], mode='lines',
                                     name='Nyquist Plot', line=dict(color='#7c3aed', width=2)))
            fig.add_trace(go.Scatter(x=[fr['real'][0]], y=[fr['imag'][0]], mode='markers',
                                     name='ω → 0', marker=dict(color='#22c55e', size=12, symbol='circle')))
            fig.add_trace(go.Scatter(x=[fr['real'][-1]], y=[fr['imag'][-1]], mode='markers',
                                     name='ω → ∞', marker=dict(color='#ef4444', size=12, symbol='x')))
            
            # Mark -1 point for stability
            fig.add_trace(go.Scatter(x=[-1], y=[0], mode='markers',
                                     name='-1 point', marker=dict(color='#000', size=10, symbol='star')))
            
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                                     name='Unit Circle', line=dict(color='#ccc', width=1, dash='dot'), showlegend=False))
            
            fig.update_layout(title='Nyquist Plot', height=500, hovermode='closest',
                              xaxis=dict(title='Real', gridcolor='#f0f0f0', scaleanchor='y', scaleratio=1),
                              yaxis=dict(title='Imaginary', gridcolor='#f0f0f0'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Frequency slider
            idx = st.slider("Frequency", 0, len(fr['omega'])-1, len(fr['omega'])//2)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ω", f"{fr['omega'][idx]:.2f} rad/s")
            c2.metric("Magnitude", f"{fr['mag_db'][idx]:.2f} dB")
            c3.metric("Phase", f"{fr['phase_deg'][idx]:.1f}°")
            c4.metric("H(jω)", f"{fr['real'][idx]:.3f} + {fr['imag'][idx]:.3f}i")
        
        with tab3:
            fig = go.Figure()
            
            if pz['poles']:
                fig.add_trace(go.Scatter(x=[p['real'] for p in pz['poles']], y=[p['imag'] for p in pz['poles']],
                                         mode='markers', name='Poles',
                                         marker=dict(color='#ef4444', size=14, symbol='x')))
            if pz['zeros']:
                fig.add_trace(go.Scatter(x=[z['real'] for z in pz['zeros']], y=[z['imag'] for z in pz['zeros']],
                                         mode='markers', name='Zeros',
                                         marker=dict(color='#2563eb', size=14, symbol='circle-open')))
            
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                                     name='Unit Circle', line=dict(color='#ccc', width=1, dash='dot'), showlegend=False))
            
            fig.update_layout(title='Pole-Zero Map', height=500, hovermode='closest',
                              xaxis=dict(title='Real', gridcolor='#f0f0f0', scaleanchor='y', scaleratio=1),
                              yaxis=dict(title='Imaginary', gridcolor='#f0f0f0'))
            st.plotly_chart(fig, use_container_width=True)
            
            if stab['stable']:
                st.success("✅ System is **stable** - all poles in left half-plane")
            else:
                st.error("⚠️ System is **unstable** - pole(s) in right half-plane")
    
    except Exception as e:
        st.error(f"Error generating plots: {str(e)}")
        st.info("Try using a different transfer function or check your input format.")

else:
    st.info("👈 Please load a transfer function from the sidebar to begin!")
    
    st.markdown("""
    ### 🎯 How to use ControlPlot Studio
    
    1. **Choose an input method** from the sidebar
    2. **Enter your transfer function** or select an example
    3. **Explore the plots** in the main area
    
    ### 💡 Examples with Finite Margins
    
    | System | Numerator | Denominator | Expected PM | Expected GM |
    |--------|-----------|-------------|-------------|-------------|
    | 3 Poles | `1` | `1, 6, 11, 6` | ~60° | ~30 dB |
    | RHP Zero | `1, -1` | `1, 3, 2` | ~20° | ~10 dB |
    | Integrator + Poles | `10` | `1, 10, 0` | ~45° | ~20 dB |
    | High Gain | `100` | `1, 10, 20, 1` | ~45° | ~20 dB |
    """)
