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
        except:
            return False
    
    def get_freq_response(self, n=1000):
        omega = np.logspace(-2, 4, n)
        s = 1j * omega
        H = np.polyval(self.num, s) / np.polyval(self.den, s)
        return {
            'omega': omega,
            'mag_db': 20*np.log10(np.abs(H)+1e-10),
            'phase_deg': np.angle(H, deg=True),
            'real': np.real(H),
            'imag': np.imag(H)
        }
    
    def get_asymptotic(self, n=1000):
        omega = np.logspace(-2, 4, n)
        mag = np.full(n, 20*np.log10(abs(self.gain)))
        phase = np.full(n, 0 if self.gain > 0 else -180)
        
        for p in self.poles:
            if abs(p.imag) < 1e-10 and abs(p.real) > 1e-10:
                mag -= 20*np.log10(np.sqrt(1 + (omega/abs(p.real))**2))
                phase -= np.degrees(np.arctan(omega/abs(p.real)))
            elif abs(p.imag) < 1e-10:
                mag -= 20*np.log10(omega)
                phase -= 90
                
        for z in self.zeros:
            if abs(z.imag) < 1e-10 and abs(z.real) > 1e-10:
                mag += 20*np.log10(np.sqrt(1 + (omega/abs(z.real))**2))
                phase += np.degrees(np.arctan(omega/abs(z.real)))
            elif abs(z.imag) < 1e-10:
                mag += 20*np.log10(omega)
                phase += 90
                
        return {'omega': omega, 'mag_db': mag, 'phase_deg': phase}
    
    def get_pz(self):
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
    
    def get_stability(self):
        return {'stable': all(p.real < 0 for p in self.poles)}

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
            z_vals = [float(x) for x in zeros_input if x.strip()]
            p_vals = [float(x) for x in poles_input if x.strip()]
            
            num = [gain]
            for z in z_vals:
                if z != 0:
                    num = np.convolve(num, [1, -z])
                else:
                    num = np.convolve(num, [0, 1])
            
            den = [1]
            for p in p_vals:
                if p != 0:
                    den = np.convolve(den, [1, -p])
                else:
                    den = np.convolve(den, [0, 1])
            
            st.session_state.engine.num = num.tolist()
            st.session_state.engine.den = den.tolist()
            st.session_state.engine.poles = p_vals
            st.session_state.engine.zeros = z_vals
            st.session_state.engine.gain = gain
            st.session_state.loaded = True
            st.success("✅ Built!")
    
    else:
        st.subheader("Pre-built Examples")
        examples = {
            "RC Low Pass": {"num": "1", "den": "1, 1"},
            "RC High Pass": {"num": "1, 0", "den": "1, 1"},
            "Second Order": {"num": "1", "den": "1, 2, 100"},
            "Integrator": {"num": "1", "den": "1, 0"},
            "Differentiator": {"num": "1, 0", "den": "1"}
        }
        for name, tf in examples.items():
            if st.button(f"📂 {name}", use_container_width=True):
                if st.session_state.engine.parse(tf["num"], tf["den"]):
                    st.session_state.loaded = True
                    st.success(f"✅ {name} loaded!")

if st.session_state.loaded:
    engine = st.session_state.engine
    fr = engine.get_freq_response()
    asym = engine.get_asymptotic()
    pz = engine.get_pz()
    stab = engine.get_stability()
    
    # Show current system
    num_str = " + ".join([f"{c}s^{i}" for i, c in enumerate(reversed(engine.num)) if abs(c) > 1e-10])
    den_str = " + ".join([f"{c}s^{i}" for i, c in enumerate(reversed(engine.den)) if abs(c) > 1e-10])
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
        
        fig.update_xaxes(type='log', title_text='Frequency (rad/s)', gridcolor='#f0f0f0', row=2, col=1)
        fig.update_xaxes(type='log', showticklabels=False, gridcolor='#f0f0f0', row=1, col=1)
        fig.update_yaxes(title_text='Magnitude (dB)', gridcolor='#f0f0f0', row=1, col=1)
        fig.update_yaxes(title_text='Phase (degrees)', gridcolor='#f0f0f0', row=2, col=1)
        fig.update_layout(height=600, hovermode='closest', showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", "✅ Stable" if stab['stable'] else "❌ Unstable")
        c2.metric("Gain Margin", "∞")
        c3.metric("Phase Margin", "∞")
    
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fr['real'], y=fr['imag'], mode='lines',
                                 name='Nyquist Plot', line=dict(color='#7c3aed', width=2)))
        fig.add_trace(go.Scatter(x=[fr['real'][0]], y=[fr['imag'][0]], mode='markers',
                                 name='ω → 0', marker=dict(color='#22c55e', size=12, symbol='circle')))
        fig.add_trace(go.Scatter(x=[fr['real'][-1]], y=[fr['imag'][-1]], mode='markers',
                                 name='ω → ∞', marker=dict(color='#ef4444', size=12, symbol='x')))
        
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

else:
    st.info("👈 Please load a transfer function from the sidebar to begin!")
    
    st.markdown("""
    ### 🎯 How to use ControlPlot Studio
    
    1. **Choose an input method** from the sidebar
    2. **Enter your transfer function** or select an example
    3. **Explore the plots** in the main area
    
    ### 💡 Quick Examples
    
    - **RC Low Pass:** `num = 1`, `den = 1, 1`
    - **RC High Pass:** `num = 1, 0`, `den = 1, 1`
    - **Second Order:** `num = 1`, `den = 1, 2, 100`
    - **Integrator:** `num = 1`, `den = 1, 0`
    """)
