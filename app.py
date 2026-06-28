import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

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
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton > button {
        background: #667eea;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background: #5a6fd6;
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
            self.gain = num[0] / den[0] if den[0] != 0 else 1
            self.poles = np.roots(den)
            self.zeros = np.roots(num)
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
                    num = np.convolve(num, [1, -z])
                else:
                    num = np.convolve(num, [0, 1])
                    while len(num) > 1 and abs(num[0]) < 1e-10:
                        num = num[1:]
            
            den = [1]
            for p in poles:
                if p != 0:
                    den = np.convolve(den, [1, -p])
                else:
                    den = np.convolve(den, [0, 1])
                    while len(den) > 1 and abs(den[0]) < 1e-10:
                        den = den[1:]
            
            self.num = num.tolist()
            self.den = den.tolist()
            self.poles = poles
            self.zeros = zeros
            self.gain = gain
            return True
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return False
    
    def get_frequency_response(self, omega_min=0.01, omega_max=10000, num_points=1000):
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_points)
        s = 1j * omega
        num_eval = np.polyval(self.num, s)
        den_eval = np.polyval(self.den, s)
        H = num_eval / den_eval
        return {
            'omega': omega,
            'mag_db': 20 * np.log10(np.abs(H) + 1e-10),
            'mag_linear': np.abs(H),
            'phase_deg': np.angle(H, deg=True),
            'real': np.real(H),
            'imag': np.imag(H)
        }
    
    def get_asymptotic_response(self, omega_min=0.01, omega_max=10000, num_points=1000):
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_points)
        mag_asym = np.full(len(omega), 20 * np.log10(abs(self.gain)))
        phase_asym = np.full(len(omega), 0.0 if self.gain > 0 else -180.0)
        
        for pole in self.poles:
            if abs(pole.imag) < 1e-10:
                pole = pole.real
                if abs(pole) > 1e-10:
                    omega_c = abs(pole)
                    mag_asym -= 20 * np.log10(np.sqrt(1 + (omega/omega_c)**2))
                    phase_asym -= np.degrees(np.arctan(omega/omega_c))
                else:
                    mag_asym -= 20 * np.log10(omega)
                    phase_asym -= 90
        
        for zero in self.zeros:
            if abs(zero.imag) < 1e-10:
                zero = zero.real
                if abs(zero) > 1e-10:
                    omega_c = abs(zero)
                    mag_asym += 20 * np.log10(np.sqrt(1 + (omega/omega_c)**2))
                    phase_asym += np.degrees(np.arctan(omega/omega_c))
                else:
                    mag_asym += 20 * np.log10(omega)
                    phase_asym += 90
        
        return {'omega': omega, 'mag_db': mag_asym, 'phase_deg': phase_asym}
    
    def get_pole_zero_data(self):
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
        stable = all(p.real < 0 for p in self.poles)
        freq_response = self.get_frequency_response()
        phase = freq_response['phase_deg']
        mag_db = freq_response['mag_db']
        
        phase_cross_idx = np.where(np.diff(np.sign(phase + 180)))[0]
        gain_margin = None
        phase_margin = None
        
        if len(phase_cross_idx) > 0:
            idx = phase_cross_idx[0]
            if idx < len(mag_db) - 1:
                gain_margin = -mag_db[idx]
        
        gain_cross_idx = np.where(np.diff(np.sign(mag_db)))[0]
        if len(gain_cross_idx) > 0:
            idx = gain_cross_idx[0]
            if idx < len(phase) - 1:
                phase_margin = 180 + phase[idx]
        
        return {'stable': stable, 'gain_margin': gain_margin, 'phase_margin': phase_margin}

def create_bode_plot(data):
    freq_response = data['frequency_response']
    asymptotic = data['asymptotic']
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=('Magnitude Response', 'Phase Response'))
    
    fig.add_trace(go.Scatter(x=freq_response['omega'], y=freq_response['mag_db'],
                             mode='lines', name='Actual Magnitude',
                             line=dict(color='#2563eb', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=asymptotic['omega'], y=asymptotic['mag_db'],
                             mode='lines', name='Asymptotic',
                             line=dict(color='#22c55e', width=1.5, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=freq_response['omega'], y=freq_response['phase_deg'],
                             mode='lines', name='Actual Phase',
                             line=dict(color='#dc2626', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=asymptotic['omega'], y=asymptotic['phase_deg'],
                             mode='lines', name='Asymptotic Phase',
                             line=dict(color='#f59e0b', width=1.5, dash='dash')), row=2, col=1)
    
    fig.update_xaxes(type='log', title_text='Frequency (rad/s)', gridcolor='#f0f0f0', row=2, col=1)
    fig.update_xaxes(type='log', showticklabels=False, gridcolor='#f0f0f0', row=1, col=1)
    fig.update_yaxes(title_text='Magnitude (dB)', gridcolor='#f0f0f0', row=1, col=1)
    fig.update_yaxes(title_text='Phase (degrees)', gridcolor='#f0f0f0', row=2, col=1)
    fig.update_layout(height=600, hovermode='closest', showlegend=True,
                      legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'))
    return fig

def create_nyquist_plot(freq_response):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_response['real'], y=freq_response['imag'],
                             mode='lines', name='Nyquist Plot',
                             line=dict(color='#7c3aed', width=2),
                             hovertemplate='Real: %{x:.3f}<br>Imag: %{y:.3f}<br>ω: %{text:.2f} rad/s<extra></extra>',
                             text=freq_response['omega']))
    fig.add_trace(go.Scatter(x=[freq_response['real'][0]], y=[freq_response['imag'][0]],
                             mode='markers', name='ω → 0',
                             marker=dict(color='#22c55e', size=12, symbol='circle')))
    fig.add_trace(go.Scatter(x=[freq_response['real'][-1]], y=[freq_response['imag'][-1]],
                             mode='markers', name='ω → ∞',
                             marker=dict(color='#ef4444', size=12, symbol='x')))
    
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                             name='Unit Circle', line=dict(color='#ccc', width=1, dash='dot'),
                             showlegend=False))
    
    fig.update_layout(title='Nyquist Plot', height=500, hovermode='closest',
                      xaxis=dict(title='Real', gridcolor='#f0f0f0', zeroline=True,
                                 scaleanchor='y', scaleratio=1),
                      yaxis=dict(title='Imaginary', gridcolor='#f0f0f0', zeroline=True),
                      legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'))
    return fig

def create_pole_zero_plot(pole_zero_data):
    fig = go.Figure()
    poles = pole_zero_data['poles']
    zeros = pole_zero_data['zeros']
    
    if poles:
        fig.add_trace(go.Scatter(x=[p['real'] for p in poles], y=[p['imag'] for p in poles],
                                 mode='markers', name='Poles',
                                 marker=dict(color='#ef4444', size=14, symbol='x')))
    if zeros:
        fig.add_trace(go.Scatter(x=[z['real'] for z in zeros], y=[z['imag'] for z in zeros],
                                 mode='markers', name='Zeros',
                                 marker=dict(color='#2563eb', size=14, symbol='circle-open')))
    
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                             name='Unit Circle', line=dict(color='#ccc', width=1, dash='dot'),
                             showlegend=False))
    
    fig.update_layout(title='Pole-Zero Map', height=500, hovermode='closest',
                      xaxis=dict(title='Real', gridcolor='#f0f0f0', scaleanchor='y', scaleratio=1),
                      yaxis=dict(title='Imaginary', gridcolor='#f0f0f0'),
                      legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'))
    return fig

def main():
    st.markdown("""
        <div class="main-header">
            <h1>🎛️ ControlPlot Studio</h1>
            <p>Interactive Control System Visualization Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    if 'engine' not in st.session_state:
        st.session_state.engine = ControlEngine()
        st.session_state.tf_loaded = False
    
    with st.sidebar:
        st.header("📥 Input Methods")
        input_method = st.radio("Choose Input Method", ["Direct Entry", "Builder Interface", "Example Library"])
        st.divider()
        
        if input_method == "Direct Entry":
            st.subheader("Transfer Function")
            num_str = st.text_input("Numerator (e.g., 10, 20):", value="10, 20")
            den_str = st.text_input("Denominator (e.g., 1, 3, 2):", value="1, 3, 2")
            if st.button("🚀 Generate Plots", use_container_width=True):
                if st.session_state.engine.parse_transfer_function(num_str, den_str):
                    st.session_state.tf_loaded = True
                    st.success("✅ Loaded!")
        
        elif input_method == "Builder Interface":
            st.subheader("Build Your System")
            gain = st.number_input("Gain:", value=10.0, step=0.1)
            zeros_input = []
            num_zeros = st.number_input("Number of zeros:", min_value=0, max_value=10, value=1, step=1)
            for i in range(num_zeros):
                zeros_input.append(st.text_input(f"Zero {i+1}:", value="2" if i == 0 else ""))
            
            poles_input = []
            num_poles = st.number_input("Number of poles:", min_value=0, max_value=10, value=2, step=1)
            for i in range(num_poles):
                poles_input.append(st.text_input(f"Pole {i+1}:", value="1" if i == 0 else "5" if i == 1 else ""))
            
            if st.button("🏗️ Build & Generate", use_container_width=True):
                if st.session_state.engine.parse_from_builder(gain, zeros_input, poles_input):
                    st.session_state.tf_loaded = True
                    st.success("✅ Built!")
        
        elif input_method == "Example Library":
            st.subheader("Pre-built Examples")
            examples = {
                "RC Low Pass": {"num": "1", "den": "1, 1"},
                "RC High Pass": {"num": "1, 0", "den": "1, 1"},
                "Second Order": {"num": "1", "den": "1, 2, 100"},
                "Integrator": {"num": "1", "den": "1, 0"},
            }
            for name, tf in examples.items():
                if st.button(f"📂 {name}", use_container_width=True):
                    if st.session_state.engine.parse_transfer_function(tf["num"], tf["den"]):
                        st.session_state.tf_loaded = True
                        st.success(f"✅ {name} loaded!")
        
        if st.session_state.tf_loaded:
            st.divider()
            st.subheader("📊 System")
            num_str = " + ".join([f"{c}s^{i}" for i, c in enumerate(reversed(st.session_state.engine.num)) if abs(c) > 1e-10])
            den_str = " + ".join([f"{c}s^{i}" for i, c in enumerate(reversed(st.session_state.engine.den)) if abs(c) > 1e-10])
            st.latex(f"H(s) = \\frac{{{num_str}}}{{{den_str}}}")
    
    if st.session_state.tf_loaded:
        engine = st.session_state.engine
        freq_response = engine.get_frequency_response()
        asymptotic = engine.get_asymptotic_response()
        pole_zero = engine.get_pole_zero_data()
        stability = engine.get_stability_info()
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Bode Plot", "🌀 Nyquist Plot", "📍 Pole-Zero", "ℹ️ Info"])
        
        with tab1:
            st.plotly_chart(create_bode_plot({'frequency_response': freq_response, 'asymptotic': asymptotic}), use_container_width=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Gain Margin", f"{stability['gain_margin']:.2f} dB" if stability['gain_margin'] else "∞")
            col2.metric("Phase Margin", f"{stability['phase_margin']:.2f}°" if stability['phase_margin'] else "∞")
            col3.metric("Status", "✅ Stable" if stability['stable'] else "❌ Unstable")
        
        with tab2:
            st.plotly_chart(create_nyquist_plot(freq_response), use_container_width=True)
            freq_idx = st.slider("Frequency", 0, len(freq_response['omega'])-1, len(freq_response['omega'])//2)
            omega = freq_response['omega'][freq_idx]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ω", f"{omega:.2f} rad/s")
            col2.metric("Magnitude", f"{freq_response['mag_db'][freq_idx]:.2f} dB")
            col3.metric("Phase", f"{freq_response['phase_deg'][freq_idx]:.1f}°")
            col4.metric("H(jω)", f"{freq_response['real'][freq_idx]:.3f} + {freq_response['imag'][freq_idx]:.3f}i")
        
        with tab3:
            st.plotly_chart(create_pole_zero_plot(pole_zero), use_container_width=True)
            st.success("✅ Stable" if stability['stable'] else "⚠️ Unstable")
        
        with tab4:
            st.subheader("System Analysis")
            st.markdown(f"**Stability:** {'✅ Stable' if stability['stable'] else '❌ Unstable'}")
            st.markdown(f"**Gain Margin:** {stability['gain_margin']:.2f} dB" if stability['gain_margin'] else "∞")
            st.markdown(f"**Phase Margin:** {stability['phase_margin']:.2f}°" if stability['phase_margin'] else "∞")
    else:
        st.info("👈 Load a transfer function from the sidebar!")

if __name__ == "__main__":
    main()
