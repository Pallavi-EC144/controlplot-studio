import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sympy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import math

# Set page config
st.set_page_config(
    page_title="ControlPlot Studio",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: #5a6fd6;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .info-box {
        background: #f8f9fe;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .stability-stable {
        color: #22c55e;
        font-weight: bold;
    }
    .stability-unstable {
        color: #ef4444;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# Control Engine Class
# ============================================

class ControlEngine:
    def __init__(self):
        self.num = [1]
        self.den = [1]
        self.poles = []
        self.zeros = []
        self.gain = 1
        
    def parse_transfer_function(self, num_str, den_str):
        """Parse transfer function from string input"""
        try:
            if num_str.strip():
                num = [float(x.strip()) for x in num_str.split(',') if x.strip()]
            else:
                num = [1]
            
            if den_str.strip():
                den = [float(x.strip()) for x in den_str.split(',') if x.strip()]
            else:
                den = [1]
            
            # Remove trailing zeros
            while len(num) > 1 and abs(num[-1]) < 1e-10:
                num.pop()
            while len(den) > 1 and abs(den[-1]) < 1e-10:
                den.pop()
            
            self.num = num
            self.den = den
            self.gain = num[0] / den[0] if den[0] != 0 else 1
            
            # Find poles and zeros
            self.poles = np.roots(den)
            self.zeros = np.roots(num)
            
            return True
        except Exception as e:
            st.error(f"Error parsing transfer function: {str(e)}")
            return False
    
    def parse_from_builder(self, gain, zeros, poles):
        """Parse from builder interface"""
        try:
            gain = float(gain)
            zeros = [float(z) for z in zeros if z and z.strip()]
            poles = [float(p) for p in poles if p and p.strip()]
            
            # Build polynomial from zeros
            num = [gain]
            for z in zeros:
                if z != 0:
                    num = np.convolve(num, [1, -z])
                else:
                    # Zero at origin
                    num = np.convolve(num, [0, 1])
                    while len(num) > 1 and abs(num[0]) < 1e-10:
                        num = num[1:]
            
            # Build polynomial from poles
            den = [1]
            for p in poles:
                if p != 0:
                    den = np.convolve(den, [1, -p])
                else:
                    # Pole at origin
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
            st.error(f"Error building transfer function: {str(e)}")
            return False
    
    def get_frequency_response(self, omega_min=0.01, omega_max=10000, num_points=1000):
        """Calculate frequency response"""
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), num_points)
        
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
            'omega': omega,
            'mag_db': mag_db,
            'mag_linear': mag_linear,
            'phase_deg': phase_deg,
            'real': real,
            'imag': imag
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
                if abs(pole) > 1e-10:
                    # Pole at -a
                    omega_c = abs(pole)
                    mag_asym -= 20 * np.log10(np.sqrt(1 + (omega/omega_c)**2))
                    phase_asym -= np.degrees(np.arctan(omega/omega_c))
                else:  # Pole at origin
                    mag_asym -= 20 * np.log10(omega)
                    phase_asym -= 90
        
        # Add contributions from zeros
        for zero in self.zeros:
            if abs(zero.imag) < 1e-10:  # Real zero
                zero = zero.real
                if abs(zero) > 1e-10:
                    omega_c = abs(zero)
                    mag_asym += 20 * np.log10(np.sqrt(1 + (omega/omega_c)**2))
                    phase_asym += np.degrees(np.arctan(omega/omega_c))
                else:  # Zero at origin
                    mag_asym += 20 * np.log10(omega)
                    phase_asym += 90
        
        return {
            'omega': omega,
            'mag_db': mag_asym,
            'phase_deg': phase_asym
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
        freq_response = self.get_frequency_response()
        phase = freq_response['phase_deg']
        mag_db = freq_response['mag_db']
        omega = freq_response['omega']
        
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

# ============================================
# Plotting Functions
# ============================================

def create_bode_plot(data):
    """Create Bode plot using Plotly"""
    freq_response = data['frequency_response']
    asymptotic = data['asymptotic']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Magnitude Response', 'Phase Response')
    )
    
    # Magnitude plot - actual
    fig.add_trace(
        go.Scatter(
            x=freq_response['omega'],
            y=freq_response['mag_db'],
            mode='lines',
            name='Actual Magnitude',
            line=dict(color='#2563eb', width=2),
            hovertemplate='ω: %{x:.2f} rad/s<br>Mag: %{y:.2f} dB<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Magnitude plot - asymptotic
    fig.add_trace(
        go.Scatter(
            x=asymptotic['omega'],
            y=asymptotic['mag_db'],
            mode='lines',
            name='Asymptotic',
            line=dict(color='#22c55e', width=1.5, dash='dash'),
            hovertemplate='ω: %{x:.2f} rad/s<br>Mag: %{y:.2f} dB<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Phase plot - actual
    fig.add_trace(
        go.Scatter(
            x=freq_response['omega'],
            y=freq_response['phase_deg'],
            mode='lines',
            name='Actual Phase',
            line=dict(color='#dc2626', width=2),
            hovertemplate='ω: %{x:.2f} rad/s<br>Phase: %{y:.1f}°<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Phase plot - asymptotic
    fig.add_trace(
        go.Scatter(
            x=asymptotic['omega'],
            y=asymptotic['phase_deg'],
            mode='lines',
            name='Asymptotic Phase',
            line=dict(color='#f59e0b', width=1.5, dash='dash'),
            hovertemplate='ω: %{x:.2f} rad/s<br>Phase: %{y:.1f}°<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(
        type='log',
        title_text='Frequency (rad/s)',
        gridcolor='#f0f0f0',
        row=2, col=1
    )
    
    fig.update_xaxes(
        type='log',
        showticklabels=False,
        gridcolor='#f0f0f0',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text='Magnitude (dB)',
        gridcolor='#f0f0f0',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text='Phase (degrees)',
        gridcolor='#f0f0f0',
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_nyquist_plot(freq_response):
    """Create Nyquist plot"""
    fig = go.Figure()
    
    # Main Nyquist curve
    fig.add_trace(
        go.Scatter(
            x=freq_response['real'],
            y=freq_response['imag'],
            mode='lines',
            name='Nyquist Plot',
            line=dict(color='#7c3aed', width=2),
            hovertemplate='Real: %{x:.3f}<br>Imag: %{y:.3f}<br>ω: %{text:.2f} rad/s<extra></extra>',
            text=freq_response['omega']
        )
    )
    
    # Start point (ω → 0)
    fig.add_trace(
        go.Scatter(
            x=[freq_response['real'][0]],
            y=[freq_response['imag'][0]],
            mode='markers',
            name='ω → 0',
            marker=dict(color='#22c55e', size=12, symbol='circle')
        )
    )
    
    # End point (ω → ∞)
    fig.add_trace(
        go.Scatter(
            x=[freq_response['real'][-1]],
            y=[freq_response['imag'][-1]],
            mode='markers',
            name='ω → ∞',
            marker=dict(color='#ef4444', size=12, symbol='x')
        )
    )
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            name='Unit Circle',
            line=dict(color='#ccc', width=1, dash='dot'),
            showlegend=False
        )
    )
    
    fig.update_layout(
        title='Nyquist Plot',
        height=500,
        hovermode='closest',
        showlegend=True,
        xaxis=dict(
            title='Real',
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinecolor='#ccc',
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            title='Imaginary',
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinecolor='#ccc'
        ),
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_pole_zero_plot(pole_zero_data):
    """Create Pole-Zero map"""
    fig = go.Figure()
    
    poles = pole_zero_data['poles']
    zeros = pole_zero_data['zeros']
    
    # Plot poles
    if poles:
        fig.add_trace(
            go.Scatter(
                x=[p['real'] for p in poles],
                y=[p['imag'] for p in poles],
                mode='markers',
                name='Poles',
                marker=dict(color='#ef4444', size=14, symbol='x', line=dict(width=2)),
                hovertemplate='Pole: %{x:.2f} + %{y:.2f}i<extra></extra>'
            )
        )
    
    # Plot zeros
    if zeros:
        fig.add_trace(
            go.Scatter(
                x=[z['real'] for z in zeros],
                y=[z['imag'] for z in zeros],
                mode='markers',
                name='Zeros',
                marker=dict(color='#2563eb', size=14, symbol='circle-open', line=dict(width=2)),
                hovertemplate='Zero: %{x:.2f} + %{y:.2f}i<extra></extra>'
            )
        )
    
    # Add unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            name='Unit Circle',
            line=dict(color='#ccc', width=1, dash='dot'),
            showlegend=False
        )
    )
    
    # Add axes
    fig.add_shape(type='line', x0=-5, y0=0, x1=5, y1=0, 
                  line=dict(color='#ccc', width=1))
    fig.add_shape(type='line', x0=0, y0=-5, x1=0, y1=5,
                  line=dict(color='#ccc', width=1))
    
    fig.update_layout(
        title='Pole-Zero Map',
        height=500,
        hovermode='closest',
        showlegend=True,
        xaxis=dict(
            title='Real',
            gridcolor='#f0f0f0',
            zeroline=False,
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            title='Imaginary',
            gridcolor='#f0f0f0',
            zeroline=False
        ),
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# ============================================
# Main App
# ============================================

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎛️ ControlPlot Studio</h1>
            <p>Interactive Control System Visualization Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'engine' not in st.session_state:
        st.session_state.engine = ControlEngine()
        st.session_state.tf_loaded = False
    
    # Sidebar - Input Methods
    with st.sidebar:
        st.header("📥 Input Methods")
        
        input_method = st.radio(
            "Choose Input Method",
            ["Direct Entry", "Builder Interface", "Example Library"],
            index=0
        )
        
        st.divider()
        
        if input_method == "Direct Entry":
            st.subheader("Transfer Function")
            
            num_str = st.text_input(
                "Numerator coefficients (e.g., 10, 20):",
                value="10, 20",
                help="Enter coefficients separated by commas"
            )
            
            den_str = st.text_input(
                "Denominator coefficients (e.g., 1, 3, 2):",
                value="1, 3, 2",
                help="Enter coefficients separated by commas"
            )
            
            if st.button("🚀 Generate Plots", use_container_width=True):
                with st.spinner("Calculating..."):
                    success = st.session_state.engine.parse_transfer_function(num_str, den_str)
                    if success:
                        st.session_state.tf_loaded = True
                        st.success("✅ Transfer function loaded!")
        
        elif input_method == "Builder Interface":
            st.subheader("Build Your System")
            
            gain = st.number_input("Gain:", value=10.0, step=0.1)
            
            st.write("**Zeros:**")
            zeros_input = []
            num_zeros = st.number_input("Number of zeros:", min_value=0, max_value=10, value=1, step=1)
            
            for i in range(num_zeros):
                z = st.text_input(f"Zero {i+1}:", value="2" if i == 0 else "")
                zeros_input.append(z)
            
            st.write("**Poles:**")
            poles_input = []
            num_poles = st.number_input("Number of poles:", min_value=0, max_value=10, value=2, step=1)
            
            for i in range(num_poles):
                p = st.text_input(f"Pole {i+1}:", value="1" if i == 0 else "5" if i == 1 else "")
                poles_input.append(p)
            
            if st.button("🏗️ Build & Generate", use_container_width=True):
                with st.spinner("Building system..."):
                    success = st.session_state.engine.parse_from_builder(gain, zeros_input, poles_input)
                    if success:
                        st.session_state.tf_loaded = True
                        st.success("✅ System built successfully!")
        
        elif input_method == "Example Library":
            st.subheader("Pre-built Examples")
            
            examples = {
                "RC Low Pass": {"num": "1", "den": "1, 1"},
                "RC High Pass": {"num": "1, 0", "den": "1, 1"},
                "Second Order": {"num": "1", "den": "1, 2, 100"},
                "Integrator": {"num": "1", "den": "1, 0"},
                "Differentiator": {"num": "1, 0", "den": "1"},
                "Band Pass": {"num": "10, 0", "den": "1, 2, 100"},
                "PID Controller": {"num": "100, 10, 1", "den": "1, 0"}
            }
            
            for name, tf in examples.items():
                if st.button(f"📂 {name}", use_container_width=True):
                    with st.spinner(f"Loading {name}..."):
                        success = st.session_state.engine.parse_transfer_function(tf["num"], tf["den"])
                        if success:
                            st.session_state.tf_loaded = True
                            st.success(f"✅ {name} loaded!")
        
        st.divider()
        
        # Display current transfer function
        if st.session_state.tf_loaded:
            st.subheader("📊 Current System")
            
            # Show transfer function as a nice string
            num_str = " + ".join([f"{c}s^{i}" for i, c in enumerate(reversed(st.session_state.engine.num)) if abs(c) > 1e-10])
            den_str = " + ".join([f"{c}s^{i}" for i, c in enumerate(reversed(st.session_state.engine.den)) if abs(c) > 1e-10])
            
            if not num_str:
                num_str = "0"
            if not den_str:
                den_str = "0"
            
            st.latex(f"H(s) = \\frac{{{num_str}}}{{{den_str}}}")
            
            # Show poles and zeros
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Poles:**")
                for p in st.session_state.engine.poles:
                    if abs(p.imag) < 1e-10:
                        st.write(f"• {p.real:.3f}")
                    else:
                        st.write(f"• {p.real:.3f} ± {abs(p.imag):.3f}i")
            
            with col2:
                st.write("**Zeros:**")
                for z in st.session_state.engine.zeros:
                    if abs(z.imag) < 1e-10:
                        st.write(f"• {z.real:.3f}")
                    else:
                        st.write(f"• {z.real:.3f} ± {abs(z.imag):.3f}i")
    
    # Main area - Plots
    if st.session_state.tf_loaded:
        # Get data
        engine = st.session_state.engine
        
        with st.spinner("Generating plots..."):
            freq_response = engine.get_frequency_response()
            asymptotic = engine.get_asymptotic_response()
            pole_zero = engine.get_pole_zero_data()
            stability = engine.get_stability_info()
            
            bode_data = {
                'frequency_response': freq_response,
                'asymptotic': asymptotic
            }
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Bode Plot", "🌀 Nyquist Plot", "📍 Pole-Zero Map", "ℹ️ System Info"])
        
        with tab1:
            st.plotly_chart(create_bode_plot(bode_data), use_container_width=True)
            
            # Additional info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Gain Margin",
                    f"{stability['gain_margin']:.2f} dB" if stability['gain_margin'] else "∞"
                )
            with col2:
                st.metric(
                    "Phase Margin",
                    f"{stability['phase_margin']:.2f}°" if stability['phase_margin'] else "∞"
                )
            with col3:
                status = "✅ Stable" if stability['stable'] else "❌ Unstable"
                st.metric("System Status", status)
        
        with tab2:
            st.plotly_chart(create_nyquist_plot(freq_response), use_container_width=True)
            
            # Frequency slider for interactive exploration
            st.subheader("🎚️ Explore Frequency Response")
            
            omega_min = float(np.min(freq_response['omega']))
            omega_max = float(np.max(freq_response['omega']))
            
            freq_idx = st.slider(
                "Frequency (rad/s)",
                min_value=0,
                max_value=len(freq_response['omega']) - 1,
                value=len(freq_response['omega']) // 2,
                step=1,
                format="%d"
            )
            
            omega = freq_response['omega'][freq_idx]
            real = freq_response['real'][freq_idx]
            imag = freq_response['imag'][freq_idx]
            mag = freq_response['mag_db'][freq_idx]
            phase = freq_response['phase_deg'][freq_idx]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Frequency", f"{omega:.2f} rad/s")
            with col2:
                st.metric("Magnitude", f"{mag:.2f} dB")
            with col3:
                st.metric("Phase", f"{phase:.1f}°")
            with col4:
                st.metric("H(jω)", f"{real:.3f} + {imag:.3f}i")
        
        with tab3:
            st.plotly_chart(create_pole_zero_plot(pole_zero), use_container_width=True)
            
            # Stability explanation
            if stability['stable']:
                st.success("✅ System is **stable** - all poles are in the left half-plane")
            else:
                st.error("⚠️ System is **unstable** - one or more poles are in the right half-plane")
        
        with tab4:
            st.subheader("📊 System Analysis")
            
            # Stability details
            st.markdown(f"""
                <div class="info-box">
                    <h4>🔒 Stability Analysis</h4>
                    <p>Status: <span class="{'stability-stable' if stability['stable'] else 'stability-unstable'}">
                        {'✅ Stable' if stability['stable'] else '❌ Unstable'}
                    </span></p>
                    <p>Gain Margin: <b>{f'{stability["gain_margin"]:.2f} dB' if stability["gain_margin"] else "∞"}</b></p>
                    <p>Phase Margin: <b>{f'{stability["phase_margin"]:.2f}°' if stability["phase_margin"] else "∞"}</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Transfer function in different forms
            st.subheader("📐 Transfer Function Representations")
            
            num_str = " + ".join([f"{c:.3f}s^{i}" for i, c in enumerate(reversed(engine.num)) if abs(c) > 1e-10])
            den_str = " + ".join([f"{c:.3f}s^{i}" for i, c in enumerate(reversed(engine.den)) if abs(c) > 1e-10])
            
            st.write("**Standard Form:**")
            st.latex(f"H(s) = \\frac{{{num_str}}}{{{den_str}}}")
            
            # Factorized form
            if len(engine.zeros) > 0 or len(engine.poles) > 0:
                zero_str = " \\cdot ".join([f"(s - {z.real:.2f})" if abs(z.imag) < 1e-10 
                                           else f"(s - {z.real:.2f} - {abs(z.imag):.2f}i)" 
                                           for z in engine.zeros])
                pole_str = " \\cdot ".join([f"(s - {p.real:.2f})" if abs(p.imag) < 1e-10 
                                           else f"(s - {p.real:.2f} - {abs(p.imag):.2f}i)" 
                                           for p in engine.poles])
                
                if not zero_str:
                    zero_str = "1"
                if not pole_str:
                    pole_str = "1"
                
                st.write("**Factorized Form:**")
                st.latex(f"H(s) = {engine.gain:.2f} \\frac{{{zero_str}}}{{{pole_str}}}")
            
            # Frequency response stats
            st.subheader("📊 Frequency Response Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                max_mag = np.max(freq_response['mag_db'])
                min_mag = np.min(freq_response['mag_db'])
                max_phase = np.max(freq_response['phase_deg'])
                min_phase = np.min(freq_response['phase_deg'])
                
                st.write(f"**Magnitude Range:** {min_mag:.2f} dB → {max_mag:.2f} dB")
                st.write(f"**Phase Range:** {min_phase:.1f}° → {max_phase:.1f}°")
            
            with col2:
                # Find peak resonance if any
                if len(freq_response['mag_linear']) > 0:
                    peak_idx = np.argmax(freq_response['mag_linear'])
                    peak_freq = freq_response['omega'][peak_idx]
                    peak_mag = freq_response['mag_db'][peak_idx]
                    
                    st.write(f"**Resonance Peak:** {peak_mag:.2f} dB at {peak_freq:.2f} rad/s")
                
                # Bandwidth (where magnitude crosses -3 dB)
                mag_linear = freq_response['mag_linear']
                mag_normalized = mag_linear / np.max(mag_linear)
                bw_idx = np.where(mag_normalized <= 0.707)[0]
                if len(bw_idx) > 0:
                    bandwidth = freq_response['omega'][bw_idx[0]]
                    st.write(f"**Bandwidth (-3 dB):** {bandwidth:.2f} rad/s")
                else:
                    st.write("**Bandwidth (-3 dB):** > max frequency")
    
    else:
        # Show welcome message when no system is loaded
        st.info("👈 Please load a transfer function from the sidebar to begin!")
        
        st.markdown("""
        ### 🎯 How to use ControlPlot Studio
        
        1. **Choose an input method** from the sidebar
        2. **Enter your transfer function** or select an example
        3. **Explore the plots** in the main area
        4. **Analyze system properties** like stability and margins
        
        ### 💡 Quick Examples
        
        - **RC Low Pass:** `num = 1`, `den = 1, 1`
        - **RC High Pass:** `num = 1, 0`, `den = 1, 1`
        - **Second Order System:** `num = 1`, `den = 1, 2, 100`
        - **Integrator:** `num = 1`, `den = 1, 0`
        
        ### 📚 Features
        
        - ✅ Bode plots with asymptotic approximation
        - ✅ Nyquist plots with interactive exploration
        - ✅ Pole-zero maps
        - ✅ Stability analysis
        - ✅ Gain and phase margins
        - ✅ Frequency response explorer
        """)

if __name__ == "__main__":
    main()
