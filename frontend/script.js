const API_URL = 'http://localhost:8000';

let currentData = null;
let bodeMagnitudePlot = null;
let bodePhasePlot = null;
let nyquistPlot = null;
let poleZeroPlot = null;

// Example configurations
const examples = {
    'rc_lowpass': {
        num: '1',
        den: '1, 1'
    },
    'rc_highpass': {
        num: '1, 0',
        den: '1, 1'
    },
    'second_order': {
        num: '1',
        den: '1, 2, 100'
    },
    'integrator': {
        num: '1',
        den: '1, 0'
    }
};

// Load transfer function
async function loadTransferFunction() {
    const num = document.getElementById('numerator-input').value;
    const den = document.getElementById('denominator-input').value;
    
    try {
        const response = await fetch(`${API_URL}/parse`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ numerator: num, denominator: den })
        });
        
        if (!response.ok) throw new Error('Invalid transfer function');
        
        const data = await response.json();
        await loadBodeData();
        await loadPoleZeroData();
        await loadStabilityInfo();
        await loadNyquistData();
        
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

// Load from builder
async function loadFromBuilder() {
    const gain = parseFloat(document.getElementById('gain-input').value);
    const zeros = Array.from(document.querySelectorAll('.zero-input')).map(inp => inp.value);
    const poles = Array.from(document.querySelectorAll('.pole-input')).map(inp => inp.value);
    
    try {
        const response = await fetch(`${API_URL}/builder`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gain, zeros, poles })
        });
        
        if (!response.ok) throw new Error('Invalid builder input');
        
        const data = await response.json();
        await loadBodeData();
        await loadPoleZeroData();
        await loadStabilityInfo();
        await loadNyquistData();
        
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

// Load Bode plot data
async function loadBodeData() {
    try {
        const response = await fetch(`${API_URL}/bode`, {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error('Failed to load Bode data');
        
        const data = await response.json();
        currentData = data;
        
        createBodePlots(data);
        
    } catch (error) {
        console.error('Error loading Bode data:', error);
    }
}

// Load Nyquist data
async function loadNyquistData() {
    try {
        const response = await fetch(`${API_URL}/frequency_response`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ omega_min: 0.01, omega_max: 10000, num_points: 500 })
        });
        
        if (!response.ok) throw new Error('Failed to load Nyquist data');
        
        const data = await response.json();
        createNyquistPlot(data);
        
    } catch (error) {
        console.error('Error loading Nyquist data:', error);
    }
}

// Load pole-zero data
async function loadPoleZeroData() {
    try {
        const response = await fetch(`${API_URL}/pole_zero`, {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error('Failed to load pole-zero data');
        
        const data = await response.json();
        createPoleZeroPlot(data);
        
    } catch (error) {
        console.error('Error loading pole-zero data:', error);
    }
}

// Load stability info
async function loadStabilityInfo() {
    try {
        const response = await fetch(`${API_URL}/stability`, {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error('Failed to load stability info');
        
        const data = await response.json();
        displayStabilityInfo(data);
        
    } catch (error) {
        console.error('Error loading stability info:', error);
    }
}

// Create Bode plots
function createBodePlots(data) {
    const { frequency_response, asymptotic } = data;
    const omega = frequency_response.omega;
    
    // Magnitude plot
    const magTrace = {
        x: omega,
        y: frequency_response.mag_db,
        mode: 'lines',
        name: 'Magnitude (actual)',
        line: { color: '#2563eb', width: 2 }
    };
    
    const asymMagTrace = {
        x: asymptotic.omega,
        y: asymptotic.mag_db,
        mode: 'lines',
        name: 'Asymptotic approximation',
        line: { color: '#22c55e', width: 1.5, dash: 'dash' }
    };
    
    const magLayout = {
        title: 'Bode Magnitude Plot',
        xaxis: { 
            title: 'Frequency (rad/s)', 
            type: 'log',
            gridcolor: '#f0f0f0'
        },
        yaxis: { 
            title: 'Magnitude (dB)',
            gridcolor: '#f0f0f0'
        },
        showlegend: true,
        legend: { x: 1, y: 1 },
        hovermode: 'closest'
    };
    
    // Phase plot
    const phaseTrace = {
        x: omega,
        y: frequency_response.phase_deg,
        mode: 'lines',
        name: 'Phase (actual)',
        line: { color: '#dc2626', width: 2 }
    };
    
    const asymPhaseTrace = {
        x: asymptotic.omega,
        y: asymptotic.phase_deg,
        mode: 'lines',
        name: 'Asymptotic approximation',
        line: { color: '#22c55e', width: 1.5, dash: 'dash' }
    };
    
    const phaseLayout = {
        title: 'Bode Phase Plot',
        xaxis: { 
            title: 'Frequency (rad/s)', 
            type: 'log',
            gridcolor: '#f0f0f0'
        },
        yaxis: { 
            title: 'Phase (degrees)',
            gridcolor: '#f0f0f0'
        },
        showlegend: true,
        legend: { x: 1, y: 1 },
        hovermode: 'closest'
    };
    
    bodeMagnitudePlot = Plotly.newPlot('bode-magnitude', [magTrace, asymMagTrace], magLayout);
    bodePhasePlot = Plotly.newPlot('bode-phase', [phaseTrace, asymPhaseTrace], phaseLayout);
}

// Create Nyquist plot
function createNyquistPlot(data) {
    const trace = {
        x: data.real,
        y: data.imag,
        mode: 'lines',
        name: 'Nyquist Plot',
        line: { color: '#7c3aed', width: 2 },
        hovertemplate: 'Real: %{x:.3f}<br>Imag: %{y:.3f}<br>ω: %{text:.2f} rad/s<extra></extra>',
        text: data.omega
    };
    
    // Add start and end markers
    const startTrace = {
        x: [data.real[0]],
        y: [data.imag[0]],
        mode: 'markers',
        name: 'ω → 0',
        marker: { color: '#22c55e', size: 10, symbol: 'circle' }
    };
    
    const endTrace = {
        x: [data.real[data.real.length - 1]],
        y: [data.imag[data.imag.length - 1]],
        mode: 'markers',
        name: 'ω → ∞',
        marker: { color: '#ef4444', size: 10, symbol: 'x' }
    };
    
    const layout = {
        title: 'Nyquist Plot',
        xaxis: { 
            title: 'Real',
            gridcolor: '#f0f0f0',
            zeroline: true,
            zerolinecolor: '#ccc'
        },
        yaxis: { 
            title: 'Imaginary',
            gridcolor: '#f0f0f0',
            zeroline: true,
            zerolinecolor: '#ccc',
            scaleanchor: 'x',
            scaleratio: 1
        },
        showlegend: true,
        legend: { x: 1, y: 1 },
        hovermode: 'closest',
        width: undefined,
        height: undefined
    };
    
    nyquistPlot = Plotly.newPlot('nyquist-plot', [trace, startTrace, endTrace], layout);
}

// Create Pole-Zero plot
function createPoleZeroPlot(data) {
    const poles = data.poles || [];
    const zeros = data.zeros || [];
    
    const poleTrace = {
        x: poles.map(p => p.real),
        y: poles.map(p => p.imag),
        mode: 'markers',
        name: 'Poles',
        marker: { 
            color: '#ef4444', 
            size: 12, 
            symbol: 'x',
            line: { width: 2 }
        },
        text: poles.map(p => `Pole: ${p.real.toFixed(2)} + ${p.imag.toFixed(2)}i`),
        hovertemplate: '%{text}<extra></extra>'
    };
    
    const zeroTrace = {
        x: zeros.map(z => z.real),
        y: zeros.map(z => z.imag),
        mode: 'markers',
        name: 'Zeros',
        marker: { 
            color: '#2563eb', 
            size: 12, 
            symbol: 'circle-open',
            line: { width: 2 }
        },
        text: zeros.map(z => `Zero: ${z.real.toFixed(2)} + ${z.imag.toFixed(2)}i`),
        hovertemplate: '%{text}<extra></extra>'
    };
    
    const layout = {
        title: 'Pole-Zero Map',
        xaxis: { 
            title: 'Real',
            gridcolor: '#f0f0f0',
            zeroline: true,
            zerolinecolor: '#ccc'
        },
        yaxis: { 
            title: 'Imaginary',
            gridcolor: '#f0f0f0',
            zeroline: true,
            zerolinecolor: '#ccc',
            scaleanchor: 'x',
            scaleratio: 1
        },
        showlegend: true,
        legend: { x: 1, y: 1 },
        hovermode: 'closest'
    };
    
    // Add unit circle for reference (if needed)
    if (poles.length > 0 || zeros.length > 0) {
        // Optional: add unit circle reference
        const theta = Array.from({ length: 100 }, (_, i) => (i / 99) * 2 * Math.PI);
        const circleTrace = {
            x: theta.map(t => Math.cos(t)),
            y: theta.map(t => Math.sin(t)),
            mode: 'lines',
            name: 'Unit Circle',
            line: { color: '#ccc', width: 1, dash: 'dot' },
            showlegend: false
        };
        poleZeroPlot = Plotly.newPlot('polezero-plot', [circleTrace, poleTrace, zeroTrace], layout);
    } else {
        poleZeroPlot = Plotly.newPlot('polezero-plot', [poleTrace, zeroTrace], layout);
    }
}

// Display stability information
function displayStabilityInfo(data) {
    const container = document.getElementById('system-info');
    const stabilityText = data.stable ? '✅ Stable' : '❌ Unstable';
    const stabilityClass = data.stable ? 'stable' : 'unstable';
    
    let html = `
        <div class="info-item">
            <span class="info-label">System Stability:</span>
            <span class="info-value ${stabilityClass}">${stabilityText}</span>
        </div>
    `;
    
    if (data.gain_margin !== null && data.gain_margin !== undefined) {
        html += `
            <div class="info-item">
                <span class="info-label">Gain Margin:</span>
                <span class="info-value">${data.gain_margin.toFixed(2)} dB</span>
            </div>
        `;
    } else {
        html += `
            <div class="info-item">
                <span class="info-label">Gain Margin:</span>
                <span class="info-value">∞ (no phase crossover)</span>
            </div>
        `;
    }
    
    if (data.phase_margin !== null && data.phase_margin !== undefined) {
        html += `
            <div class="info-item">
                <span class="info-label">Phase Margin:</span>
                <span class="info-value">${data.phase_margin.toFixed(2)}°</span>
            </div>
        `;
    } else {
        html += `
            <div class="info-item">
                <span class="info-label">Phase Margin:</span>
                <span class="info-value">∞ (no gain crossover)</span>
            </div>
        `;
    }
    
    // Add additional info
    if (currentData) {
        const freq = currentData.frequency_response;
        const maxMag = Math.max(...freq.mag_db);
        const minMag = Math.min(...freq.mag_db);
        
        html
