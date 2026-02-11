import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from scipy.spatial import Voronoi
import plotly.graph_objects as go

st.set_page_config(page_title="Porous Membrane Analyzer", layout="wide")

# Initialize session state
if 'calculated' not in st.session_state:
    st.session_state.calculated = False

st.title("🔬 다공성 수전해 멤브레인 특성 분석기")
st.markdown("**Ionic Resistance, Gas Permeability, Bubble Point Pressure 계산**")

# Sidebar inputs
st.sidebar.header("📊 입력 파라미터")

st.sidebar.subheader("1️⃣ Pore Structure")
porosity = st.sidebar.slider("Porosity (ε)", 0.1, 0.9, 0.4, 0.01)
mean_pore_size = st.sidebar.number_input("Mean Pore Diameter (nm)", 10.0, 10000.0, 500.0, 10.0)
pore_std = st.sidebar.number_input("Pore Size Std Dev (nm)", 1.0, 5000.0, 100.0, 10.0,
                                   help="표준편차: 값이 클수록 pore 크기 분포가 넓음")
thickness = st.sidebar.number_input("Membrane Thickness (μm)", 10.0, 1000.0, 100.0, 10.0)
rve_size = st.sidebar.number_input("RVE Lateral Size (μm)", 10.0, 500.0, 100.0, 10.0,
                                   help="Representative Volume Element - 계산 영역 크기 (권장: thickness와 비슷하게)")

st.sidebar.subheader("2️⃣ Electrolyte Properties")
bulk_conductivity = st.sidebar.number_input("Bulk Ionic Conductivity (S/cm)", 0.01, 2.0, 0.8, 0.01, 
                                            help="예: 30% KOH ~ 0.8 S/cm")
electrolyte_viscosity = st.sidebar.number_input("Electrolyte Viscosity (mPa·s)", 0.1, 10.0, 1.0, 0.1)

st.sidebar.subheader("3️⃣ Gas-Liquid Interface")
surface_tension = st.sidebar.number_input("Surface Tension (mN/m)", 10.0, 100.0, 72.0, 1.0,
                                         help="Water ~ 72 mN/m")
contact_angle = st.sidebar.slider("Contact Angle (°)", 0, 180, 60, 1)

st.sidebar.subheader("4️⃣ PNM Settings")
enable_pnm = st.sidebar.checkbox("PNM 시뮬레이션 실행", value=True,
                                 help="체크 해제하면 해석 모델만 계산 (빠름)")
if enable_pnm:
    max_pores = st.sidebar.number_input("최대 Pore 개수", 1000, 50000, 15000, 1000,
                                       help="많을수록 정확하지만 느림. 권장: 10,000-20,000")

st.sidebar.markdown("---")
run_calculation = st.sidebar.button("🚀 Run Calculation", type="primary", use_container_width=True)

# Convert units
thickness_m = thickness * 1e-6  # μm to m
thickness_cm = thickness * 1e-4  # μm to cm
mean_pore_m = mean_pore_size * 1e-9  # nm to m
pore_std_m = pore_std * 1e-9  # nm to m
rve_size_m = rve_size * 1e-6  # μm to m
surface_tension_N = surface_tension * 1e-3  # mN/m to N/m
contact_angle_rad = np.radians(contact_angle)
viscosity_Pa_s = electrolyte_viscosity * 1e-3  # mPa·s to Pa·s

# ============================================================================
# ANALYTICAL MODELS
# ============================================================================

def calculate_analytical_models():
    results = {}
    
    # Ionic Resistance
    tortuosity_bruggeman = porosity ** (-0.5)
    conductivity_bruggeman = bulk_conductivity * porosity ** 1.5
    resistance_bruggeman = thickness_cm / conductivity_bruggeman
    
    conductivity_archie = bulk_conductivity * porosity ** 2
    resistance_archie = thickness_cm / conductivity_archie
    
    results['ionic'] = {
        'bruggeman': {
            'tortuosity': tortuosity_bruggeman,
            'conductivity': conductivity_bruggeman,
            'resistance': resistance_bruggeman
        },
        'archie': {
            'conductivity': conductivity_archie,
            'resistance': resistance_archie
        }
    }
    
    # Gas Permeability
    K_kozeny = (mean_pore_m ** 2 * porosity ** 3) / (180 * (1 - porosity) ** 2)
    K_kozeny_darcy = K_kozeny * 1e12
    
    c_factor = 100
    K_pore = (porosity * mean_pore_m ** 2) / c_factor
    K_pore_darcy = K_pore * 1e12
    
    results['permeability'] = {
        'kozeny_carman': {
            'permeability_m2': K_kozeny,
            'permeability_darcy': K_kozeny_darcy
        },
        'pore_model': {
            'permeability_m2': K_pore,
            'permeability_darcy': K_pore_darcy
        }
    }
    
    # Bubble Point Pressure
    d_max = mean_pore_m + 3 * pore_std_m
    BPP_laplace = 4 * surface_tension_N * np.cos(contact_angle_rad) / d_max
    BPP_laplace_bar = BPP_laplace * 1e-5
    
    BPP_mean = 4 * surface_tension_N * np.cos(contact_angle_rad) / mean_pore_m
    BPP_mean_bar = BPP_mean * 1e-5
    
    results['bpp'] = {
        'largest_pore': {
            'diameter': d_max * 1e6,  # m to μm for display
            'bpp_pa': BPP_laplace,
            'bpp_bar': BPP_laplace_bar
        },
        'mean_pore': {
            'bpp_pa': BPP_mean,
            'bpp_bar': BPP_mean_bar
        }
    }
    
    return results

# ============================================================================
# PORE NETWORK MODEL
# ============================================================================

class PoreNetworkModel:
    def __init__(self, size, porosity, mean_pore, pore_std, thickness, rve_size, coord_num, max_pores=15000):
        self.size = size
        self.target_porosity = porosity
        self.mean_pore = mean_pore
        self.pore_std = pore_std
        self.thickness = thickness
        self.rve_size = rve_size
        self.coord_num = coord_num
        self.max_pores = max_pores
        
        # Calculate required number of pores to match porosity
        self.n_pores = self.calculate_n_pores()
        
        self.generate_network()
    
    def calculate_n_pores(self):
        """Calculate number of pores needed to achieve target porosity"""
        # Assume mean sphere volume
        mean_pore_volume = (4/3) * np.pi * (self.mean_pore / 2) ** 3
        rve_volume = self.rve_size ** 3
        
        # Required total pore volume
        required_pore_volume = self.target_porosity * rve_volume
        
        # Number of pores (with overlap factor ~1.5 to account for connectivity)
        n_pores = int(required_pore_volume / mean_pore_volume * 1.5)
        
        # Ensure minimum and maximum limits
        n_pores = max(1000, min(n_pores, self.max_pores))
        
        return n_pores
    
    def generate_network(self):
        """Generate Voronoi-based pore network in cubic RVE"""
        np.random.seed(42)
        
        # Generate random pore centers in 3D cubic domain
        # ALL dimensions use rve_size (cubic domain)
        self.coords = np.random.rand(self.n_pores, 3) * self.rve_size
        
        # Pore diameters (log-normal distribution)
        sigma = np.log(1 + (self.pore_std / self.mean_pore) ** 2)
        mu = np.log(self.mean_pore) - sigma ** 2 / 2
        self.pore_diameters = np.random.lognormal(mu, sigma, self.n_pores)
        self.pore_diameters = np.clip(self.pore_diameters, self.mean_pore * 0.1, self.mean_pore * 5)
        
        # Create connectivity using Voronoi tessellation
        self.create_voronoi_connectivity()
        
        # Calculate actual porosity
        self.calculate_actual_porosity()
    
    def calculate_actual_porosity(self):
        """Calculate actual porosity (overlap ignored)"""
        pore_volumes = (4/3) * np.pi * (self.pore_diameters / 2) ** 3
        total_pore_volume = np.sum(pore_volumes)
        rve_volume = self.rve_size ** 3
        self.actual_porosity = total_pore_volume / rve_volume
    
    def create_voronoi_connectivity(self):
        """Create throat connections using Voronoi neighbors"""
        # Use Delaunay triangulation (dual of Voronoi)
        from scipy.spatial import Delaunay
        
        tri = Delaunay(self.coords)
        
        # Build neighbor list from simplices
        neighbors = [set() for _ in range(self.n_pores)]
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    neighbors[simplex[i]].add(simplex[j])
                    neighbors[simplex[j]].add(simplex[i])
        
        # Create throats
        throats = []
        throat_diameters = []
        throat_lengths = []
        
        for i in range(self.n_pores):
            for j in neighbors[i]:
                if i < j:  # Avoid duplicates
                    throats.append([i, j])
                    
                    # Throat diameter = minimum of connected pores
                    d_throat = min(self.pore_diameters[i], self.pore_diameters[j])
                    throat_diameters.append(d_throat)
                    
                    # Throat length = distance between pore centers
                    length = np.linalg.norm(self.coords[i] - self.coords[j])
                    throat_lengths.append(length)
        
        self.throats = np.array(throats)
        self.throat_diameters = np.array(throat_diameters)
        self.throat_lengths = np.array(throat_lengths)
        self.n_throats = len(throats)
    
    def calculate_ionic_resistance(self, bulk_conductivity):
        """Calculate ionic resistance - bulk_conductivity in S/m"""
        throat_areas = np.pi * (self.throat_diameters / 2) ** 2
        conductances = bulk_conductivity * throat_areas / self.throat_lengths
        
        G = sparse.lil_matrix((self.n_pores, self.n_pores))
        
        for idx, (i, j) in enumerate(self.throats):
            g = conductances[idx]
            G[i, i] += g
            G[j, j] += g
            G[i, j] -= g
            G[j, i] -= g
        
        # Identify inlet/outlet pores based on z-coordinate
        z_coords = self.coords[:, 2]
        inlet_pores = np.where(z_coords < self.rve_size * 0.1)[0]
        outlet_pores = np.where(z_coords > self.rve_size * 0.9)[0]
        
        internal_pores = np.setdiff1d(np.arange(self.n_pores), 
                                     np.concatenate([inlet_pores, outlet_pores]))
        
        G_internal = G[internal_pores][:, internal_pores].tocsr()
        
        I = np.zeros(len(internal_pores))
        
        for idx, pore in enumerate(internal_pores):
            for throat_idx, (i, j) in enumerate(self.throats):
                if i == pore and j in inlet_pores:
                    I[idx] += conductances[throat_idx] * 1.0
                elif j == pore and i in inlet_pores:
                    I[idx] += conductances[throat_idx] * 1.0
        
        try:
            V_internal = linalg.spsolve(G_internal, I)
        except:
            return None
        
        total_current = 0
        for throat_idx, (i, j) in enumerate(self.throats):
            if i in inlet_pores and j in internal_pores:
                j_internal_idx = np.where(internal_pores == j)[0][0]
                total_current += conductances[throat_idx] * (1.0 - V_internal[j_internal_idx])
            elif j in inlet_pores and i in internal_pores:
                i_internal_idx = np.where(internal_pores == i)[0][0]
                total_current += conductances[throat_idx] * (1.0 - V_internal[i_internal_idx])
        
        area = self.rve_size ** 2  # Cross-sectional area
        delta_V = 1.0
        # Effective conductivity per unit length in RVE
        effective_conductivity_rve = (total_current * self.rve_size) / (area * delta_V)
        
        # Scale to actual thickness
        # R = ρL/A, so R_actual = R_rve * (thickness/rve_size)
        resistance = (self.thickness / self.rve_size) * (self.rve_size / effective_conductivity_rve) * 1e4
        
        tortuosity = bulk_conductivity / effective_conductivity_rve * self.target_porosity
        
        return {
            'resistance': resistance,
            'conductivity': effective_conductivity_rve,
            'tortuosity': tortuosity
        }
    
    def calculate_permeability(self, viscosity):
        """Calculate permeability using Hagen-Poiseuille"""
        throat_radii = self.throat_diameters / 2
        hydraulic_conductances = (np.pi * throat_radii ** 4) / (8 * viscosity * self.throat_lengths)
        
        G = sparse.lil_matrix((self.n_pores, self.n_pores))
        
        for idx, (i, j) in enumerate(self.throats):
            g = hydraulic_conductances[idx]
            G[i, i] += g
            G[j, j] += g
            G[i, j] -= g
            G[j, i] -= g
        
        # Identify inlet/outlet based on z-coordinate
        z_coords = self.coords[:, 2]
        inlet_pores = np.where(z_coords < self.rve_size * 0.1)[0]
        outlet_pores = np.where(z_coords > self.rve_size * 0.9)[0]
        
        internal_pores = np.setdiff1d(np.arange(self.n_pores), 
                                     np.concatenate([inlet_pores, outlet_pores]))
        
        G_internal = G[internal_pores][:, internal_pores].tocsr()
        
        Q = np.zeros(len(internal_pores))
        
        for idx, pore in enumerate(internal_pores):
            for throat_idx, (i, j) in enumerate(self.throats):
                if i == pore and j in inlet_pores:
                    Q[idx] += hydraulic_conductances[throat_idx] * 1.0
                elif j == pore and i in inlet_pores:
                    Q[idx] += hydraulic_conductances[throat_idx] * 1.0
        
        try:
            P_internal = linalg.spsolve(G_internal, Q)
        except:
            return None
        
        total_flow = 0
        for throat_idx, (i, j) in enumerate(self.throats):
            if i in inlet_pores and j in internal_pores:
                j_internal_idx = np.where(internal_pores == j)[0][0]
                total_flow += hydraulic_conductances[throat_idx] * (1.0 - P_internal[j_internal_idx])
            elif j in inlet_pores and i in internal_pores:
                i_internal_idx = np.where(internal_pores == i)[0][0]
                total_flow += hydraulic_conductances[throat_idx] * (1.0 - P_internal[i_internal_idx])
        
        area = self.rve_size ** 2
        delta_P = 1.0
        # Permeability in RVE
        permeability_rve = (total_flow * viscosity * self.rve_size) / (area * delta_P)
        
        # Permeability is intrinsic property - doesn't scale with thickness
        # (thickness only affects pressure drop, not permeability itself)
        permeability = permeability_rve
        
        return {
            'permeability_m2': permeability,
            'permeability_darcy': permeability * 1e12
        }
    
    def calculate_bpp(self, surface_tension, contact_angle):
        """Calculate bubble point pressure with invasion curve and animation data"""
        capillary_pressures = 4 * surface_tension * np.cos(contact_angle) / self.throat_diameters
        
        # Identify inlet/outlet based on z-coordinate
        z_coords = self.coords[:, 2]
        inlet_pores = set(np.where(z_coords < self.rve_size * 0.1)[0])
        outlet_pores = set(np.where(z_coords > self.rve_size * 0.9)[0])
        
        sorted_indices = np.argsort(capillary_pressures)
        
        invaded_pores = set(inlet_pores)
        invasion_curve = []  # Store (pressure, saturation)
        invasion_frames = []  # Store invasion state for animation
        
        # Store frames at regular intervals
        frame_interval = max(1, len(sorted_indices) // 50)  # 50 frames max
        
        for idx, throat_idx in enumerate(sorted_indices):
            i, j = self.throats[throat_idx]
            
            if i in invaded_pores and j not in invaded_pores:
                invaded_pores.add(j)
            elif j in invaded_pores and i not in invaded_pores:
                invaded_pores.add(i)
            
            # Record saturation
            saturation = len(invaded_pores) / self.n_pores
            invasion_curve.append((capillary_pressures[throat_idx], saturation))
            
            # Store frame for animation
            if idx % frame_interval == 0 or (invaded_pores & outlet_pores):
                invasion_frames.append({
                    'step': idx,
                    'pressure': capillary_pressures[throat_idx],
                    'invaded_pores': invaded_pores.copy(),
                    'saturation': saturation
                })
            
            # Check for breakthrough
            if invaded_pores & outlet_pores:
                bpp = capillary_pressures[throat_idx]
                breakthrough_diameter = self.throat_diameters[throat_idx]
                return {
                    'bpp_pa': bpp,
                    'bpp_bar': bpp * 1e-5,
                    'breakthrough_diameter': breakthrough_diameter * 1e6,  # m to μm
                    'invasion_curve': invasion_curve,
                    'invasion_frames': invasion_frames
                }
        
        return None

# ============================================================================
# RUN CALCULATIONS
# ============================================================================

if run_calculation:
    with st.spinner("계산 중..."):
        # Analytical models (fast)
        st.session_state.analytical = calculate_analytical_models()
        
        if enable_pnm:
            # PNM simulation with progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Generate network
            status_text.text("🔨 Pore network 생성 중...")
            progress_bar.progress(10)
            
            st.session_state.pnm = PoreNetworkModel(
                size=None,
                porosity=porosity,
                mean_pore=mean_pore_m,
                pore_std=pore_std_m,
                thickness=thickness_m,
                rve_size=rve_size_m,
                coord_num=None,
                max_pores=max_pores
            )
            progress_bar.progress(30)
            
            # Step 2: Ionic resistance
            status_text.text("⚡ Ionic resistance 계산 중...")
            st.session_state.pnm_ionic = st.session_state.pnm.calculate_ionic_resistance(bulk_conductivity * 100)
            progress_bar.progress(50)
            
            # Step 3: Permeability
            status_text.text("💨 Gas permeability 계산 중...")
            st.session_state.pnm_perm = st.session_state.pnm.calculate_permeability(viscosity_Pa_s)
            progress_bar.progress(70)
            
            # Step 4: BPP
            status_text.text("💧 Bubble point pressure 계산 중...")
            st.session_state.pnm_bpp = st.session_state.pnm.calculate_bpp(surface_tension_N, contact_angle_rad)
            progress_bar.progress(100)
            
            status_text.text("✅ 계산 완료!")
            progress_bar.empty()
            status_text.empty()
        else:
            # Skip PNM
            st.session_state.pnm = None
            st.session_state.pnm_ionic = None
            st.session_state.pnm_perm = None
            st.session_state.pnm_bpp = None
        
        st.session_state.calculated = True

# Display results if calculated
if st.session_state.get('calculated', False) and hasattr(st.session_state, 'analytical'):
    analytical = st.session_state.analytical
    pnm = st.session_state.pnm
    pnm_ionic = st.session_state.pnm_ionic
    pnm_perm = st.session_state.pnm_perm
    pnm_bpp = st.session_state.pnm_bpp
    
    st.success("✅ 계산 완료!")
    
    # Display porosity info only if PNM was run
    if pnm is not None:
        st.info(f"**PNM 정보**\n\n"
                f"- Pore 개수: {pnm.n_pores:,}개\n"
                f"- 실제 Porosity: {pnm.actual_porosity:.3f} (목표: {porosity:.3f})\n"
                f"- 오차: {abs(pnm.actual_porosity - porosity)/porosity*100:.1f}%\n\n"
                f"*Note: Sphere volume 기준, overlap 무시*")
    
    st.header("📊 계산 결과")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("⚡ Ionic Resistance")
        st.markdown("**해석 모델:**")
        st.metric("Bruggeman", 
                 f"{analytical['ionic']['bruggeman']['resistance']:.4f} Ω·cm²",
                 delta=f"τ = {analytical['ionic']['bruggeman']['tortuosity']:.2f}")
        st.metric("Archie's Law", 
                 f"{analytical['ionic']['archie']['resistance']:.4f} Ω·cm²")
        
        if pnm_ionic:
            st.markdown("**PNM 시뮬레이션:**")
            st.metric("Network Model", 
                     f"{pnm_ionic['resistance']:.4f} Ω·cm²",
                     delta=f"τ = {pnm_ionic['tortuosity']:.2f}")
            error = abs(pnm_ionic['resistance'] - analytical['ionic']['bruggeman']['resistance']) / \
                    analytical['ionic']['bruggeman']['resistance'] * 100
            st.info(f"Bruggeman 대비 오차: {error:.1f}%")
    
    with col2:
        st.subheader("💨 Gas Permeability")
        st.markdown("**해석 모델:**")
        st.metric("Kozeny-Carman", 
                 f"{analytical['permeability']['kozeny_carman']['permeability_darcy']:.2e} Darcy")
        st.metric("Pore Model", 
                 f"{analytical['permeability']['pore_model']['permeability_darcy']:.2e} Darcy")
        
        if pnm_perm:
            st.markdown("**PNM 시뮬레이션:**")
            st.metric("Network Model", 
                     f"{pnm_perm['permeability_darcy']:.2e} Darcy")
            error = abs(pnm_perm['permeability_darcy'] - 
                       analytical['permeability']['kozeny_carman']['permeability_darcy']) / \
                    analytical['permeability']['kozeny_carman']['permeability_darcy'] * 100
            st.info(f"Kozeny-Carman 대비 오차: {error:.1f}%")
    
    with col3:
        st.subheader("💧 Bubble Point Pressure")
        st.markdown("**해석 모델:**")
        st.metric("Largest Pore", 
                 f"{analytical['bpp']['largest_pore']['bpp_bar']:.3f} bar",
                 delta=f"d = {analytical['bpp']['largest_pore']['diameter']*1000:.1f} nm")
        st.metric("Mean Pore", 
                 f"{analytical['bpp']['mean_pore']['bpp_bar']:.3f} bar")
        
        if pnm_bpp:
            st.markdown("**PNM 시뮬레이션:**")
            st.metric("Invasion Percolation", 
                     f"{pnm_bpp['bpp_bar']:.3f} bar",
                     delta=f"d = {pnm_bpp['breakthrough_diameter']*1000:.1f} nm")
            error = abs(pnm_bpp['bpp_bar'] - analytical['bpp']['largest_pore']['bpp_bar']) / \
                    analytical['bpp']['largest_pore']['bpp_bar'] * 100
            st.info(f"Largest Pore 대비 오차: {error:.1f}%")
    
    # Visualizations
    st.header("📈 시각화")
    
    if pnm is None:
        st.warning("⚠️ PNM 시뮬레이션이 비활성화되었습니다. 시각화를 보려면 사이드바에서 'PNM 시뮬레이션 실행'을 체크하세요.")
    else:
        # Invasion Percolation Curve
        if pnm_bpp and 'invasion_curve' in pnm_bpp:
        st.subheader("💧 Invasion Percolation Curve")
        
        invasion_data = np.array(pnm_bpp['invasion_curve'])
        pressures_bar = invasion_data[:, 0] * 1e-5
        saturations = invasion_data[:, 1]
        
        fig_invasion = go.Figure()
        fig_invasion.add_trace(go.Scatter(
            x=saturations * 100,
            y=pressures_bar,
            mode='lines',
            name='Invasion Curve',
            line=dict(color='steelblue', width=2)
        ))
        
        # Mark breakthrough point
        fig_invasion.add_trace(go.Scatter(
            x=[saturations[-1] * 100],
            y=[pnm_bpp['bpp_bar']],
            mode='markers',
            name='Breakthrough',
            marker=dict(color='red', size=12, symbol='star')
        ))
        
        fig_invasion.update_layout(
            title="Invasion Percolation: Capillary Pressure vs Gas Saturation",
            xaxis_title="Gas Saturation (%)",
            yaxis_title="Capillary Pressure (bar)",
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig_invasion, use_container_width=True)
    
    # Invasion Percolation Animation
    if pnm_bpp and 'invasion_frames' in pnm_bpp:
        st.subheader("🎬 Invasion Percolation Animation (X-Z Side View)")
        
        frames = pnm_bpp['invasion_frames']
        
        # Create frames for animation
        fig_frames = []
        
        # Select middle Y-slice for X-Z visualization (side view)
        y_coords = pnm.coords[:, 1]
        y_mid = pnm.rve_size / 2
        # Adaptive tolerance based on pore density
        y_tolerance = pnm.rve_size / max(10, int(pnm.n_pores ** (1/3) / 2))
        slice_mask = np.abs(y_coords - y_mid) < y_tolerance
        slice_pores = np.where(slice_mask)[0]
        
        z_coords = pnm.coords[:, 2]
        
        for frame_idx, frame in enumerate(frames):
            invaded = frame['invaded_pores']
            
            # Pore colors: red (invaded), blue (uninvaded), green (inlet), yellow (outlet)
            colors = []
            for p in slice_pores:
                if p in invaded:
                    colors.append('red')
                elif z_coords[p] < pnm.rve_size * 0.1:
                    colors.append('green')
                elif z_coords[p] > pnm.rve_size * 0.9:
                    colors.append('yellow')
                else:
                    colors.append('lightblue')
            
            # Reduce marker size
            sizes = [pnm.pore_diameters[p] * 1e9 * 0.5 for p in slice_pores]  # nm scale * 0.5
            
            x_coords_slice = [pnm.coords[p, 0] * 1e6 for p in slice_pores]  # X axis
            z_coords_slice = [pnm.coords[p, 2] * 1e6 for p in slice_pores]  # Z axis (flow direction)
            
            # Create scatter plot for this frame
            trace = go.Scatter(
                x=x_coords_slice,
                y=z_coords_slice,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(width=1, color='black')
                ),
                text=[f"Pore {p}" for p in slice_pores],
                hoverinfo='text'
            )
            
            # Add throats
            throat_x = []
            throat_z = []
            for i, j in pnm.throats:
                if i in slice_pores and j in slice_pores:
                    throat_x.extend([pnm.coords[i, 0] * 1e6, pnm.coords[j, 0] * 1e6, None])
                    throat_z.extend([pnm.coords[i, 2] * 1e6, pnm.coords[j, 2] * 1e6, None])
            
            throat_trace = go.Scatter(
                x=throat_x,
                y=throat_z,
                mode='lines',
                line=dict(color='gray', width=0.5),
                hoverinfo='skip',
                showlegend=False
            )
            
            frame_data = go.Frame(
                data=[throat_trace, trace],
                name=str(frame_idx),
                layout=go.Layout(
                    title=f"Invasion Step {frame['step']}<br>P = {frame['pressure']*1e-5:.3f} bar, Saturation = {frame['saturation']*100:.1f}%"
                )
            )
            fig_frames.append(frame_data)
        
        # Initial frame
        initial_invaded = frames[0]['invaded_pores']
        colors_init = []
        for p in slice_pores:
            if p in initial_invaded:
                colors_init.append('red')
            elif z_coords[p] < pnm.rve_size * 0.1:
                colors_init.append('green')
            elif z_coords[p] > pnm.rve_size * 0.9:
                colors_init.append('yellow')
            else:
                colors_init.append('lightblue')
        
        sizes_init = [pnm.pore_diameters[p] * 1e9 * 0.5 for p in slice_pores]  # nm scale * 0.5
        x_init = [pnm.coords[p, 0] * 1e6 for p in slice_pores]  # X
        z_init = [pnm.coords[p, 2] * 1e6 for p in slice_pores]  # Z (flow direction)
        
        # Throats
        throat_x_init = []
        throat_z_init = []
        for i, j in pnm.throats:
            if i in slice_pores and j in slice_pores:
                throat_x_init.extend([pnm.coords[i, 0] * 1e6, pnm.coords[j, 0] * 1e6, None])
                throat_z_init.extend([pnm.coords[i, 2] * 1e6, pnm.coords[j, 2] * 1e6, None])
        
        fig_anim = go.Figure(
            data=[
                go.Scatter(
                    x=throat_x_init,
                    y=throat_z_init,
                    mode='lines',
                    line=dict(color='gray', width=0.5),
                    hoverinfo='skip',
                    showlegend=False
                ),
                go.Scatter(
                    x=x_init,
                    y=z_init,
                    mode='markers',
                    marker=dict(
                        size=sizes_init,
                        color=colors_init,
                        line=dict(width=1, color='black')
                    ),
                    showlegend=False
                )
            ],
            frames=fig_frames,
            layout=go.Layout(
                title=f"Invasion Step 0<br>P = {frames[0]['pressure']*1e-5:.3f} bar, Saturation = {frames[0]['saturation']*100:.1f}%",
                xaxis=dict(title="X (μm)", scaleanchor="y", scaleratio=1),
                yaxis=dict(title="Z (μm) - Flow Direction →"),
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 100, 'redraw': True},
                                'fromcurrent': True,
                                'mode': 'immediate',
                                'transition': {'duration': 50}
                            }]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        }
                    ]
                }],
                sliders=[{
                    'steps': [
                        {
                            'args': [[f.name], {
                                'frame': {'duration': 0, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }],
                            'label': str(i),
                            'method': 'animate'
                        }
                        for i, f in enumerate(fig_frames)
                    ],
                    'active': 0,
                    'y': 0,
                    'len': 0.9,
                    'x': 0.1,
                    'xanchor': 'left',
                    'yanchor': 'top'
                }],
                height=600
            )
        )
        
        st.plotly_chart(fig_anim, use_container_width=True)
        
        st.info("🟢 초록색 = 입구 (z=0, 하단), 🟡 노란색 = 출구 (z=thickness, 상단), 🔴 빨간색 = 기체 침투, 🔵 파란색 = 전해질 포화\n\n**X-Z 단면도** (Side View): 기체가 아래(초록)에서 위(노랑)로 침투")
        
        # Pore Size Distribution
        fig_psd = go.Figure()
        pore_sizes_nm = pnm.pore_diameters * 1e9  # m to nm
        fig_psd.add_trace(go.Histogram(
            x=pore_sizes_nm,
            nbinsx=30,
            name="Pore Sizes",
            marker_color='steelblue'
        ))
        fig_psd.update_layout(
            title="Pore Size Distribution",
            xaxis_title="Pore Diameter (nm)",
            yaxis_title="Count",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_psd, use_container_width=True)
        
        # Network Structure
        st.subheader("🕸️ Network Structure (X-Z Side View)")
        fig_net, ax = plt.subplots(figsize=(10, 10))
        
        # Select pores in middle Y-slice for X-Z view
        y_coords_viz = pnm.coords[:, 1]
        y_mid = pnm.rve_size / 2
        y_tolerance = pnm.rve_size / max(10, int(pnm.n_pores ** (1/3) / 2))
        slice_pores_viz = np.where(np.abs(y_coords_viz - y_mid) < y_tolerance)[0]
        
        # Plot throats in slice
        for throat_idx, (i, j) in enumerate(pnm.throats):
            if i in slice_pores_viz and j in slice_pores_viz:
                xi = pnm.coords[i, 0]
                zi = pnm.coords[i, 2]
                xj = pnm.coords[j, 0]
                zj = pnm.coords[j, 2]
                ax.plot([xi * 1e6, xj * 1e6], [zi * 1e6, zj * 1e6], 'k-', alpha=0.3, linewidth=0.5)
        
        # Plot pores in slice
        for pore_id in slice_pores_viz:
            x = pnm.coords[pore_id, 0]
            z = pnm.coords[pore_id, 2]
            d = pnm.pore_diameters[pore_id] * 1e9  # to nm
            ax.scatter(x * 1e6, z * 1e6, s=d*0.5, c='steelblue', alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel("X (μm)")
        ax.set_ylabel("Z (μm) - Flow Direction →")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_net)
    
    # Comparison table
    st.header("📋 비교표")
    
    comparison_data = {
        "Property": ["Ionic Resistance (Ω·cm²)", "Gas Permeability (Darcy)", "BPP (bar)"],
        "Analytical (1)": [
            f"{analytical['ionic']['bruggeman']['resistance']:.4f}",
            f"{analytical['permeability']['kozeny_carman']['permeability_darcy']:.2e}",
            f"{analytical['bpp']['largest_pore']['bpp_bar']:.3f}"
        ],
        "Analytical (2)": [
            f"{analytical['ionic']['archie']['resistance']:.4f}",
            f"{analytical['permeability']['pore_model']['permeability_darcy']:.2e}",
            f"{analytical['bpp']['mean_pore']['bpp_bar']:.3f}"
        ],
        "PNM Simulation": [
            f"{pnm_ionic['resistance']:.4f}" if pnm_ionic else "N/A",
            f"{pnm_perm['permeability_darcy']:.2e}" if pnm_perm else "N/A",
            f"{pnm_bpp['bpp_bar']:.3f}" if pnm_bpp else "N/A"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Export
    st.header("💾 데이터 내보내기")
    
    export_data = {
        "Parameter": [
            "Porosity", "Mean Pore Size (μm)", "Std Dev (μm)", "Thickness (μm)",
            "Bulk Conductivity (S/cm)", "Viscosity (mPa·s)", 
            "Surface Tension (mN/m)", "Contact Angle (°)",
            "", "--- Ionic Resistance (Ω·cm²) ---",
            "Bruggeman", "Archie", "PNM",
            "", "--- Gas Permeability (Darcy) ---",
            "Kozeny-Carman", "Pore Model", "PNM",
            "", "--- BPP (bar) ---",
            "Largest Pore", "Mean Pore", "PNM"
        ],
        "Value": [
            porosity, mean_pore_size, pore_std, thickness,
            bulk_conductivity, electrolyte_viscosity,
            surface_tension, contact_angle,
            "", "",
            analytical['ionic']['bruggeman']['resistance'],
            analytical['ionic']['archie']['resistance'],
            pnm_ionic['resistance'] if pnm_ionic else "N/A",
            "", "",
            analytical['permeability']['kozeny_carman']['permeability_darcy'],
            analytical['permeability']['pore_model']['permeability_darcy'],
            pnm_perm['permeability_darcy'] if pnm_perm else "N/A",
            "", "",
            analytical['bpp']['largest_pore']['bpp_bar'],
            analytical['bpp']['mean_pore']['bpp_bar'],
            pnm_bpp['bpp_bar'] if pnm_bpp else "N/A"
        ]
    }
    
    df_export = pd.DataFrame(export_data)
    csv = df_export.to_csv(index=False)
    st.download_button(
        label="📥 CSV 다운로드",
        data=csv,
        file_name="membrane_analysis_results.csv",
        mime="text/csv"
    )
else:
    st.info("👈 사이드바에서 파라미터를 설정하고 '🚀 Run Calculation' 버튼을 눌러주세요!")