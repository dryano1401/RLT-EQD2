import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math

# Organ-specific parameters database
ORGAN_PARAMETERS = {
    "Kidneys": {
        "alpha_beta": 2.6,
        "repair_half_time": 2.5,
        "description": "Renal cortex/medulla"
    },
    "Bone Marrow": {
        "alpha_beta": 10.0,
        "repair_half_time": 0.5,
        "description": "Hematopoietic tissue"
    },
    "Liver": {
        "alpha_beta": 2.5,
        "repair_half_time": 1.5,
        "description": "Hepatocytes"
    },
    "Lungs": {
        "alpha_beta": 3.0,
        "repair_half_time": 1.5,
        "description": "Pulmonary parenchyma"
    },
    "Heart": {
        "alpha_beta": 3.0,
        "repair_half_time": 2.0,
        "description": "Myocardium"
    },
    "Spinal Cord": {
        "alpha_beta": 2.0,
        "repair_half_time": 1.5,
        "description": "Neural tissue"
    },
    "Salivary Glands": {
        "alpha_beta": 3.5,
        "repair_half_time": 1.0,
        "description": "Parotid/submandibular"
    },
    "Thyroid": {
        "alpha_beta": 10.0,
        "repair_half_time": 1.0,
        "description": "Thyroid follicular cells"
    },
    "Lacrimal Glands": {
        "alpha_beta": 3.0,
        "repair_half_time": 1.0,
        "description": "Tear-producing glands"
    },
    "Prostate": {
        "alpha_beta": 1.5,
        "repair_half_time": 1.5,
        "description": "Prostate adenocarcinoma"
    },
    "Breast": {
        "alpha_beta": 4.0,
        "repair_half_time": 1.5,
        "description": "Breast tissue/carcinoma"
    },
    "Bladder": {
        "alpha_beta": 5.0,
        "repair_half_time": 1.5,
        "description": "Bladder wall"
    },
    "Custom": {
        "alpha_beta": 3.0,
        "repair_half_time": 1.5,
        "description": "User-defined parameters"
    }
}

def calculate_g_factor_simplified(effective_half_life, repair_half_time):
    """
    Calculate G-factor for exponential dose delivery
    G = λ_eff / (λ_eff + μ_repair)
    where λ_eff = ln(2)/T_eff and μ_repair = ln(2)/T_repair
    """
    lambda_eff = 0.693 / effective_half_life      # Effective decay constant (1/h)
    mu_repair = 0.693 / repair_half_time          # Repair constant (1/h)
    
    g_factor = lambda_eff / (lambda_eff + mu_repair)
    return g_factor

def calculate_bed_radiopharm(dose, alpha_beta, effective_half_life, repair_half_time):
    """
    Calculate BED for radiopharmaceutical with exponential dose delivery
    BED = D × (1 + G × D/(α/β))
    Using G-factor: G = λ_eff / (λ_eff + μ_repair)
    """
    g_factor = calculate_g_factor_simplified(effective_half_life, repair_half_time)
    bed = dose * (1 + g_factor * dose / alpha_beta)
    return bed, g_factor

def calculate_eqd2(bed, alpha_beta):
    """
    Calculate EQD2 (Equivalent Dose in 2 Gy fractions)
    EQD2 = BED / (1 + 2/(α/β))
    """
    eqd2 = bed / (1 + 2 / alpha_beta)
    return eqd2

def calculate_time_for_99_percent_delivery(effective_half_life):
    """
    Calculate time required for 99% of dose delivery
    For exponential decay: 1 - exp(-λt) = 0.99
    Solving: t = -ln(0.01) / λ = -ln(0.01) × T_eff / ln(2)
    """
    time_99 = -np.log(0.01) * effective_half_life / np.log(2)
    return time_99

def calculate_eqd299(dose, alpha_beta, effective_half_life, repair_half_time):
    """
    Calculate EQD2₉₉ - EQD2 when 99% of dose has been delivered
    
    This represents the biological effect when 99% of the total dose
    has been delivered to the treatment site (temporal delivery concept)
    """
    # Calculate time for 99% delivery
    time_99 = calculate_time_for_99_percent_delivery(effective_half_life)
    
    # Dose delivered at 99% time point
    dose_99 = dose * 0.99
    
    # Calculate BED for 99% of the dose
    # Use modified G-factor for the partial delivery time
    # For partial delivery, we need to account for the actual delivery pattern
    g_factor_99 = calculate_g_factor_simplified(effective_half_life, repair_half_time)
    
    # BED calculation for the 99% delivery considers the dose rate pattern up to that point
    bed_99 = dose_99 * (1 + g_factor_99 * dose_99 / alpha_beta)
    
    # Convert to EQD2
    eqd299 = calculate_eqd2(bed_99, alpha_beta)
    
    return eqd299, time_99, dose_99

def calculate_eqd2max(dose, alpha_beta, effective_half_life, repair_half_time):
    """
    Calculate EQD2Max - EQD2 calculated as if the initial dose rate 
    continued through infinity to contribute to the total absorbed dose
    
    For exponential decay: dose_rate(t) = dose_rate_0 * exp(-λt)
    Initial dose rate: dose_rate_0 = D_total * λ_eff
    
    If this initial rate continued forever at constant rate:
    Total dose would be infinite, but we calculate the equivalent
    biological effect using the initial rate as if it were sustained.
    
    For sustained constant dose rate: BED = D × (1 + (2R)/(α/β × μ))
    where R is the dose rate and μ is the repair constant
    """
    # Calculate effective decay constant
    lambda_eff = 0.693 / effective_half_life  # h^-1
    mu_repair = 0.693 / repair_half_time       # h^-1
    
    # Initial dose rate (Gy/h)
    initial_dose_rate = dose * lambda_eff
    
    # For a sustained constant dose rate, the BED formula is:
    # BED = D × (1 + (2R)/(α/β × μ))
    # But since we're asking "what if this rate continued forever"
    # We need to use the actual total dose but with the initial rate characteristics
    
    # Calculate BED as if the total dose was delivered at the constant initial rate
    # This uses the sustained dose rate BED formula
    bed_max = dose * (1 + (2 * initial_dose_rate) / (alpha_beta * mu_repair))
    
    # Convert to EQD2
    eqd2max = calculate_eqd2(bed_max, alpha_beta)
    
    return eqd2max, initial_dose_rate, dose

def calculate_eqd2r(bed_target, bed_previous, alpha_beta):
    """
    Calculate EQD2R (Remaining dose after previous treatment)
    EQD2R = EQD2_total - EQD2_previous
    """
    eqd2_target = calculate_eqd2(bed_target, alpha_beta)
    eqd2_previous = calculate_eqd2(bed_previous, alpha_beta)
    eqd2r = eqd2_target - eqd2_previous
    return eqd2r

def calculate_dose_rate_factor(effective_half_life, repair_half_time):
    """
    Calculate dose rate factor for different half-lives
    Using the same approach as G-factor
    """
    lambda_eff = 0.693 / effective_half_life
    mu_repair = 0.693 / repair_half_time
    
    # Dose rate factor using same formula as G-factor
    drf = lambda_eff / (lambda_eff + mu_repair)
    return drf

def main():
    st.set_page_config(
        page_title="Enhanced Radiopharmaceutical Dosimetry",
        page_icon="⚛️",
        layout="wide"
    )
    
    st.title("⚛️ Enhanced Radiopharmaceutical Dosimetry Calculator")
    st.markdown("Advanced calculator for EQD2, EQD2₉₉, EQD2Max, and EQD2R in radiopharmaceutical therapy")
    
    # Sidebar for organ selection
    st.sidebar.header("🎯 Organ Selection")
    
    selected_organ = st.sidebar.selectbox(
        "Select target organ:",
        list(ORGAN_PARAMETERS.keys())
    )
    
    # Display organ information
    organ_info = ORGAN_PARAMETERS[selected_organ]
    st.sidebar.info(f"**{selected_organ}**\n{organ_info['description']}")
    
    # Organ parameters
    if selected_organ == "Custom":
        alpha_beta = st.sidebar.number_input("α/β ratio (Gy):", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
        repair_half_time = st.sidebar.number_input("Repair half-time (hours):", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
    else:
        alpha_beta = organ_info["alpha_beta"]
        repair_half_time = organ_info["repair_half_time"]
        st.sidebar.write(f"α/β ratio: {alpha_beta} Gy")
        st.sidebar.write(f"Repair t₁/₂: {repair_half_time} h")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧮 Primary Calculation", 
        "📊 Advanced Metrics", 
        "🔄 Treatment Planning", 
        "📈 Dose-Response Analysis",
        "⚖️ Safety Assessment"
    ])
    
    with tab1:
        st.header("Primary Radiopharmaceutical Dosimetry")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Treatment Parameters")
            organ_dose = st.number_input(
                "Organ Absorbed Dose (Gy):", 
                min_value=0.01, 
                max_value=200.0, 
                value=10.0, 
                step=0.1,
                help="Total absorbed dose to the organ from the radiopharmaceutical"
            )
            
            organ_effective_half_life = st.number_input(
                "Organ Effective Half-life (hours):", 
                min_value=0.1, 
                max_value=2000.0, 
                value=67.0, 
                step=0.1,
                help="Effective half-life of the radiopharmaceutical in the specific organ"
            )
            
            st.info(f"**Selected Organ:** {selected_organ}\n**α/β:** {alpha_beta} Gy\n**Repair t₁/₂:** {repair_half_time} h")
            
            if st.button("Calculate Primary Dosimetry", type="primary"):
                # Calculate BED and G-factor
                bed, g_factor = calculate_bed_radiopharm(organ_dose, alpha_beta, organ_effective_half_life, repair_half_time)
                
                # Calculate EQD2
                eqd2 = calculate_eqd2(bed, alpha_beta)
                
                # Calculate dose rate factor
                drf = calculate_dose_rate_factor(organ_effective_half_life, repair_half_time)
                
                # Store results
                st.session_state.primary_results = {
                    'organ_dose': organ_dose,
                    'bed': bed,
                    'eqd2': eqd2,
                    'g_factor': g_factor,
                    'drf': drf,
                    'organ': selected_organ,
                    'alpha_beta': alpha_beta,
                    'repair_half_time': repair_half_time,
                    'effective_half_life': organ_effective_half_life
                }
        
        with col2:
            st.subheader("Primary Results")
            if 'primary_results' in st.session_state:
                results = st.session_state.primary_results
                
                # Key metrics
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Organ Dose", f"{results['organ_dose']:.2f} Gy")
                    st.metric("BED", f"{results['bed']:.2f} Gy", help="Biologically Effective Dose")
                
                with col2b:
                    st.metric("EQD2", f"{results['eqd2']:.2f} Gy", help="Equivalent Dose in 2 Gy fractions")
                    st.metric("G-factor", f"{results['g_factor']:.4f}", help="Repair correction factor")
                
                # Additional parameters
                st.write("**Dosimetric Parameters:**")
                st.write(f"• Dose Rate Factor: {results['drf']:.3f}")
                st.write(f"• Effective Half-life: {results['effective_half_life']:.1f} hours")
                st.write(f"• Repair Half-time: {results['repair_half_time']:.1f} hours")
                
                # Show G-factor formula
                lambda_eff = 0.693 / results['effective_half_life']
                mu_repair = 0.693 / results['repair_half_time']
                st.info(f"**G-factor Formula:** G = λ_eff/(λ_eff + μ_repair) = {lambda_eff:.4f}/({lambda_eff:.4f} + {mu_repair:.4f}) = {results['g_factor']:.4f}")
                st.write(f"Where: λ_eff = ln(2)/T_eff = 0.693/{results['effective_half_life']:.1f} = {lambda_eff:.4f} h⁻¹")
                st.write(f"μ_repair = ln(2)/T_repair = 0.693/{results['repair_half_time']:.1f} = {mu_repair:.4f} h⁻¹")
    
    with tab2:
        st.header("Advanced Dosimetric Metrics")
        
        if 'primary_results' in st.session_state:
            results = st.session_state.primary_results
            
            # Calculate advanced metrics
            eqd299, time_99, dose_99 = calculate_eqd299(results['organ_dose'], results['alpha_beta'], 
                                                       results['effective_half_life'], results['repair_half_time'])
            eqd2max, initial_dose_rate, total_dose_used = calculate_eqd2max(results['organ_dose'], results['alpha_beta'], 
                                                                           results['effective_half_life'], results['repair_half_time'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "EQD2₉₉", 
                    f"{eqd299:.2f} Gy",
                    help="EQD2 when 99% of dose has been delivered"
                )
                
                st.write(f"**Delivery Details:**")
                st.write(f"• Time to 99%: {time_99:.1f} hours")
                st.write(f"• Dose at 99%: {dose_99:.2f} Gy")
                
                # Show temporal progression
                days_99 = time_99 / 24
                st.info(f"99% delivery in {days_99:.1f} days")
            
            with col2:
                st.metric(
                    "EQD2Max", 
                    f"{eqd2max:.2f} Gy",
                    help="EQD2 if initial dose rate continued forever"
                )
                
                st.write(f"**Dose Rate Details:**")
                st.write(f"• Initial rate: {initial_dose_rate:.4f} Gy/h")
                st.write(f"• Total dose: {total_dose_used:.2f} Gy")
                st.write(f"• Rate assumption: Constant")
                
                # Rate comparison
                rate_ratio = results['eqd2'] / eqd2max
                if rate_ratio <= 0.3:
                    st.success(f"✅ Major rate reduction benefit ({rate_ratio:.2f})")
                elif rate_ratio <= 0.6:
                    st.info(f"ℹ️ Significant rate reduction benefit ({rate_ratio:.2f})")
                elif rate_ratio <= 0.8:
                    st.warning(f"⚠️ Moderate rate reduction benefit ({rate_ratio:.2f})")
                else:
                    st.error(f"❌ Limited rate reduction benefit ({rate_ratio:.2f})")
            
            with col3:
                # Calculate dose rate benefit
                dose_rate_benefit = eqd2max - results['eqd2']
                st.metric(
                    "Dose Rate Benefit", 
                    f"{dose_rate_benefit:.2f} Gy",
                    help="EQD2 reduction due to exponential vs constant rate delivery"
                )
                
                if dose_rate_benefit > 0:
                    st.info(f"Exponential decay advantage")
                else:
                    st.warning(f"No dose rate advantage")
            
            # Temporal delivery analysis
            st.subheader("Temporal Delivery Analysis")
            
            # Calculate dose delivery over time
            time_points = np.linspace(0, time_99 * 1.2, 100)  # Extended beyond 99%
            delivered_fraction = 1 - np.exp(-np.log(2) * time_points / results['effective_half_life'])
            delivered_dose = delivered_fraction * results['organ_dose']
            
            # Calculate corresponding EQD2 values over time
            eqd2_time = []
            for dose_t in delivered_dose:
                if dose_t > 0:
                    bed_t, _ = calculate_bed_radiopharm(dose_t, results['alpha_beta'], 
                                                     results['effective_half_life'], results['repair_half_time'])
                    eqd2_t = calculate_eqd2(bed_t, results['alpha_beta'])
                    eqd2_time.append(eqd2_t)
                else:
                    eqd2_time.append(0)
            
            # Create temporal plot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Cumulative Dose Delivery', 'Cumulative EQD2'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Dose delivery plot
            fig.add_trace(
                go.Scatter(x=time_points, y=delivered_dose, mode='lines', name='Delivered Dose', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_vline(x=time_99, line_dash="dash", line_color="red", annotation_text="99% delivery")
            fig.add_vline(x=24, line_dash="dash", line_color="green", annotation_text="24h")
            
            # EQD2 plot
            fig.add_trace(
                go.Scatter(x=time_points, y=eqd2_time, mode='lines', name='EQD2', line=dict(color='green')),
                row=1, col=2
            )
            fig.add_vline(x=time_99, line_dash="dash", line_color="red", annotation_text="99% delivery")
            fig.add_vline(x=24, line_dash="dash", line_color="green", annotation_text="24h")
            
            fig.update_xaxes(title_text="Time (hours)")
            fig.update_yaxes(title_text="Dose (Gy)", row=1, col=1)
            fig.update_yaxes(title_text="EQD2 (Gy)", row=1, col=2)
            fig.update_layout(height=400, title_text="Temporal Dose Delivery Analysis")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed breakdown
            st.subheader("Detailed Analysis")
            
            analysis_data = {
                'Metric': ['Total Organ Dose', 'BED', 'EQD2', 'EQD2₉₉', 'EQD2Max', 'Dose Rate Benefit'],
                'Value (Gy)': [
                    results['organ_dose'],
                    results['bed'],
                    results['eqd2'],
                    eqd299,
                    eqd2max,
                    eqd2max - results['eqd2']
                ],
                'Interpretation': [
                    'Physical dose absorbed',
                    'Biological effectiveness',
                    'Equivalent 2 Gy fractions',
                    '99% delivery timepoint',
                    'Initial rate sustained EQD2',
                    'Exponential decay benefit'
                ]
            }
            
            df_analysis = pd.DataFrame(analysis_data)
            st.dataframe(df_analysis, use_container_width=True)
            
        else:
            st.info("Please calculate primary dosimetry first.")
    
    with tab3:
        st.header("Treatment Planning (BED-Based)")
        
        st.markdown("Calculate cumulative BED and plan treatments based on BED limits")
        
        # Kidney risk assessment for dose limits
        if selected_organ == "Kidneys":
            st.subheader("Kidney Risk Assessment")
            kidney_risk = st.radio(
                "Patient kidney risk status:",
                ["Low risk (no existing kidney disease)", "High risk (existing kidney disease/risk factors)"],
                help="Select patient's kidney risk status to determine appropriate BED limit"
            )
            
            if "Low risk" in kidney_risk:
                kidney_bed_limit = 40.0  # Gy BED
                st.info(f"🟢 **Low Risk Patient**: BED limit = {kidney_bed_limit} Gy")
            else:
                kidney_bed_limit = 28.0  # Gy BED
                st.warning(f"🟡 **High Risk Patient**: BED limit = {kidney_bed_limit} Gy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Previous Treatment History")
            
            num_previous = st.number_input("Number of previous treatments:", min_value=0, max_value=10, value=1, step=1)
            
            previous_treatments = []
            total_previous_bed = 0
            
            for i in range(num_previous):
                st.write(f"**Treatment {i+1}:**")
                prev_dose = st.number_input(f"Dose {i+1} (Gy):", min_value=0.0, value=5.0, step=0.1, key=f"prev_dose_{i}")
                prev_half_life = st.number_input(f"Half-life {i+1} (h):", min_value=0.1, value=67.0, step=0.1, key=f"prev_hl_{i}")
                
                prev_bed, _ = calculate_bed_radiopharm(prev_dose, alpha_beta, prev_half_life, repair_half_time)
                total_previous_bed += prev_bed
                
                previous_treatments.append({
                    'dose': prev_dose,
                    'half_life': prev_half_life,
                    'bed': prev_bed
                })
                
                st.write(f"   • BED: {prev_bed:.2f} Gy")
            
            if num_previous > 0:
                st.write(f"**Total Previous BED: {total_previous_bed:.2f} Gy**")
            
            st.subheader("Planned Additional Treatment")
            planned_dose = st.number_input("Planned additional dose (Gy):", min_value=0.0, value=5.0, step=0.1)
            planned_half_life = st.number_input("Planned half-life (h):", min_value=0.1, value=67.0, step=0.1)
            
            if st.button("Calculate Treatment Plan", type="primary"):
                # Calculate planned BED
                planned_bed, _ = calculate_bed_radiopharm(planned_dose, alpha_beta, planned_half_life, repair_half_time)
                
                # Total cumulative BED
                total_bed = total_previous_bed + planned_bed
                
                # Get BED limit based on organ and risk
                if selected_organ == "Kidneys":
                    bed_limit = kidney_bed_limit
                else:
                    # Convert EQD2 tolerance limits to BED limits
                    organ_eqd2_limits = {
                        "Bone Marrow": 2.0, "Liver": 30.0, "Lungs": 20.0,
                        "Heart": 26.0, "Spinal Cord": 50.0, "Salivary Glands": 26.0, "Thyroid": 45.0,
                        "Lacrimal Glands": 30.0, "Bladder": 65.0, "Prostate": 76.0, "Breast": 50.0
                    }
                    eqd2_limit = organ_eqd2_limits.get(selected_organ, 25.0)
                    # Convert EQD2 limit to BED limit: BED = EQD2 × (1 + 2/(α/β))
                    bed_limit = eqd2_limit * (1 + 2 / alpha_beta)
                
                # Calculate remaining BED capacity
                remaining_bed = bed_limit - total_previous_bed
                
                st.session_state.treatment_results = {
                    'previous_bed': total_previous_bed,
                    'planned_bed': planned_bed,
                    'total_bed': total_bed,
                    'remaining_bed': remaining_bed,
                    'bed_limit': bed_limit,
                    'bed_ratio': total_bed / bed_limit,
                    'organ': selected_organ
                }
        
        with col2:
            st.subheader("Treatment Plan Analysis")
            
            if 'treatment_results' in st.session_state:
                results = st.session_state.treatment_results
                
                # Key metrics
                st.metric("Previous BED", f"{results['previous_bed']:.2f} Gy")
                st.metric("Planned BED", f"{results['planned_bed']:.2f} Gy")
                st.metric("Total BED", f"{results['total_bed']:.2f} Gy")
                st.metric("BED Limit", f"{results['bed_limit']:.2f} Gy")
                st.metric("Remaining BED", f"{results['remaining_bed']:.2f} Gy")
                
                # Safety assessment
                st.subheader("Safety Assessment")
                bed_ratio = results['bed_ratio']
                
                if bed_ratio <= 0.8:
                    st.success(f"✅ Safe for treatment (Ratio: {bed_ratio:.2f})")
                elif bed_ratio <= 1.0:
                    st.warning(f"⚠️ Caution advised (Ratio: {bed_ratio:.2f})")
                else:
                    st.error(f"❌ Exceeds BED limit (Ratio: {bed_ratio:.2f})")
                
                # Treatment recommendations
                if results['remaining_bed'] > 0:
                    max_additional_bed = results['remaining_bed']
                    st.info(f"""
                    **Treatment Recommendations:**
                    - Maximum additional BED: {max_additional_bed:.2f} Gy
                    - Current planned BED: {results['planned_bed']:.2f} Gy
                    - Utilization: {(results['planned_bed']/max_additional_bed)*100:.1f}% of remaining capacity
                    """)
                    
                    if results['planned_bed'] > max_additional_bed:
                        st.error(f"⚠️ Planned dose exceeds remaining capacity by {results['planned_bed'] - max_additional_bed:.2f} Gy BED")
                        suggested_dose_reduction = (max_additional_bed / results['planned_bed']) * planned_dose
                        st.write(f"💡 **Suggested dose reduction:** {suggested_dose_reduction:.1f} Gy (from {planned_dose:.1f} Gy)")
                else:
                    st.error("❌ No remaining BED capacity. Treatment not recommended.")
                
                # Visual representation
                fig = go.Figure()
                
                # Stacked bar showing BED components
                fig.add_trace(go.Bar(
                    name='Previous BED',
                    x=['Current', 'After Planned'],
                    y=[results['previous_bed'], results['previous_bed']],
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Bar(
                    name='Planned BED',
                    x=['Current', 'After Planned'],
                    y=[0, results['planned_bed']],
                    marker_color='orange'
                ))
                
                # Add BED limit line
                fig.add_hline(y=results['bed_limit'], line_dash="dash", line_color="red", 
                            annotation_text=f"BED Limit: {results['bed_limit']:.1f} Gy")
                
                fig.update_layout(
                    title="BED Treatment Planning",
                    yaxis_title="BED (Gy)",
                    barmode='stack',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed breakdown table
                st.subheader("BED Analysis Summary")
                
                planning_data = {
                    'Component': ['Previous Treatments', 'Planned Treatment', 'Total BED', 'BED Limit', 'Remaining Capacity'],
                    'BED (Gy)': [
                        results['previous_bed'], 
                        results['planned_bed'], 
                        results['total_bed'], 
                        results['bed_limit'], 
                        results['remaining_bed']
                    ],
                    'Percentage of Limit': [
                        (results['previous_bed']/results['bed_limit'])*100,
                        (results['planned_bed']/results['bed_limit'])*100,
                        (results['total_bed']/results['bed_limit'])*100,
                        100.0,
                        (results['remaining_bed']/results['bed_limit'])*100
                    ],
                    'Status': [
                        'Completed',
                        'Planned',
                        'Total',
                        'Maximum',
                        'Available'
                    ]
                }
                
                planning_df = pd.DataFrame(planning_data)
                st.dataframe(planning_df, use_container_width=True)
    
    with tab4:
        st.header("Dose-Response Analysis")
        
        st.subheader("Organ Dose vs. Biological Effect")
        
        # Dose range analysis
        dose_range = st.slider("Dose range for analysis (Gy):", 1.0, 50.0, (5.0, 30.0), step=1.0)
        half_life_analysis = st.number_input("Effective half-life for analysis (h):", min_value=1.0, value=67.0, step=1.0)
        
        if st.button("Generate Dose-Response Analysis"):
            doses = np.linspace(dose_range[0], dose_range[1], 50)
            
            bed_values = []
            eqd2_values = []
            eqd299_values = []
            eqd2max_values = []
            g_factors = []
            
            for dose in doses:
                bed, g_factor = calculate_bed_radiopharm(dose, alpha_beta, half_life_analysis, repair_half_time)
                eqd2 = calculate_eqd2(bed, alpha_beta)
                eqd299, _, _ = calculate_eqd299(dose, alpha_beta, half_life_analysis, repair_half_time)
                eqd2max_dose, _, _ = calculate_eqd2max(dose, alpha_beta, half_life_analysis, repair_half_time)
                
                bed_values.append(bed)
                eqd2_values.append(eqd2)
                eqd299_values.append(eqd299)
                eqd2max_values.append(eqd2max_dose)
                g_factors.append(g_factor)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('BED vs Dose', 'EQD2 vs Dose', 'G-factor (Constant)', 'EQD2 vs EQD2₉₉ vs EQD2Max'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # BED plot
            fig.add_trace(
                go.Scatter(x=doses, y=bed_values, mode='lines', name='BED', line=dict(color='blue')),
                row=1, col=1
            )
            
            # EQD2 plot
            fig.add_trace(
                go.Scatter(x=doses, y=eqd2_values, mode='lines', name='EQD2', line=dict(color='red')),
                row=1, col=2
            )
            
            # G-factor plot (will be constant with simplified formula)
            fig.add_trace(
                go.Scatter(x=doses, y=g_factors, mode='lines', name='G-factor', line=dict(color='green')),
                row=2, col=1
            )
            
            # EQD2 vs EQD299 vs EQD2Max comparison
            fig.add_trace(
                go.Scatter(x=doses, y=eqd2_values, mode='lines', name='Total EQD2', line=dict(color='red')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=doses, y=eqd299_values, mode='lines', name='EQD2₉₉', line=dict(color='orange')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=doses, y=eqd2max_values, mode='lines', name='EQD2Max', line=dict(color='purple')),
                row=2, col=2
            )
            
            # Add organ tolerance line if available
            organ_tolerance_limits = {
                "Kidneys": 23.0, "Bone Marrow": 2.0, "Liver": 30.0, "Lungs": 20.0,
                "Heart": 26.0, "Spinal Cord": 50.0, "Salivary Glands": 26.0, "Thyroid": 45.0,
                "Lacrimal Glands": 30.0, "Bladder": 65.0, "Prostate": 76.0, "Breast": 50.0
            }
            
            if selected_organ in organ_tolerance_limits:
                tolerance_limit = organ_tolerance_limits[selected_organ]
                
                fig.add_hline(y=tolerance_limit, line_dash="dot", line_color="green", 
                            annotation_text=f"Tolerance: {tolerance_limit} Gy", row=2, col=2)
            
            fig.update_layout(height=800, title_text="Comprehensive Dose-Response Analysis")
            fig.update_xaxes(title_text="Dose (Gy)")
            fig.update_yaxes(title_text="BED (Gy)", row=1, col=1)
            fig.update_yaxes(title_text="EQD2 (Gy)", row=1, col=2)
            fig.update_yaxes(title_text="G-factor", row=2, col=1)
            fig.update_yaxes(title_text="EQD2 (Gy)", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            analysis_df = pd.DataFrame({
                'Dose (Gy)': doses,
                'BED (Gy)': bed_values,
                'EQD2 (Gy)': eqd2_values,
                'EQD2₉₉ (Gy)': eqd299_values,
                'EQD2Max (Gy)': eqd2max_values,
                'G-factor': g_factors
            })
            
            st.subheader("Dose-Response Data")
            st.dataframe(analysis_df.round(3))
    
    with tab5:
        st.header("Safety Assessment Dashboard (BED-Based)")
        
        if 'primary_results' in st.session_state:
            results = st.session_state.primary_results
            
            # Calculate safety metrics in BED
            bed_99, time_99, dose_99 = calculate_eqd299(results['organ_dose'], results['alpha_beta'], 
                                                       results['effective_half_life'], results['repair_half_time'])
            # Convert EQD2₉₉ back to BED for consistency
            bed_99_actual = bed_99 * (1 + 2 / results['alpha_beta'])
            
            bed_max, initial_dose_rate, total_dose_used = calculate_eqd2max(results['organ_dose'], results['alpha_beta'], 
                                                                          results['effective_half_life'], results['repair_half_time'])
            # Convert EQD2Max back to BED for consistency  
            bed_max_actual = bed_max * (1 + 2 / results['alpha_beta'])
            
            # Get organ BED tolerance limits
            if selected_organ == "Kidneys":
                # Use risk-stratified BED limits for kidneys
                st.subheader("Kidney Risk Assessment")
                kidney_risk = st.radio(
                    "Patient kidney risk status:",
                    ["Low risk (no existing kidney disease)", "High risk (existing kidney disease/risk factors)"],
                    help="Select patient's kidney risk status to determine appropriate BED limit",
                    key="safety_kidney_risk"
                )
                
                if "Low risk" in kidney_risk:
                    organ_bed_tolerance = 40.0  # Gy BED
                    st.info(f"🟢 **Low Risk Patient**: BED limit = {organ_bed_tolerance} Gy")
                else:
                    organ_bed_tolerance = 28.0  # Gy BED
                    st.warning(f"🟡 **High Risk Patient**: BED limit = {organ_bed_tolerance} Gy")
            else:
                # Convert EQD2 tolerance limits to BED limits for other organs
                organ_eqd2_limits = {
                    "Bone Marrow": 2.0, "Liver": 30.0, "Lungs": 20.0,
                    "Heart": 26.0, "Spinal Cord": 50.0, "Salivary Glands": 26.0, "Thyroid": 45.0,
                    "Lacrimal Glands": 30.0, "Bladder": 65.0, "Prostate": 76.0, "Breast": 50.0
                }
                eqd2_limit = organ_eqd2_limits.get(selected_organ, 25.0)
                # Convert EQD2 limit to BED limit: BED = EQD2 × (1 + 2/(α/β))
                organ_bed_tolerance = eqd2_limit * (1 + 2 / results['alpha_beta'])
            
            # Safety ratios based on BED
            tolerance_ratio = results['bed'] / organ_bed_tolerance
            delivery_ratio = results['bed'] / bed_99_actual  # Current vs 99% delivery point
            dose_rate_ratio = results['bed'] / bed_max_actual  # Current vs maximum rate effect
            
            # Safety dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current BED", f"{results['bed']:.1f} Gy")
                
            with col2:
                st.metric("BED₉₉", f"{bed_99_actual:.1f} Gy", f"At {time_99:.0f}h")
                
            with col3:
                st.metric("BEDMax", f"{bed_max_actual:.1f} Gy", f"Constant rate")
                    
            with col4:
                if tolerance_ratio <= 0.8:
                    st.metric("Tolerance Status", "✅ LOW", f"{tolerance_ratio:.2f}")
                elif tolerance_ratio <= 1.0:
                    st.metric("Tolerance Status", "⚠️ MODERATE", f"{tolerance_ratio:.2f}")
                else:
                    st.metric("Tolerance Status", "❌ HIGH", f"{tolerance_ratio:.2f}")
            
            # Visual safety assessment
            fig = go.Figure()
            
            # Add bars for different BED levels
            categories = ['Current BED', 'BED₉₉', 'BEDMax', 'BED Tolerance']
            values = [results['bed'], bed_99_actual, bed_max_actual, organ_bed_tolerance]
            colors = ['blue', 'orange', 'purple', 'red']
            
            fig.add_trace(go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f"{v:.1f} Gy" for v in values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"BED Safety Assessment for {selected_organ}",
                yaxis_title="BED (Gy)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Delivery timeline
            st.subheader("BED Delivery Timeline Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Key Timepoints:**")
                st.write(f"• 50% delivery: {0.693 * results['effective_half_life']:.1f} hours")
                st.write(f"• 90% delivery: {2.303 * results['effective_half_life']:.1f} hours")
                st.write(f"• 99% delivery: {time_99:.1f} hours ({time_99/24:.1f} days)")
                st.write(f"• 99.9% delivery: {6.908 * results['effective_half_life']:.1f} hours")
                st.write(f"• Initial dose rate: {initial_dose_rate:.4f} Gy/h")
                
            with col2:
                st.write("**Biological Effects (BED):**")
                st.write(f"• Current total BED: {results['bed']:.1f} Gy")
                st.write(f"• BED at 99% delivery: {bed_99_actual:.1f} Gy")
                st.write(f"• BED at constant rate: {bed_max_actual:.1f} Gy")
                st.write(f"• Organ BED tolerance: {organ_bed_tolerance:.1f} Gy")
                
                dose_rate_benefit = bed_max_actual - results['bed']
                st.write(f"• Dose rate benefit: {dose_rate_benefit:.1f} Gy BED")
            
            # Risk assessment table
            st.subheader("BED Risk Assessment Summary")
            
            risk_data = {
                'Parameter': ['Current Treatment', 'At 99% Delivery', 'Constant Rate Delivery', 'Organ Tolerance'],
                'BED (Gy)': [results['bed'], bed_99_actual, bed_max_actual, organ_bed_tolerance],
                'Ratio to Tolerance': [tolerance_ratio, bed_99_actual/organ_bed_tolerance, bed_max_actual/organ_bed_tolerance, 1.0],
                'Interpretation': [
                    'Actual treatment effect',
                    '99% delivery timepoint',
                    'Initial rate sustained',
                    'BED safety limit'
                ]
            }
            
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True)
            
            # Recommendations
            st.subheader("Clinical Recommendations")
            
            if tolerance_ratio <= 0.8:
                st.success("✅ **LOW RISK**: Current BED is well within safety limits. Treatment can proceed as planned.")
            elif tolerance_ratio <= 1.0:
                st.warning("⚠️ **MODERATE RISK**: BED approaches tolerance threshold. Consider close monitoring.")
            else:
                st.error("❌ **HIGH RISK**: BED exceeds tolerance threshold. Consider dose reduction or fractionation.")
            
            # Dose rate interpretation
            dose_rate_benefit = bed_max_actual - results['bed']
            if dose_rate_benefit > 0:
                st.info(f"""
                **Dose Rate Benefit (BED Analysis):**
                - Exponential decay reduces BED by {dose_rate_benefit:.1f} Gy compared to constant rate delivery
                - Initial dose rate = {initial_dose_rate:.4f} Gy/h would be sustained
                - G-factor = {results['g_factor']:.3f} indicates repair during exponential delivery
                - Exponential decay provides significant normal tissue sparing vs constant rate
                - BED reduction = {(dose_rate_benefit/bed_max_actual)*100:.1f}% sparing effect
                """)
            else:
                st.warning("⚠️ **No dose rate benefit**: Very fast effective half-life provides limited sparing")
                
            # BED interpretation
            st.info(f"""
            **BED Metrics Interpretation:**
            
            **Current BED = {results['bed']:.1f} Gy**
            - Actual biological effectiveness of the treatment
            - Accounts for dose delivery pattern and repair kinetics
            - Primary metric for safety assessment
            
            **BED₉₉ = {bed_99_actual:.1f} Gy**
            - BED when 99% of dose has been delivered
            - Occurs at t = {time_99:.1f} hours ({time_99/24:.1f} days) after administration
            - Shows temporal progression of biological effects
            
            **BEDMax = {bed_max_actual:.1f} Gy**
            - BED if initial dose rate continued forever
            - Initial dose rate = {initial_dose_rate:.4f} Gy/h sustained constantly
            - Maximum possible biological effect from this dose rate pattern
            - Shows protective effect of exponential decay
            
            **Organ BED Tolerance = {organ_bed_tolerance:.1f} Gy**
            {"- Risk-stratified limit for kidney patients" if selected_organ == "Kidneys" else "- Organ-specific safety threshold"}
            - Based on clinical outcome data and NTCP models
            - Primary safety constraint for treatment planning
            """)
            
            # Additional safety metrics
            st.subheader("Additional Safety Metrics")
            
            remaining_bed = organ_bed_tolerance - results['bed']
            utilization = (results['bed'] / organ_bed_tolerance) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Remaining BED Capacity", f"{remaining_bed:.1f} Gy")
            with col2:
                st.metric("BED Utilization", f"{utilization:.1f}%")
            with col3:
                safety_margin = (organ_bed_tolerance - results['bed']) / organ_bed_tolerance * 100
                st.metric("Safety Margin", f"{safety_margin:.1f}%")
                
            if remaining_bed > 0:
                st.success(f"✅ **Available for future treatments:** {remaining_bed:.1f} Gy BED remaining")
            else:
                st.error("❌ **No capacity for re-treatment:** BED limit exceeded")
        
        else:
            st.info("Please calculate primary dosimetry first to see safety assessment.")
    
    # Footer with methodology
    st.markdown("---")
    st.markdown("""
    ### 📚 Methodology & Formulas
    
    **Primary Calculations:**
    - **BED (Radiopharmaceutical):** D × (1 + G × D/(α/β))
    - **G-factor:** λ_eff/(λ_eff + μ_repair) where λ_eff = ln(2)/T_eff, μ_repair = ln(2)/T_repair
    - **EQD2:** BED / (1 + 2/(α/β))
    
    **Advanced Metrics:**
    - **EQD2₉₉:** EQD2 when 99% of dose has been delivered (temporal concept)
    - **Time for 99% delivery:** t₉₉ = -ln(0.01) × T_eff / ln(2) ≈ 6.64 × T_eff
    - **EQD2Max:** EQD2 from exponential delivery over initial 24 hours
    - **EQD2R:** EQD2_tolerance - EQD2_previous (remaining dose capacity)
    
    **Kidney-Specific BED Limits:**
    - **High Risk Patients:** 28 Gy BED (existing kidney disease/risk factors)
    - **Low Risk Patients:** 40 Gy BED (no existing kidney disease)
    - **Treatment Planning:** Uses BED-based approach for cumulative dose tracking
    
    **G-factor Formula:**
    - **Formula:** G = λ_eff/(λ_eff + μ_repair)
    - **Where:** λ_eff = ln(2)/T_eff (effective decay constant), μ_repair = ln(2)/T_repair (repair constant)
    - **Physical meaning:** Fraction of damage that becomes permanent due to limited repair during dose delivery
    - **Range:** 0 < G < 1 (G approaches 1 for very fast delivery, approaches 0 for very slow delivery)
    
    **EQD2Max Concept:**
    - **Sustained dose rate:** Uses initial dose rate as if continued forever
    - **BED calculation:** BED = D × (1 + (2R)/(α/β × μ)) for sustained rate R
    - **EQD2Max:** BED of total dose delivered at constant initial rate
    - **Purpose:** Compare exponential decay vs sustained constant rate delivery
    
    **EQD2₉₉ Concept:**
    - **Initial dose rate:** dose_rate_0 = D_total × λ_eff (Gy/h)
    - **Maximum 24h dose:** D_max_24h = dose_rate_0 × 24 (Gy)
    - **EQD2Max:** BED of D_max_24h delivered acutely (G = 1)
    - **Purpose:** Compare protracted vs acute delivery biological effects
    
    **Clinical Applications:**
    - Treatment planning optimization with BED-based limits
    - Dose rate effect quantification
    - Cumulative dose tracking across multiple treatments
    - Kidney risk-stratified dose planning
    - Temporal effect analysis
    - Risk-benefit analysis
    
    **⚠️ Important Notes:**
    - This calculator is for research and educational purposes
    - Clinical decisions require qualified medical physics consultation
    - Organ-specific parameters may vary between patients
    - Consider individual patient factors and clinical context
    - EQD2₉₉ is a kinetic endpoint, EQD2Max is a delivery timing comparison
    - **EQD2Max** = EQD2 from exponential delivery over initial 24 hours
    - **Dose rate effects** (EQD2Max)
    - **Treatment optimization** (Extended delivery benefit)
    """)
    
    # Data export functionality
    st.subheader("📥 Data Export")
    
    if st.button("Export Current Session Data"):
        if 'primary_results' in st.session_state:
            results = st.session_state.primary_results
            
            # Calculate additional metrics for export
            eqd299, time_99, dose_99 = calculate_eqd299(results['organ_dose'], results['alpha_beta'], 
                                                       results['effective_half_life'], results['repair_half_time'])
            eqd2max, initial_dose_rate, total_dose_used = calculate_eqd2max(results['organ_dose'], results['alpha_beta'], 
                                                                           results['effective_half_life'], results['repair_half_time'])
            
            # Get organ tolerance
            organ_tolerance_limits = {
                "Bone Marrow": 2.0, "Liver": 30.0, "Lungs": 20.0,
                "Heart": 26.0, "Spinal Cord": 50.0, "Salivary Glands": 26.0, "Thyroid": 45.0,
                "Lacrimal Glands": 30.0, "Bladder": 65.0, "Prostate": 76.0, "Breast": 50.0
            }
            
            if selected_organ == "Kidneys":
                # Use BED-based limits for kidneys (converted to EQD2 for export)
                organ_tolerance_bed = 28.0  # Default to high risk
                organ_tolerance = calculate_eqd2(organ_tolerance_bed * (1 + 2/alpha_beta), alpha_beta)
            else:
                organ_tolerance = organ_tolerance_limits.get(selected_organ, 25.0)
            
            export_data = {
                'Parameter': [
                    'Organ', 'Alpha/Beta Ratio (Gy)', 'Repair Half-time (h)',
                    'Organ Dose (Gy)', 'Effective Half-life (h)', 'BED (Gy)',
                    'EQD2 (Gy)', 'EQD2₉₉ (Gy)', 'Time to 99% (h)', 'EQD2Max (Gy)',
                    'Initial Dose Rate (Gy/h)', 'Total Dose Used (Gy)', 'Organ Tolerance (Gy)',
                    'G-factor', 'Dose Rate Factor', 'Tolerance Ratio', 'Dose Rate Benefit (Gy)'
                ],
                'Value': [
                    results['organ'], results['alpha_beta'], results['repair_half_time'],
                    results['organ_dose'], results['effective_half_life'], results['bed'],
                    results['eqd2'], eqd299, time_99, eqd2max,
                    initial_dose_rate, total_dose_used, organ_tolerance,
                    results['g_factor'], results['drf'], results['eqd2']/organ_tolerance,
                    eqd2max - results['eqd2']
                ]
            }
            
            export_df = pd.DataFrame(export_data)
            
            # Convert to CSV
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"radiopharm_dosimetry_{selected_organ.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
            st.success("Data prepared for export!")
        else:
            st.warning("No calculation results to export. Please perform calculations first.")

if __name__ == "__main__":
    main()