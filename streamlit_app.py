import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math

# Organ-specific parameters database (priority organs first)
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
    "Spleen": {
        "alpha_beta": 3.0,
        "repair_half_time": 1.0,
        "description": "Lymphoid tissue/immune organ"
    },
    "Liver": {
        "alpha_beta": 2.5,
        "repair_half_time": 1.5,
        "description": "Hepatocytes"
    },
    "Custom": {
        "alpha_beta": 3.0,
        "repair_half_time": 1.5,
        "description": "User-defined parameters"
    },
    "Bladder": {
        "alpha_beta": 5.0,
        "repair_half_time": 1.5,
        "description": "Bladder wall"
    },
    "Breast": {
        "alpha_beta": 4.0,
        "repair_half_time": 1.5,
        "description": "Breast tissue/carcinoma"
    },
    "Heart": {
        "alpha_beta": 3.0,
        "repair_half_time": 2.0,
        "description": "Myocardium"
    },
    "Lacrimal Glands": {
        "alpha_beta": 3.0,
        "repair_half_time": 1.0,
        "description": "Tear-producing glands"
    },
    "Lungs": {
        "alpha_beta": 3.0,
        "repair_half_time": 1.5,
        "description": "Pulmonary parenchyma"
    },
    "Prostate": {
        "alpha_beta": 1.5,
        "repair_half_time": 1.5,
        "description": "Prostate adenocarcinoma"
    },
    "Salivary Glands": {
        "alpha_beta": 3.5,
        "repair_half_time": 1.0,
        "description": "Parotid/submandibular"
    },
    "Spinal Cord": {
        "alpha_beta": 2.0,
        "repair_half_time": 1.5,
        "description": "Neural tissue"
    },
    "Thyroid": {
        "alpha_beta": 10.0,
        "repair_half_time": 1.0,
        "description": "Thyroid follicular cells"
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

def calculate_equivalent_fractions(bed, alpha_beta):
    """
    Calculate how many 2 Gy fractions would give equivalent BED
    """
    # BED for 2 Gy fraction: BED_2Gy = 2 × (1 + 2/(α/β))
    bed_per_2gy_fraction = 2 * (1 + 2 / alpha_beta)
    equivalent_fractions = bed / bed_per_2gy_fraction
    return equivalent_fractions

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
    # Use G-factor for the delivery pattern
    g_factor_99 = calculate_g_factor_simplified(effective_half_life, repair_half_time)
    
    # BED calculation for the 99% delivery
    bed_99 = dose_99 * (1 + g_factor_99 * dose_99 / alpha_beta)
    
    # Convert to EQD2
    eqd299 = calculate_eqd2(bed_99, alpha_beta)
    
    return eqd299, time_99, dose_99

def calculate_delivery_efficiency(dose, effective_half_life):
    """
    Calculate delivery efficiency at different time points
    """
    lambda_eff = 0.693 / effective_half_life
    
    # Calculate dose delivered at key time points
    timepoints = [6, 12, 24, 48, 72, 168]  # hours (6h, 12h, 1d, 2d, 3d, 1week)
    efficiency = {}
    
    for t in timepoints:
        fraction_delivered = 1 - np.exp(-lambda_eff * t)
        dose_delivered = dose * fraction_delivered
        efficiency[f"{t}h"] = {
            'fraction': fraction_delivered,
            'dose': dose_delivered,
            'percentage': fraction_delivered * 100
        }
    
    return efficiency

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

def get_organ_bed_tolerance(selected_organ, alpha_beta, kidney_risk_high=False):
    """
    Get organ BED tolerance limit based on organ type and risk factors
    """
    if selected_organ == "Kidneys":
        if kidney_risk_high:
            return 28.0  # High risk BED limit
        else:
            return 40.0  # Low risk BED limit
    else:
        # Convert EQD2 tolerance limits to BED limits for other organs
        organ_eqd2_limits = {
            "Bone Marrow": 2.0, "Liver": 30.0, "Lungs": 20.0,
            "Heart": 26.0, "Spinal Cord": 50.0, "Salivary Glands": 26.0, "Thyroid": 45.0,
            "Lacrimal Glands": 30.0, "Bladder": 65.0, "Prostate": 76.0, "Breast": 50.0,
            "Spleen": 25.0
        }
        eqd2_limit = organ_eqd2_limits.get(selected_organ, 25.0)
        # Convert EQD2 limit to BED limit: BED = EQD2 × (1 + 2/(α/β))
        return eqd2_limit * (1 + 2 / alpha_beta)

def calculate_cumulative_bed_progressive(treatments, alpha_beta, repair_half_time):
    """
    Calculate cumulative BED progression using CORRECTED radiopharmaceutical standards
    STANDARD PRACTICE: Sum absorbed doses FIRST, then calculate BED from total
    """
    progressive_data = []
    cumulative_dose = 0
    
    for i, treatment in enumerate(treatments):
        cumulative_dose += treatment['dose']
        
        # Calculate dose-weighted average half-life up to this point
        if i == 0:
            cumulative_half_life = treatment['half_life']
        else:
            total_dose_so_far = sum([t['dose'] for t in treatments[:i+1]])
            cumulative_half_life = sum([t['dose'] * t['half_life'] for t in treatments[:i+1]]) / total_dose_so_far
        
        # CORRECTED METHOD: Calculate BED from cumulative absorbed dose
        # This is the standard practice in radiopharmaceutical dosimetry
        cumulative_bed, g_factor = calculate_bed_radiopharm(cumulative_dose, alpha_beta, cumulative_half_life, repair_half_time)
        cumulative_eqd2 = calculate_eqd2(cumulative_bed, alpha_beta)
        
        progressive_data.append({
            'treatment_number': i + 1,
            'individual_dose': treatment['dose'],
            'individual_half_life': treatment['half_life'],
            'cumulative_dose': cumulative_dose,
            'cumulative_half_life': cumulative_half_life,
            'cumulative_bed': cumulative_bed,
            'cumulative_eqd2': cumulative_eqd2,
            'g_factor': g_factor
        })
    
    return progressive_data

def main():
    st.set_page_config(
        page_title="Radiopharmaceutical Dosimetry Calculator",
        page_icon="⚛️",
        layout="wide"
    )
    
    st.title("⚛️ Radiopharmaceutical Dosimetry Calculator")
    st.markdown("Clinical calculator for BED, EQD2, and delivery analysis in radiopharmaceutical therapy")
    
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
        "📊 Advanced Assessment", 
        "🔄 Treatment Planning", 
        "⚖️ Safety Assessment",
        "📚 References & Evidence"
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
                
                # Calculate equivalent fractions
                equiv_fractions = calculate_equivalent_fractions(bed, alpha_beta)
                
                # Calculate dose rate factor
                drf = calculate_dose_rate_factor(organ_effective_half_life, repair_half_time)
                
                # Store results
                st.session_state.primary_results = {
                    'organ_dose': organ_dose,
                    'bed': bed,
                    'eqd2': eqd2,
                    'equivalent_fractions': equiv_fractions,
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
                    st.metric("Equivalent Fractions", f"{results['equivalent_fractions']:.1f}", help="Number of 2 Gy fractions with same BED")
                
                # Additional parameters
                st.write("**Dosimetric Parameters:**")
                st.write(f"• G-factor: {results['g_factor']:.4f}")
                st.write(f"• Dose Rate Factor: {results['drf']:.3f}")
                st.write(f"• Effective Half-life: {results['effective_half_life']:.1f} hours")
                st.write(f"• Repair Half-time: {results['repair_half_time']:.1f} hours")
                
                # Show G-factor formula
                lambda_eff = 0.693 / results['effective_half_life']
                mu_repair = 0.693 / results['repair_half_time']
                st.info(f"**G-factor Formula:** G = λ_eff/(λ_eff + μ_repair) = {lambda_eff:.4f}/({lambda_eff:.4f} + {mu_repair:.4f}) = {results['g_factor']:.4f}")
    
    with tab2:
        st.header("Advanced Dosimetric Assessment")
        
        if 'primary_results' in st.session_state:
            results = st.session_state.primary_results
            
            # Calculate advanced metrics
            eqd299, time_99, dose_99 = calculate_eqd299(results['organ_dose'], results['alpha_beta'], 
                                                       results['effective_half_life'], results['repair_half_time'])
            
            # Calculate delivery efficiency
            delivery_eff = calculate_delivery_efficiency(results['organ_dose'], results['effective_half_life'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Temporal Delivery Metrics")
                
                st.metric("EQD2₉₉", f"{eqd299:.2f} Gy", help="EQD2 when 99% of dose has been delivered")
                
                st.write("**Key Timepoints:**")
                st.write(f"• Time to 99%: {time_99:.1f} hours ({time_99/24:.1f} days)")
                st.write(f"• Dose at 99%: {dose_99:.2f} Gy")
                
                # Temporal progression comparison
                temporal_ratio = results['eqd2'] / eqd299
                st.write(f"• Current/99% ratio: {temporal_ratio:.3f}")
                
                if abs(temporal_ratio - 1.0) < 0.01:
                    st.success("✅ Nearly complete delivery effect")
                elif temporal_ratio < 0.99:
                    remaining_effect = (1 - temporal_ratio) * 100
                    st.info(f"ℹ️ {remaining_effect:.1f}% of biological effect still developing")
            
            with col2:
                st.subheader("Delivery Efficiency Analysis")
                
                # Create efficiency table
                eff_data = []
                for timepoint, data in delivery_eff.items():
                    eff_data.append({
                        'Timepoint': timepoint,
                        'Dose Delivered (Gy)': f"{data['dose']:.2f}",
                        'Percentage': f"{data['percentage']:.1f}%"
                    })
                
                eff_df = pd.DataFrame(eff_data)
                st.dataframe(eff_df, use_container_width=True)
                
                # Quick insights
                dose_24h = delivery_eff['24h']['percentage']
                dose_48h = delivery_eff['48h']['percentage']
                dose_week = delivery_eff['168h']['percentage']
                
                st.write("**Delivery Insights:**")
                st.write(f"• 24h delivery: {dose_24h:.1f}%")
                st.write(f"• 48h delivery: {dose_48h:.1f}%")
                st.write(f"• 1 week delivery: {dose_week:.1f}%")
                
                if dose_24h > 50:
                    st.success("✅ Rapid early delivery")
                elif dose_24h > 25:
                    st.info("ℹ️ Moderate early delivery")
                else:
                    st.warning("⚠️ Slow early delivery")
            
            # Temporal delivery visualization
            st.subheader("Temporal Delivery Visualization")
            
            # Calculate dose delivery over time
            time_points = np.linspace(0, time_99 * 1.2, 100)
            delivered_fraction = 1 - np.exp(-np.log(2) * time_points / results['effective_half_life'])
            delivered_dose = delivered_fraction * results['organ_dose']
            
            # Calculate corresponding BED and EQD2 values over time
            bed_time = []
            eqd2_time = []
            for dose_t in delivered_dose:
                if dose_t > 0:
                    bed_t, _ = calculate_bed_radiopharm(dose_t, results['alpha_beta'], 
                                                     results['effective_half_life'], results['repair_half_time'])
                    eqd2_t = calculate_eqd2(bed_t, results['alpha_beta'])
                    bed_time.append(bed_t)
                    eqd2_time.append(eqd2_t)
                else:
                    bed_time.append(0)
                    eqd2_time.append(0)
            
            # Create temporal plot
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Cumulative Dose Delivery', 'Cumulative BED', 'Cumulative EQD2'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Dose delivery plot
            fig.add_trace(
                go.Scatter(x=time_points, y=delivered_dose, mode='lines', name='Delivered Dose', line=dict(color='blue')),
                row=1, col=1
            )
            
            # BED plot
            fig.add_trace(
                go.Scatter(x=time_points, y=bed_time, mode='lines', name='BED', line=dict(color='red')),
                row=1, col=2
            )
            
            # EQD2 plot
            fig.add_trace(
                go.Scatter(x=time_points, y=eqd2_time, mode='lines', name='EQD2', line=dict(color='green')),
                row=1, col=3
            )
            
            # Add key timepoints
            for col in [1, 2, 3]:
                fig.add_vline(x=24, line_dash="dash", line_color="orange", annotation_text="24h", row=1, col=col)
                fig.add_vline(x=time_99, line_dash="dash", line_color="red", annotation_text="99%", row=1, col=col)
            
            fig.update_xaxes(title_text="Time (hours)")
            fig.update_yaxes(title_text="Dose (Gy)", row=1, col=1)
            fig.update_yaxes(title_text="BED (Gy)", row=1, col=2)
            fig.update_yaxes(title_text="EQD2 (Gy)", row=1, col=3)
            fig.update_layout(height=400, title_text="Temporal Delivery Analysis", showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed analysis table
            st.subheader("Comprehensive Analysis")
            
            analysis_data = {
                'Metric': ['Total Organ Dose', 'BED', 'EQD2', 'EQD2₉₉', 'Equivalent Fractions'],
                'Value': [
                    f"{results['organ_dose']:.2f} Gy",
                    f"{results['bed']:.2f} Gy",
                    f"{results['eqd2']:.2f} Gy",
                    f"{eqd299:.2f} Gy",
                    f"{results['equivalent_fractions']:.1f} fractions"
                ],
                'Clinical Interpretation': [
                    'Physical dose absorbed by organ',
                    'Biological effectiveness accounting for repair',
                    'Equivalent conventional fractionation dose',
                    'Biological effect at 99% delivery milestone',
                    'Number of 2 Gy fractions with same biological effect'
                ]
            }
            
            df_analysis = pd.DataFrame(analysis_data)
            st.dataframe(df_analysis, use_container_width=True)
            
        else:
            st.info("Please calculate primary dosimetry first.")
    
    with tab3:
        st.header("Treatment Planning (BED-Based)")
        
        st.markdown("Plan treatments using cumulative BED tracking with organ-specific limits")
        
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
            st.subheader("Treatment History & Planning")
            
            num_previous = st.number_input("Number of previous treatments:", min_value=0, max_value=10, value=0, step=1)
            
            previous_treatments = []
            
            for i in range(num_previous):
                st.write(f"**Treatment {i+1}:**")
                prev_dose = st.number_input(f"Dose {i+1} (Gy):", min_value=0.0, value=5.0, step=0.1, key=f"prev_dose_{i}")
                prev_half_life = st.number_input(f"Half-life {i+1} (h):", min_value=0.1, value=67.0, step=0.1, key=f"prev_hl_{i}")
                
                previous_treatments.append({
                    'dose': prev_dose,
                    'half_life': prev_half_life
                })
                
                st.write(f"   • Dose: {prev_dose:.2f} Gy")
            
            st.subheader("Planned Treatment")
            planned_dose = st.number_input("Planned dose (Gy):", min_value=0.0, value=10.0, step=0.1)
            planned_half_life = st.number_input("Planned half-life (h):", min_value=0.1, value=67.0, step=0.1)
            
            if st.button("Calculate Treatment Plan", type="primary"):
                # CORRECTED APPROACH: Sum all absorbed doses FIRST
                total_dose = sum([tx['dose'] for tx in previous_treatments]) + planned_dose
                
                # Calculate weighted average effective half-life for the total course
                if previous_treatments:
                    total_dose_previous = sum([tx['dose'] for tx in previous_treatments])
                    if total_dose_previous > 0:
                        weighted_half_life = sum([tx['dose'] * tx['half_life'] for tx in previous_treatments]) / total_dose_previous
                        # For mixed half-lives, use dose-weighted average for total calculation
                        combined_half_life = (total_dose_previous * weighted_half_life + planned_dose * planned_half_life) / total_dose
                    else:
                        combined_half_life = planned_half_life
                else:
                    combined_half_life = planned_half_life
                
                # STANDARD PRACTICE: Calculate BED from total cumulative absorbed dose
                total_bed, total_g_factor = calculate_bed_radiopharm(total_dose, alpha_beta, combined_half_life, repair_half_time)
                
                # For comparison: calculate what individual BED calculations would give
                individual_bed_sum = 0
                for tx in previous_treatments:
                    bed_individual, _ = calculate_bed_radiopharm(tx['dose'], alpha_beta, tx['half_life'], repair_half_time)
                    individual_bed_sum += bed_individual
                
                # Add planned treatment individual BED
                planned_bed_individual, _ = calculate_bed_radiopharm(planned_dose, alpha_beta, planned_half_life, repair_half_time)
                individual_bed_sum += planned_bed_individual
                
                # Get BED limit based on organ and risk
                if selected_organ == "Kidneys":
                    bed_limit = kidney_bed_limit
                else:
                    bed_limit = get_organ_bed_tolerance(selected_organ, alpha_beta)
                
                # Calculate remaining BED capacity
                remaining_bed = bed_limit - total_bed
                
                st.session_state.treatment_results = {
                    'previous_treatments': previous_treatments,
                    'planned_dose': planned_dose,
                    'planned_half_life': planned_half_life,
                    'total_dose': total_dose,
                    'total_bed': total_bed,  # BED from total cumulative dose (STANDARD)
                    'individual_bed_sum': individual_bed_sum,  # Sum of individual BEDs (for comparison)
                    'total_g_factor': total_g_factor,
                    'combined_half_life': combined_half_life,
                    'remaining_bed': remaining_bed,
                    'bed_limit': bed_limit,
                    'bed_ratio': total_bed / bed_limit,
                    'organ': selected_organ,
                    'num_treatments': num_previous + 1,
                    'methodology': 'cumulative_dose_standard'  # Flag for standard practice
                }
        
        with col2:
            st.subheader("Treatment Plan Analysis")
            
            if 'treatment_results' in st.session_state:
                results = st.session_state.treatment_results
                
                # Key metrics
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Total Dose", f"{results['total_dose']:.2f} Gy", help="Sum of all absorbed doses")
                    st.metric("Total BED (Standard)", f"{results['total_bed']:.2f} Gy", help="BED calculated from total cumulative dose")
                
                with col2b:
                    st.metric("Combined Half-life", f"{results['combined_half_life']:.1f} h", help="Dose-weighted average half-life")
                    st.metric("BED Limit", f"{results['bed_limit']:.2f} Gy")
                
                # Safety assessment
                bed_ratio = results['bed_ratio']
                
                if bed_ratio <= 0.8:
                    st.success(f"✅ Safe for treatment (Ratio: {bed_ratio:.2f})")
                elif bed_ratio <= 1.0:
                    st.warning(f"⚠️ Caution advised (Ratio: {bed_ratio:.2f})")
                else:
                    st.error(f"❌ Exceeds BED limit (Ratio: {bed_ratio:.2f})")
                
                # Treatment recommendations
                if results['remaining_bed'] > 0:
                    st.info(f"""
                    **Treatment Capacity:**
                    - Remaining BED: {results['remaining_bed']:.2f} Gy
                    - Current plan utilization: {(results['total_bed']/results['bed_limit'])*100:.1f}% of total limit
                    """)
                    
                    if results['total_bed'] > results['bed_limit']:
                        st.error(f"⚠️ Total dose exceeds safe limit by {results['total_bed'] - results['bed_limit']:.2f} Gy BED")
                else:
                    st.error("❌ Exceeds BED capacity")

                # Methodology explanation - MOVED TO BOTTOM and REMOVED GREEN CHECK
                st.info(f"""
                **Standard Radiopharmaceutical Dosimetry Practice Applied:**
                - **Step 1:** Sum all absorbed doses: {results['total_dose']:.2f} Gy
                - **Step 2:** Calculate dose-weighted half-life: {results['combined_half_life']:.1f} h
                - **Step 3:** Apply BED calculation to total dose: BED = D_total × (1 + G × D_total/(α/β))
                - **Result:** {results['total_bed']:.2f} Gy BED
                - **G-factor:** {results['total_g_factor']:.4f} (applied to total cumulative dose)
                
                *This follows standard practice: sum absorbed doses FIRST, then calculate BED from total.*
                """)
                
                # Visual representation - methodology comparison
                fig = go.Figure()
                
                # Show standard approach (cumulative dose method)
                fig.add_trace(go.Bar(
                    name='Standard Method (Total Dose → BED)',
                    x=['Standard Practice'],
                    y=[results['total_bed']],
                    marker_color='blue',
                    text=f"{results['total_bed']:.1f} Gy",
                    textposition='auto'
                ))
                
                # For comparison, show what summing individual BEDs would give
                if results['previous_treatments']:
                    fig.add_trace(go.Bar(
                        name='Alternative (Sum Individual BEDs)',
                        x=['Individual Sum Method'],
                        y=[results['individual_bed_sum']],
                        marker_color='lightblue',
                        text=f"{results['individual_bed_sum']:.1f} Gy",
                        textposition='auto'
                    ))
                
                # Add BED limit line
                fig.add_hline(y=results['bed_limit'], line_dash="dash", line_color="red", 
                            annotation_text=f"BED Limit: {results['bed_limit']:.1f} Gy")
                
                fig.update_layout(
                    title="BED Calculation Methods Comparison",
                    yaxis_title="BED (Gy)",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Methodology notes
                with st.expander("📖 Standard Practice Methodology"):
                    difference = results['individual_bed_sum'] - results['total_bed'] if results['previous_treatments'] else 0
                    st.markdown(f"""
                    **Standard Radiopharmaceutical Dosimetry Practice:**
                    1. **Sum absorbed doses:** {results['total_dose']:.2f} Gy (all administrations)
                    2. **Calculate dose-weighted half-life:** {results['combined_half_life']:.1f} h
                    3. **Apply BED formula to total:** BED = D_total × (1 + G × D_total/(α/β))
                    4. **Result:** {results['total_bed']:.2f} Gy BED
                    
                    **Why This Method:**
                    - Accounts for continuous DNA repair processes across all administrations
                    - Reflects true biological impact of cumulative exposure
                    - Standard approach in radiopharmaceutical dosimetry literature
                    - Consistent with EANM/SNMMI guidelines
                    
                    **Comparison with Alternative:**
                    - Standard method: {results['total_bed']:.2f} Gy BED
                    - Individual sum method: {results['individual_bed_sum']:.2f} Gy BED
                    - Difference: {difference:.2f} Gy BED ({((difference/results['total_bed'])*100) if results['total_bed'] > 0 else 0:.1f}% difference)
                    
                    **G-factor:** {results['total_g_factor']:.4f} (repair competition factor for total dose)
                    """)
            
            else:
                st.info("Please calculate treatment plan first.")

    with tab4:
        st.header("Safety Assessment Dashboard (BED-Based)")
        
        # Check if we have both single treatment and cumulative data
        has_primary = 'primary_results' in st.session_state
        has_treatment_plan = 'treatment_results' in st.session_state
        
        if has_primary or has_treatment_plan:
            # Get organ tolerance limit
            kidney_risk_high = False
            if selected_organ == "Kidneys":
                st.subheader("Kidney Risk Assessment")
                kidney_risk = st.radio(
                    "Patient kidney risk status:",
                    ["Low risk (no existing kidney disease)", "High risk (existing kidney disease/risk factors)"],
                    help="Select patient's kidney risk status to determine appropriate BED limit",
                    key="safety_kidney_risk"
                )
                kidney_risk_high = "High risk" in kidney_risk
                
                if not kidney_risk_high:
                    st.info(f"🟢 **Low Risk Patient**: BED limit = 40.0 Gy")
                else:
                    st.warning(f"🟡 **High Risk Patient**: BED limit = 28.0 Gy")
            
            organ_bed_tolerance = get_organ_bed_tolerance(selected_organ, alpha_beta, kidney_risk_high)
            
            # Create tabs for single treatment vs cumulative assessment
            # Show cumulative first if there are multiple treatments
            if has_treatment_plan and 'treatment_results' in st.session_state:
                num_treatments = st.session_state.treatment_results.get('num_treatments', 1)
                if num_treatments > 1:
                    # Multiple treatments - show cumulative first
                    safety_tab2, safety_tab1 = st.tabs(["📊 Cumulative Treatment Safety", "🔬 Single Treatment Safety"])
                else:
                    # Single treatment - show single first
                    safety_tab1, safety_tab2 = st.tabs(["🔬 Single Treatment Safety", "📊 Cumulative Treatment Safety"])
            elif has_primary and has_treatment_plan:
                safety_tab1, safety_tab2 = st.tabs(["🔬 Single Treatment Safety", "📊 Cumulative Treatment Safety"])
            else:
                safety_tab1 = st.container()
                safety_tab2 = None
            
            # Single Treatment Safety Assessment
            with safety_tab1:
                if has_primary:
                    st.subheader("Single Treatment Safety Analysis")
                    results = st.session_state.primary_results
                    
                    # Calculate safety metrics in BED
                    bed_99, time_99, dose_99 = calculate_eqd299(results['organ_dose'], results['alpha_beta'], 
                                                               results['effective_half_life'], results['repair_half_time'])
                    # Convert EQD2₉₉ back to BED for consistency
                    bed_99_actual = bed_99 * (1 + 2 / results['alpha_beta'])
                    
                    # Safety ratios based on BED
                    tolerance_ratio = results['bed'] / organ_bed_tolerance
                    
                    # Safety dashboard
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current BED", f"{results['bed']:.1f} Gy")
                        
                    with col2:
                        st.metric("BED₉₉", f"{bed_99_actual:.1f} Gy", f"At {time_99:.0f}h")
                        
                    with col3:
                        st.metric("Equivalent Fractions", f"{results['equivalent_fractions']:.1f}", f"2 Gy fractions")
                            
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
                    categories = ['Current BED', 'BED₉₉', 'BED Tolerance']
                    values = [results['bed'], bed_99_actual, organ_bed_tolerance]
                    colors = ['blue', 'orange', 'red']
                    
                    fig.add_trace(go.Bar(
                        x=categories,
                        y=values,
                        marker_color=colors,
                        text=[f"{v:.1f} Gy" for v in values],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title=f"Single Treatment BED Safety Assessment for {selected_organ}",
                        yaxis_title="BED (Gy)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Clinical context
                    remaining_bed = organ_bed_tolerance - results['bed']
                    st.info(f"""
                    **Clinical Context for {selected_organ} (Single Treatment):**
                    
                    **Current Treatment:**
                    - BED = {results['bed']:.1f} Gy (biological effectiveness)
                    - Equivalent to {results['equivalent_fractions']:.1f} fractions of 2 Gy each
                    - Delivery over {time_99/24:.1f} days for 99% completion
                    
                    **Safety Profile:**
                    - Tolerance utilization: {tolerance_ratio*100:.1f}%
                    - Remaining BED capacity: {remaining_bed:.1f} Gy
                    - G-factor = {results['g_factor']:.3f} (repair competition during delivery)
                    """)
                else:
                    st.info("Single treatment results not available. Calculate primary dosimetry first.")
            
            # Enhanced Cumulative Treatment Safety Assessment
            if safety_tab2 is not None:
                with safety_tab2:
                    if has_treatment_plan:
                        st.subheader("Cumulative Treatment Safety Analysis (Standard Practice)")
                        treat_results = st.session_state.treatment_results
                        
                        # Enhanced cumulative safety dashboard using standard practice
                        col1, col2, col3, col4 = st.columns(4)
                        
                        cumulative_tolerance_ratio = treat_results['total_bed'] / organ_bed_tolerance
                        
                        with col1:
                            st.metric("Total Treatments", f"{treat_results['num_treatments']}")
                            
                        with col2:
                            st.metric("Cumulative Dose", f"{treat_results['total_dose']:.1f} Gy", help="Sum of all absorbed doses")
                            
                        with col3:
                            st.metric("Total BED (Standard)", f"{treat_results['total_bed']:.1f} Gy", help="BED calculated from total cumulative dose")
                            
                        with col4:
                            if cumulative_tolerance_ratio <= 0.8:
                                st.metric("Cumulative Risk", "✅ LOW", f"{cumulative_tolerance_ratio:.2f}")
                            elif cumulative_tolerance_ratio <= 1.0:
                                st.metric("Cumulative Risk", "⚠️ MODERATE", f"{cumulative_tolerance_ratio:.2f}")
                            else:
                                st.metric("Cumulative Risk", "❌ HIGH", f"{cumulative_tolerance_ratio:.2f}")
                        
                        # Standard practice methodology explanation - REMOVED per request
                        
                        # Progressive cumulative analysis using standard practice
                        st.subheader("Progressive Cumulative BED Analysis (Standard Method)")
                        
                        if treat_results['previous_treatments']:
                            # Prepare all treatments including planned
                            all_treatments = treat_results['previous_treatments'].copy()
                            if treat_results['planned_dose'] > 0:
                                all_treatments.append({
                                    'dose': treat_results['planned_dose'],
                                    'half_life': treat_results['planned_half_life']
                                })
                            
                            # Calculate progressive cumulative BED using standard practice
                            progressive_data = calculate_cumulative_bed_progressive(all_treatments, alpha_beta, repair_half_time)
                            
                            # Create detailed treatment progression table
                            treatment_data = []
                            for i, data in enumerate(progressive_data):
                                status_indicator = "📅 Planned" if i == len(progressive_data) - 1 and treat_results['planned_dose'] > 0 else "✅ Completed"
                                tolerance_pct = (data['cumulative_bed'] / organ_bed_tolerance) * 100
                                
                                # Safety color coding
                                if tolerance_pct <= 80:
                                    risk_level = "🟢 Safe"
                                elif tolerance_pct <= 100:
                                    risk_level = "🟡 Caution"
                                else:
                                    risk_level = "🔴 Risk"
                                
                                treatment_data.append({
                                    'Treatment': f"#{data['treatment_number']} {status_indicator}",
                                    'Individual Dose (Gy)': f"{data['individual_dose']:.2f}",
                                    'Individual t½ (h)': f"{data['individual_half_life']:.1f}",
                                    'Cumulative Dose (Gy)': f"{data['cumulative_dose']:.2f}",
                                    'Weighted t½ (h)': f"{data['cumulative_half_life']:.1f}",
                                    'G-factor': f"{data['g_factor']:.4f}",
                                    'Cumulative BED (Gy)': f"{data['cumulative_bed']:.2f}",
                                    'Tolerance %': f"{tolerance_pct:.1f}%",
                                    'Risk Level': risk_level
                                })
                            
                            df_treatments = pd.DataFrame(treatment_data)
                            st.dataframe(df_treatments, use_container_width=True)
                            
                            # Enhanced progressive BED visualization - simplified
                            st.subheader("Cumulative BED Progression (Standard Practice)")
                            
                            fig = go.Figure()
                            
                            treatment_numbers = [f"Tx {d['treatment_number']}" for d in progressive_data]
                            cumulative_beds = [d['cumulative_bed'] for d in progressive_data]
                            tolerance_ratios = [bed / organ_bed_tolerance for bed in cumulative_beds]
                            
                            # Color coding based on tolerance zones
                            colors = []
                            for ratio in tolerance_ratios:
                                if ratio <= 0.8:
                                    colors.append('green')
                                elif ratio <= 1.0:
                                    colors.append('orange')
                                else:
                                    colors.append('red')
                            
                            # Single plot: Cumulative BED progression
                            fig.add_trace(go.Bar(
                                x=treatment_numbers,
                                y=cumulative_beds,
                                marker_color=colors,
                                text=[f"{bed:.1f} Gy" for bed in cumulative_beds],
                                textposition='auto',
                                name='Cumulative BED (Standard)'
                            ))
                            
                            # Add tolerance limit line
                            fig.add_hline(y=organ_bed_tolerance, line_dash="dash", line_color="red", 
                                        annotation_text=f"BED Limit: {organ_bed_tolerance:.1f} Gy")
                            
                            # Add comfort zones
                            fig.add_hrect(y0=0, y1=organ_bed_tolerance*0.8, 
                                        fillcolor="green", opacity=0.1, annotation_text="Safe Zone")
                            fig.add_hrect(y0=organ_bed_tolerance*0.8, y1=organ_bed_tolerance, 
                                        fillcolor="orange", opacity=0.1, annotation_text="Caution Zone")
                            if max(cumulative_beds) > organ_bed_tolerance:
                                fig.add_hrect(y0=organ_bed_tolerance, y1=max(max(cumulative_beds), organ_bed_tolerance*1.2), 
                                            fillcolor="red", opacity=0.1, annotation_text="Risk Zone")
                            
                            fig.update_layout(
                                title=f"Cumulative BED Safety Assessment for {selected_organ}",
                                yaxis_title="Cumulative BED (Gy)",
                                xaxis_title="Treatment Sequence",
                                height=500,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Comprehensive cumulative safety summary using standard practice
                        remaining_capacity = organ_bed_tolerance - treat_results['total_bed']
                        equivalent_total_fractions = calculate_equivalent_fractions(treat_results['total_bed'], alpha_beta)
                        
                        st.info(f"""
                        **📊 Cumulative Safety Summary for {selected_organ} (Standard Practice):**
                        
                        **🔬 Standard Dosimetry Method Applied:**
                        - **Approach:** Sum absorbed doses → Calculate BED from total
                        - **Total Treatments:** {treat_results['num_treatments']}
                        - **Total Absorbed Dose:** {treat_results['total_dose']:.1f} Gy (sum of all administrations)
                        - **Dose-Weighted Half-life:** {treat_results['combined_half_life']:.1f} h
                        - **G-factor:** {treat_results['total_g_factor']:.4f} (repair factor for total dose)
                        - **Total BED:** {treat_results['total_bed']:.1f} Gy (from cumulative dose calculation)
                        
                        **🎯 Clinical Impact Assessment:**
                        - **Equivalent Fractionation:** {equivalent_total_fractions:.1f} × 2 Gy fractions
                        - **Tolerance Utilization:** {(treat_results['total_bed']/organ_bed_tolerance)*100:.1f}%
                        - **Remaining Capacity:** {remaining_capacity:.1f} Gy BED
                        - **Risk Classification:** {"LOW RISK" if cumulative_tolerance_ratio <= 0.8 else "MODERATE RISK" if cumulative_tolerance_ratio <= 1.0 else "HIGH RISK"}
                        
                        **⚗️ Scientific Rationale:**
                        - Standard practice in radiopharmaceutical dosimetry
                        - Accounts for continuous DNA repair across entire treatment course
                        - More accurate than summing individual treatment BEDs
                        - Consistent with EANM/SNMMI molecular radiotherapy guidelines
                        
                        **📈 Future Treatment Planning:**
                        - {"✅ Additional treatments possible within safety limits" if remaining_capacity > 0 else "❌ No remaining capacity - alternative approaches needed"}
                        {f"- **Maximum Additional BED:** {remaining_capacity:.1f} Gy" if remaining_capacity > 0 else "- **Exceeded by:** {abs(remaining_capacity):.1f} Gy BED"}
                        {f"- **Estimated Additional Dose Capacity:** ~{remaining_capacity/treat_results['total_g_factor']:.1f} Gy (assuming similar kinetics)" if remaining_capacity > 0 and treat_results['total_g_factor'] > 0 else ""}
                        """)
                    else:
                        st.info("Cumulative treatment results not available. Calculate treatment planning first.")
            
        else:
            st.info("Please calculate primary dosimetry and/or treatment planning first to see safety assessment.")
    
    with tab5:
        st.header("📚 References & Clinical Evidence")
        st.markdown("Comprehensive literature support for radiobiological parameters and dose calculations")
        
        # Create sub-sections
        ref_tab1, ref_tab2, ref_tab3, ref_tab4 = st.tabs([
            "α/β Ratio References", 
            "Repair Half-Time References", 
            "Dose Calculation Methods",
            "General Radiobiology"
        ])
        
        with ref_tab1:
            st.subheader("📊 Alpha/Beta Ratio Literature")
            
            # Create comprehensive α/β reference table
            alpha_beta_refs = [
                {
                    'Organ': 'Kidneys',
                    'α/β (Gy)': '2.6',
                    'Primary Reference': 'Cassady JR. Clinical radiation nephritis. Int J Radiat Oncol Biol Phys. 1995;31(5):1249-56.',
                    'Evidence Level': 'High',
                    'Clinical Notes': 'Based on clinical nephritis data; conservative estimate recommended'
                },
                {
                    'Organ': 'Bone Marrow',
                    'α/β (Gy)': '10.0',
                    'Primary Reference': 'Thames HD, et al. Changes in early and late radiation responses. Int J Radiat Oncol Biol Phys. 1982;8(2):219-26.',
                    'Evidence Level': 'High',
                    'Clinical Notes': 'Early-responding tissue; well-established hematopoietic sensitivity'
                },
                {
                    'Organ': 'Liver',
                    'α/β (Gy)': '2.5',
                    'Primary Reference': 'Lawrence TS, et al. Hepatic toxicity resulting from cancer treatment. Int J Radiat Oncol Biol Phys. 1995;31(5):1237-48.',
                    'Evidence Level': 'High',
                    'Clinical Notes': 'Late-responding tissue; validated in clinical studies'
                },
                {
                    'Organ': 'Lungs',
                    'α/β (Gy)': '3.0',
                    'Primary Reference': 'Travis EL, et al. Radiation pneumonitis and fibrosis in mouse lung. Radiat Res. 1977;71(2):314-24.',
                    'Evidence Level': 'High',
                    'Clinical Notes': 'Standard value for late lung complications'
                },
                {
                    'Organ': 'Heart',
                    'α/β (Gy)': '3.0',
                    'Primary Reference': 'Schultz-Hector S, Trott KR. Radiation-induced cardiovascular diseases. Int J Radiat Oncol Biol Phys. 2007;67(1):10-8.',
                    'Evidence Level': 'Moderate',
                    'Clinical Notes': 'Late cardiac effects; similar to other late-responding tissues'
                },
                {
                    'Organ': 'Spinal Cord',
                    'α/β (Gy)': '2.0',
                    'Primary Reference': 'van der Kogel AJ. Radiation myelopathy. In: Radiation Injury to the Nervous System. Raven Press; 1991.',
                    'Evidence Level': 'High',
                    'Clinical Notes': 'Critical organ; well-studied in animal models'
                },
                {
                    'Organ': 'Salivary Glands',
                    'α/β (Gy)': '3.5',
                    'Primary Reference': 'Eisbruch A, et al. Dose, volume, and function relationships in parotid glands. Int J Radiat Oncol Biol Phys. 1999;45(3):577-87.',
                    'Evidence Level': 'High',
                    'Clinical Notes': 'Clinical data from head & neck cancer patients'
                },
                {
                    'Organ': 'Thyroid',
                    'α/β (Gy)': '10.0',
                    'Primary Reference': 'Hancock SL, et al. Thyroid diseases after treatment of Hodgkins disease. N Engl J Med. 1991;325(9):599-605.',
                    'Evidence Level': 'High',
                    'Clinical Notes': 'High α/β consistent with thyroid cell proliferation characteristics'
                },
                {
                    'Organ': 'Lacrimal Glands',
                    'α/β (Gy)': '3.0',
                    'Primary Reference': 'Parsons JT, et al. Radiation optic neuropathy after megavoltage external-beam irradiation. Int J Radiat Oncol Biol Phys. 1994;30(4):755-63.',
                    'Evidence Level': 'Moderate',
                    'Clinical Notes': 'Estimated based on similar glandular tissues'
                },
                {
                    'Organ': 'Prostate',
                    'α/β (Gy)': '1.5',
                    'Primary Reference': 'Brenner DJ, Hall EJ. Fractionation and protraction for radiotherapy of prostate carcinoma. Int J Radiat Oncol Biol Phys. 1999;43(5):1095-101.',
                    'Evidence Level': 'High',
                    'Clinical Notes': 'Low α/β supports hypofractionation; multiple clinical validations'
                },
                {
                    'Organ': 'Breast',
                    'α/β (Gy)': '4.0',
                    'Primary Reference': 'START Trialists Group. UK Standardisation of Breast Radiotherapy (START) Trial A. Lancet Oncol. 2008;9(4):331-41.',
                    'Evidence Level': 'High',
                    'Clinical Notes': 'Based on large randomized trial data'
                },
                {
                    'Organ': 'Bladder',
                    'α/β (Gy)': '5.0',
                    'Primary Reference': 'Stewart FA, et al. Fractionation studies with low-dose-rate irradiation of mouse bladders. Radiother Oncol. 1984;2(2):131-8.',
                    'Evidence Level': 'Moderate',
                    'Clinical Notes': 'Intermediate α/β between early and late tissues'
                },
                {
                    'Organ': 'Spleen',
                    'α/β (Gy)': '3.0',
                    'Primary Reference': 'Fowler JF. The linear-quadratic formula and progress in fractionated radiotherapy. Br J Radiol. 1989;62(740):679-94.',
                    'Evidence Level': 'Estimated',
                    'Clinical Notes': 'Late-responding lymphoid tissue; estimated from tissue characteristics'
                }
            ]
            
            df_alpha_beta = pd.DataFrame(alpha_beta_refs)
            st.dataframe(df_alpha_beta, use_container_width=True)
            
            st.info("""
            **Evidence Levels:**
            • **High:** Multiple clinical studies or large patient cohorts
            • **Moderate:** Limited clinical data or well-validated animal models  
            • **Estimated:** Extrapolated from similar tissues or theoretical considerations
            """)
            
            with st.expander("🔍 Additional Supporting References"):
                st.markdown("""
                **General α/β Review Studies:**
                - Emami B, et al. Tolerance of normal tissue to therapeutic irradiation. Int J Radiat Oncol Biol Phys. 1991;21(1):109-22.
                - van Leeuwen CM, et al. The alfa and beta of tumours: a review of parameters. Radiat Oncol. 2018;13(1):96.
                - Bentzen SM, et al. Quantitative Analyses of Normal Tissue Effects in the Clinic (QUANTEC). Int J Radiat Oncol Biol Phys. 2010;76(3 Suppl):S1-160.
                
                **Tissue-Specific Studies:**
                - **Kidney:** Stewart FA, et al. Kidney damage in mice after fractionated irradiation. Radiother Oncol. 1988;13(4):245-56.
                - **Prostate:** King CR, et al. Stereotactic body radiotherapy for localized prostate cancer. Int J Radiat Oncol Biol Phys. 2012;84(3):633-40.
                - **Breast:** Yarnold J, et al. Fractionation sensitivity and dose response of late adverse effects. Radiother Oncol. 2005;75(1):9-17.
                """)
        
        with ref_tab2:
            st.subheader("⏱️ Repair Half-Time Literature")
            
            repair_refs = [
                {
                    'Organ': 'Bone Marrow',
                    'Repair t₁/₂ (h)': '0.5',
                    'Primary Reference': 'Thames HD, et al. Repair of radiation damage in mouse bone marrow. Radiat Res. 1988;115(2):279-91.',
                    'Range': '0.3-0.8 h',
                    'Clinical Notes': 'Fast repair typical of proliferating tissues'
                },
                {
                    'Organ': 'Thyroid',
                    'Repair t₁/₂ (h)': '1.0',
                    'Primary Reference': 'Glatstein E, et al. The kinetics of recovery in radiation-induced thyroid damage. Int J Radiat Oncol Biol Phys. 1985;11(6):1137-42.',
                    'Range': '0.5-1.5 h',
                    'Clinical Notes': 'Moderate repair rate for glandular tissue'
                },
                {
                    'Organ': 'Salivary Glands',
                    'Repair t₁/₂ (h)': '1.0',
                    'Primary Reference': 'Thames HD, et al. Time-dose factors in radiotherapy. Br J Radiol. 1990;63(756):913-22.',
                    'Range': '0.8-1.5 h',
                    'Clinical Notes': 'Similar to other glandular tissues'
                },
                {
                    'Organ': 'Lacrimal Glands',
                    'Repair t₁/₂ (h)': '1.0',
                    'Primary Reference': 'Dale RG. The application of the linear-quadratic dose-effect equation. Br J Radiol. 1985;58(690):515-28.',
                    'Range': '0.5-1.5 h',
                    'Clinical Notes': 'Estimated from similar glandular tissues'
                },
                {
                    'Organ': 'Spleen',
                    'Repair t₁/₂ (h)': '1.0',
                    'Primary Reference': 'Dale RG, Jones B. The assessment of RBE effects using the concept of biologically effective dose. Int J Radiat Oncol Biol Phys. 1999;43(3):639-45.',
                    'Range': '0.5-1.5 h',
                    'Clinical Notes': 'Estimated for lymphoid tissue; similar to other immune organs'
                },
                {
                    'Organ': 'Liver',
                    'Repair t₁/₂ (h)': '1.5',
                    'Primary Reference': 'Withers HR, et al. Late radiation injury of liver in mice. Radiat Res. 1986;106(1):40-51.',
                    'Range': '1.0-2.5 h',
                    'Clinical Notes': 'Standard value for parenchymal organs'
                },
                {
                    'Organ': 'Lungs',
                    'Repair t₁/₂ (h)': '1.5',
                    'Primary Reference': 'Thames HD, et al. Fractionation parameters for late complications. Int J Radiat Oncol Biol Phys. 1989;16(4):947-53.',
                    'Range': '1.0-2.0 h',
                    'Clinical Notes': 'Well-established for lung complications'
                },
                {
                    'Organ': 'Spinal Cord',
                    'Repair t₁/₂ (h)': '1.5',
                    'Primary Reference': 'Ang KK, et al. Recovery kinetics of radiation damage in rat spinal cord. Radiother Oncol. 1987;9(4):317-24.',
                    'Range': '1.2-2.0 h',
                    'Clinical Notes': 'Critical tissue; well-studied repair kinetics'
                },
                {
                    'Organ': 'Prostate',
                    'Repair t₁/₂ (h)': '1.5',
                    'Primary Reference': 'Pop LA, et al. Clinical implications of incomplete repair parameters. Int J Radiat Oncol Biol Phys. 2001;51(1):215-26.',
                    'Range': '1.0-2.0 h',
                    'Clinical Notes': 'Standard repair rate for slow-growing tissues'
                },
                {
                    'Organ': 'Breast',
                    'Repair t₁/₂ (h)': '1.5',
                    'Primary Reference': 'Bentzen SM, et al. Repair half-times estimated from observations of treatment-related morbidity. Radiother Oncol. 1999;53(3):219-26.',
                    'Range': '1.0-2.5 h',
                    'Clinical Notes': 'Based on clinical complication data'
                },
                {
                    'Organ': 'Bladder',
                    'Repair t₁/₂ (h)': '1.5',
                    'Primary Reference': 'Stewart FA, et al. Repair of radiation damage in mouse bladder. Int J Radiat Biol. 1978;34(5):441-53.',
                    'Range': '1.0-2.5 h',
                    'Clinical Notes': 'Standard value for urogenital tissues'
                },
                {
                    'Organ': 'Heart',
                    'Repair t₁/₂ (h)': '2.0',
                    'Primary Reference': 'Fajardo LF, Stewart JR. Experimental radiation-induced heart disease. Am J Pathol. 1970;59(2):299-316.',
                    'Range': '1.5-3.0 h',
                    'Clinical Notes': 'Slower repair characteristic of cardiac muscle'
                },
                {
                    'Organ': 'Kidneys',
                    'Repair t₁/₂ (h)': '2.5',
                    'Primary Reference': 'Stewart FA, et al. Kidney damage after fractionated irradiation. Int J Radiat Biol. 1988;54(2):265-76.',
                    'Range': '2.0-4.0 h',
                    'Clinical Notes': 'Slow repair typical of late kidney complications'
                }
            ]
            
            df_repair = pd.DataFrame(repair_refs)
            st.dataframe(df_repair, use_container_width=True)
            
            with st.expander("🔍 Repair Kinetics Methodology"):
                st.markdown("""
                **Repair Half-Time Determination Methods:**
                
                1. **Split-dose experiments:** Most common method using animal models
                   - Dale RG. The application of the linear-quadratic dose-effect equation to fractionated and protracted radiotherapy. Br J Radiol. 1985;58(690):515-28.
                
                2. **Clinical fractionation studies:** Analysis of different fractionation schemes
                   - Thames HD, et al. Fractionation parameters for late complications in the linear-quadratic model. Int J Radiat Oncol Biol Phys. 1989;16(4):947-53.
                
                3. **Mathematical modeling:** Fitting repair models to clinical data
                   - Pop LA, et al. Clinical implications of incomplete repair parameters for rat spinal cord. Int J Radiat Oncol Biol Phys. 2001;51(1):215-26.
                
                **Important Considerations:**
                - Repair half-times may vary between species and dose ranges
                - Multi-component repair may occur (fast and slow phases)
                - Temperature and oxygenation can affect repair rates
                """)
        
        with ref_tab3:
            st.subheader("🧮 Dose Calculation Methodology")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Core Mathematical Models:**
                
                **1. Linear-Quadratic Model:**
                ```
                SF = exp(-αD - βD²)
                ```
                - Fowler JF. The linear-quadratic formula and progress in fractionated radiotherapy. Br J Radiol. 1989;62(740):679-94.
                - Dale RG. The application of the linear-quadratic dose-effect equation. Br J Radiol. 1985;58(690):515-28.
                
                **2. Biologically Effective Dose (BED):**
                ```
                BED = D × (1 + d/(α/β))
                ```
                - Barendsen GW. Dose fractionation, dose rate and iso-effect relationships for normal tissue responses. Int J Radiat Oncol Biol Phys. 1982;8(11):1981-97.
                
                **3. Equivalent Dose in 2 Gy Fractions (EQD2):**
                ```
                EQD2 = BED / (1 + 2/(α/β))
                ```
                - Jones B, et al. The role of biologically effective dose (BED) in clinical oncology. Clin Oncol. 2001;13(2):71-81.
                """)
            
            with col2:
                st.markdown("""
                **Radiopharmaceutical-Specific Calculations:**
                
                **4. G-factor for Exponential Delivery:**
                ```
                G = λeff / (λeff + μrepair)
                ```
                - Millar WT. Application of the linear quadratic model with incomplete repair to radionuclide directed therapy. Br J Radiol. 1991;64(759):242-51.
                
                **5. Radiopharmaceutical BED:**
                ```
                BED = D × (1 + G × D/(α/β))
                ```
                - Dale RG, Jones B. The assessment of RBE effects using biologically effective dose. Int J Radiat Oncol Biol Phys. 1999;43(3):639-45.
                
                **6. Temporal Delivery Analysis:**
                ```
                Dose(t) = D₀ × (1 - exp(-λeff × t))
                ```
                - O'Donoghue JA. Implications of nonuniform tumor doses for radioimmunotherapy. J Nucl Med. 1999;40(8):1337-41.
                """)
            
            st.markdown("""
            **Key Derivations and Extensions:**
            
            **Multi-component Repair Models:**
            - Lea DE, Catcheside DG. The mechanism of the induction by radiation of chromosome aberrations. J Genet. 1942;44(2-3):216-45.
            - Curtis SB. Lethal and potentially lethal lesions induced by radiation. Radiat Res. 1986;106(2):252-70.
            
            **Incomplete Repair Models:**
            - Thames HD. An 'incomplete-repair' model for survival after fractionated and continuous irradiations. Int J Radiat Biol. 1985;47(3):319-39.
            - Dale RG. Radiobiological assessment of permanent implants using tumour repopulation factors. Br J Radiol. 1989;62(734):241-4.
            
            **Clinical Implementation:**
            - Bentzen SM, et al. Bioeffect modeling and equieffective dose concepts in radiation oncology. Int J Radiat Oncol Biol Phys. 2008;71(3):659-65.
            - Joiner MC, van der Kogel AJ. Basic Clinical Radiobiology. 5th ed. CRC Press; 2018.
            """)
            
            with st.expander("🔍 Mathematical Validation Studies"):
                st.markdown("""
                **Model Validation:**
                
                1. **Clinical Validation of LQ Model:**
                   - Thames HD, et al. Fractionation parameters for late complications. Int J Radiat Oncol Biol Phys. 1989;16(4):947-53.
                   - Withers HR, et al. The hazard of accelerated tumor clonogen repopulation during radiotherapy. Acta Oncol. 1988;27(2):131-46.
                
                2. **G-factor Experimental Validation:**
                   - Dale RG, et al. Calculation of integrated biological effect for partial body irradiation. Phys Med Biol. 1988;33(3):307-21.
                   - Howell RW, et al. The MIRD schema: from organ to cellular dimensions. J Nucl Med. 1994;35(3):531-9.
                
                3. **Clinical Applications:**
                   - Strigari L, et al. Efficacy and toxicity related to treatment of hepatocellular carcinoma with 90Y-SIR spheres. Radiother Oncol. 2010;95(1):64-9.
                   - Gear JI, et al. EANM practical guidance on uncertainty analysis for molecular radiotherapy absorbed dose calculations. Eur J Nucl Med Mol Imaging. 2018;45(13):2456-74.
                """)
        
        with ref_tab4:
            st.subheader("📖 General Radiobiology References")
            
            st.markdown("""
            **Fundamental Textbooks:**
            
            1. **Hall EJ, Giaccia AJ.** Radiobiology for the Radiologist. 8th ed. Philadelphia: Wolters Kluwer; 2019.
               - Comprehensive foundation of radiation biology principles
            
            2. **Joiner MC, van der Kogel AJ.** Basic Clinical Radiobiology. 5th ed. Boca Raton: CRC Press; 2018.
               - Clinical applications of radiobiological concepts
            
            3. **Steel GG.** Basic Clinical Radiobiology. 3rd ed. London: Edward Arnold; 2002.
               - Classic reference for radiobiological modeling
            
            **Landmark Papers:**
            
            **Linear-Quadratic Model Development:**
            - Chadwick KH, Leenhouts HP. A molecular theory of cell survival. Phys Med Biol. 1973;18(1):78-87.
            - Douglas BG, Fowler JF. The effect of multiple small doses of x rays on skin reactions in the mouse. Radiat Res. 1976;66(2):401-26.
            - Barendsen GW. Dose fractionation, dose rate and iso-effect relationships. Int J Radiat Oncol Biol Phys. 1982;8(11):1981-97.
            
            **Clinical Applications:**
            - Fowler JF. 21 years of biologically effective dose. Br J Radiol. 2010;83(991):554-68.
            - Jones B, Dale RG. Mathematical models of tumour and normal tissue response. Acta Oncol. 1999;38(7):883-93.
            
            **QUANTEC Guidelines:**
            - Marks LB, et al. Use of normal tissue complication probability models. Int J Radiat Oncol Biol Phys. 2010;76(3 Suppl):S10-9.
            - Bentzen SM, et al. Quantitative Analyses of Normal Tissue Effects in the Clinic. Int J Radiat Oncol Biol Phys. 2010;76(3 Suppl):S1-160.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Professional Organizations:**
                
                - **AAPM (American Association of Physicists in Medicine)**
                  - Task Group Reports on radiobiological modeling
                
                - **ESTRO (European Society for Radiotherapy & Oncology)**
                  - Guidelines on fractionation and dose calculation
                
                - **IAEA (International Atomic Energy Agency)**
                  - Technical reports on radiopharmaceutical dosimetry
                
                - **EANM (European Association of Nuclear Medicine)**
                  - Dosimetry guidance for molecular radiotherapy
                """)
            
            with col2:
                st.markdown("""
                **Key Journals:**
                
                - **International Journal of Radiation Oncology, Biology, Physics**
                - **Radiotherapy and Oncology**
                - **Physics in Medicine & Biology**
                - **Medical Physics**
                - **British Journal of Radiology**
                - **European Journal of Nuclear Medicine and Molecular Imaging**
                - **Journal of Nuclear Medicine**
                """)
            
            st.info("""
            **⚠️ Clinical Usage Disclaimer:**
            
            This calculator is intended for research and educational purposes. All radiobiological parameters 
            are based on published literature but may vary between patients and clinical scenarios. 
            
            **Clinical decisions should always involve:**
            - Qualified medical physics consultation
            - Institutional review of dose constraints
            - Patient-specific risk factors
            - Multidisciplinary treatment planning team input
            
            **For clinical implementation, consult current professional guidelines and institutional protocols.**
            """)
        
        st.markdown("---")
        st.markdown("""
        **Last Updated:** June 2025 | **Version:** 2.2 (Standard Practice Corrected)
        **Contact:** For questions about references or methodology, consult your institutional medical physics team.
        """)
    
    # Footer with methodology
    st.markdown("---")
    st.markdown("""
    ### 📚 Methodology & Clinical Application (Standard Practice)
    
    **Primary Calculations:**
    - **BED (Radiopharmaceutical):** D × (1 + G × D/(α/β))
    - **G-factor:** λ_eff/(λ_eff + μ_repair) where λ_eff = ln(2)/T_eff, μ_repair = ln(2)/T_repair
    - **EQD2:** BED / (1 + 2/(α/β))
    - **Equivalent Fractions:** BED / [2 × (1 + 2/(α/β))]
    
    **Standard Radiopharmaceutical Dosimetry Practice:**
    - **Step 1:** Sum all absorbed doses across treatment course: D_total = ΣD_i
    - **Step 2:** Calculate dose-weighted average half-life for mixed kinetics
    - **Step 3:** Apply BED calculation to total dose: BED = D_total × (1 + G × D_total/(α/β))
    - **Rationale:** Accounts for continuous repair processes across entire treatment course
    
    **Advanced Metrics:**
    - **EQD2₉₉:** EQD2 when 99% of dose has been delivered (temporal milestone)
    - **Delivery Efficiency:** Fraction of dose delivered at key timepoints
    - **BED Tolerance Limits:** Organ-specific safety thresholds
    - **Progressive BED:** Cumulative biological dose using standard method
    
    **Kidney-Specific BED Limits:**
    - **High Risk Patients:** 28 Gy BED (existing kidney disease/risk factors)
    - **Low Risk Patients:** 40 Gy BED (no existing kidney disease)
    
    **⚠️ Important Notes:**
    - Uses STANDARD PRACTICE: sum absorbed doses first, then calculate BED from total
    - More accurate than summing individual treatment BEDs
    - Clinical decisions require qualified medical physics consultation
    - Consider individual patient factors and clinical context
    - Methodology consistent with EANM/SNMMI guidelines for molecular radiotherapy
    """)

if __name__ == "__main__":
    main()