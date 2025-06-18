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
            "Lacrimal Glands": 30.0, "Bladder": 65.0, "Prostate": 76.0, "Breast": 50.0
        }
        eqd2_limit = organ_eqd2_limits.get(selected_organ, 25.0)
        # Convert EQD2 limit to BED limit: BED = EQD2 × (1 + 2/(α/β))
        return eqd2_limit * (1 + 2 / alpha_beta)

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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧮 Primary Calculation", 
        "📊 Advanced Assessment", 
        "🔄 Treatment Planning", 
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
            
            st.subheader("Planned Treatment")
            planned_dose = st.number_input("Planned dose (Gy):", min_value=0.0, value=10.0, step=0.1)
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
                    bed_limit = get_organ_bed_tolerance(selected_organ, alpha_beta)
                
                # Calculate remaining BED capacity
                remaining_bed = bed_limit - total_previous_bed
                
                st.session_state.treatment_results = {
                    'previous_treatments': previous_treatments,
                    'previous_bed': total_previous_bed,
                    'planned_bed': planned_bed,
                    'planned_dose': planned_dose,
                    'planned_half_life': planned_half_life,
                    'total_bed': total_bed,
                    'remaining_bed': remaining_bed,
                    'bed_limit': bed_limit,
                    'bed_ratio': total_bed / bed_limit,
                    'organ': selected_organ,
                    'num_treatments': num_previous + 1
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
                    max_additional_bed = results['remaining_bed']
                    st.info(f"""
                    **Treatment Capacity:**
                    - Remaining BED: {max_additional_bed:.2f} Gy
                    - Current plan: {results['planned_bed']:.2f} Gy
                    - Utilization: {(results['planned_bed']/max_additional_bed)*100:.1f}% of remaining
                    """)
                    
                    if results['planned_bed'] > max_additional_bed:
                        st.error(f"⚠️ Planned dose exceeds capacity by {results['planned_bed'] - max_additional_bed:.2f} Gy BED")
                else:
                    st.error("❌ No remaining BED capacity")
                
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
            if has_primary and has_treatment_plan:
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
            
            # Cumulative Treatment Safety Assessment
            if safety_tab2 is not None:
                with safety_tab2:
                    if has_treatment_plan:
                        st.subheader("Cumulative Treatment Safety Analysis")
                        treat_results = st.session_state.treatment_results
                        
                        # Enhanced cumulative safety dashboard
                        col1, col2, col3, col4 = st.columns(4)
                        
                        cumulative_tolerance_ratio = treat_results['total_bed'] / organ_bed_tolerance
                        
                        with col1:
                            st.metric("Total Treatments", f"{treat_results['num_treatments']}")
                            
                        with col2:
                            st.metric("Cumulative BED", f"{treat_results['total_bed']:.1f} Gy")
                            
                        with col3:
                            st.metric("BED Limit", f"{organ_bed_tolerance:.1f} Gy")
                            
                        with col4:
                            if cumulative_tolerance_ratio <= 0.8:
                                st.metric("Cumulative Risk", "✅ LOW", f"{cumulative_tolerance_ratio:.2f}")
                            elif cumulative_tolerance_ratio <= 1.0:
                                st.metric("Cumulative Risk", "⚠️ MODERATE", f"{cumulative_tolerance_ratio:.2f}")
                            else:
                                st.metric("Cumulative Risk", "❌ HIGH", f"{cumulative_tolerance_ratio:.2f}")
                        
                        # Detailed treatment breakdown
                        st.subheader("Treatment History Summary")
                        
                        if treat_results['previous_treatments']:
                            treatment_data = []
                            cumulative_bed = 0
                            
                            for i, treatment in enumerate(treat_results['previous_treatments']):
                                cumulative_bed += treatment['bed']
                                treatment_data.append({
                                    'Treatment #': i + 1,
                                    'Dose (Gy)': f"{treatment['dose']:.2f}",
                                    'Half-life (h)': f"{treatment['half_life']:.1f}",
                                    'BED (Gy)': f"{treatment['bed']:.2f}",
                                    'Cumulative BED (Gy)': f"{cumulative_bed:.2f}",
                                    'Tolerance %': f"{(cumulative_bed/organ_bed_tolerance)*100:.1f}%"
                                })
                            
                            # Add planned treatment
                            if treat_results['planned_bed'] > 0:
                                cumulative_bed += treat_results['planned_bed']
                                treatment_data.append({
                                    'Treatment #': f"{len(treat_results['previous_treatments']) + 1} (Planned)",
                                    'Dose (Gy)': f"{treat_results['planned_dose']:.2f}",
                                    'Half-life (h)': f"{treat_results['planned_half_life']:.1f}",
                                    'BED (Gy)': f"{treat_results['planned_bed']:.2f}",
                                    'Cumulative BED (Gy)': f"{cumulative_bed:.2f}",
                                    'Tolerance %': f"{(cumulative_bed/organ_bed_tolerance)*100:.1f}%"
                                })
                            
                            df_treatments = pd.DataFrame(treatment_data)
                            st.dataframe(df_treatments, use_container_width=True)
                        
                        # Cumulative BED progression visualization
                        st.subheader("Cumulative BED Progression")
                        
                        fig = go.Figure()
                        
                        # Calculate cumulative progression
                        treatments = treat_results['previous_treatments'].copy()
                        if treat_results['planned_bed'] > 0:
                            treatments.append({
                                'dose': treat_results['planned_dose'],
                                'half_life': treat_results['planned_half_life'],
                                'bed': treat_results['planned_bed']
                            })
                        
                        cumulative_beds = []
                        treatment_numbers = []
                        colors = []
                        
                        cumulative = 0
                        for i, treatment in enumerate(treatments):
                            cumulative += treatment['bed']
                            cumulative_beds.append(cumulative)
                            treatment_numbers.append(f"Treatment {i+1}")
                            
                            # Color coding based on tolerance level
                            ratio = cumulative / organ_bed_tolerance
                            if ratio <= 0.8:
                                colors.append('green')
                            elif ratio <= 1.0:
                                colors.append('orange')
                            else:
                                colors.append('red')
                        
                        # Add bars showing cumulative BED
                        fig.add_trace(go.Bar(
                            x=treatment_numbers,
                            y=cumulative_beds,
                            marker_color=colors,
                            text=[f"{bed:.1f} Gy" for bed in cumulative_beds],
                            textposition='auto'
                        ))
                        
                        # Add tolerance limit line
                        fig.add_hline(y=organ_bed_tolerance, line_dash="dash", line_color="red", 
                                    annotation_text=f"BED Limit: {organ_bed_tolerance:.1f} Gy")
                        
                        # Add comfort zones
                        fig.add_hrect(y0=0, y1=organ_bed_tolerance*0.8, 
                                    fillcolor="green", opacity=0.1, annotation_text="Safe Zone")
                        fig.add_hrect(y0=organ_bed_tolerance*0.8, y1=organ_bed_tolerance, 
                                    fillcolor="orange", opacity=0.1, annotation_text="Caution Zone")
                        fig.add_hrect(y0=organ_bed_tolerance, y1=max(max(cumulative_beds), organ_bed_tolerance*1.2), 
                                    fillcolor="red", opacity=0.1, annotation_text="Risk Zone")
                        
                        fig.update_layout(
                            title=f"Cumulative BED Safety Assessment for {selected_organ}",
                            yaxis_title="Cumulative BED (Gy)",
                            xaxis_title="Treatment Sequence",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Comprehensive cumulative safety summary
                        remaining_capacity = organ_bed_tolerance - treat_results['total_bed']
                        equivalent_total_fractions = calculate_equivalent_fractions(treat_results['total_bed'], alpha_beta)
                        
                        st.info(f"""
                        **Cumulative Safety Summary for {selected_organ}:**
                        
                        **Treatment History:**
                        - Total treatments: {treat_results['num_treatments']}
                        - Cumulative BED: {treat_results['total_bed']:.1f} Gy
                        - Equivalent total fractions: {equivalent_total_fractions:.1f} × 2 Gy
                        
                        **Risk Assessment:**
                        - Tolerance utilization: {(treat_results['total_bed']/organ_bed_tolerance)*100:.1f}%
                        - Remaining capacity: {remaining_capacity:.1f} Gy BED
                        - Risk level: {"LOW" if cumulative_tolerance_ratio <= 0.8 else "MODERATE" if cumulative_tolerance_ratio <= 1.0 else "HIGH"}
                        
                        **Future Treatment Capacity:**
                        - {"Additional treatments possible" if remaining_capacity > 0 else "No remaining capacity"}
                        {f"- Maximum additional BED: {remaining_capacity:.1f} Gy" if remaining_capacity > 0 else ""}
                        """)
                    else:
                        st.info("Cumulative treatment results not available. Calculate treatment planning first.")
            
        else:
            st.info("Please calculate primary dosimetry and/or treatment planning first to see safety assessment.")
    
    # Footer with methodology
    st.markdown("---")
    st.markdown("""
    ### 📚 Methodology & Clinical Application
    
    **Primary Calculations:**
    - **BED (Radiopharmaceutical):** D × (1 + G × D/(α/β))
    - **G-factor:** λ_eff/(λ_eff + μ_repair) where λ_eff = ln(2)/T_eff, μ_repair = ln(2)/T_repair
    - **EQD2:** BED / (1 + 2/(α/β))
    - **Equivalent Fractions:** BED / [2 × (1 + 2/(α/β))]
    
    **Advanced Metrics:**
    - **EQD2₉₉:** EQD2 when 99% of dose has been delivered (temporal milestone)
    - **Delivery Efficiency:** Fraction of dose delivered at key timepoints
    - **BED Tolerance Limits:** Organ-specific safety thresholds
    - **Cumulative BED:** Sum of BED from multiple treatments
    
    **Kidney-Specific BED Limits:**
    - **High Risk Patients:** 28 Gy BED (existing kidney disease/risk factors)
    - **Low Risk Patients:** 40 Gy BED (no existing kidney disease)
    
    **⚠️ Important Notes:**
    - This calculator is for research and educational purposes
    - Clinical decisions require qualified medical physics consultation
    - Consider individual patient factors and clinical context
    - Cumulative BED tracking essential for multiple treatment safety
    """)

if __name__ == "__main__":
    main()