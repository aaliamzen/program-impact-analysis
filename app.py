import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Streamlit app configuration
st.set_page_config(page_title="Teaching Competency Analysis", layout="wide")

# Streamlit app configuration
st.set_page_config(page_title="Teaching Competency Analysis", layout="wide")

# ====================== NEW: EXAMPLE FILE DOWNLOAD ======================
st.markdown("### Example Data File")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    **Download Example Excel File**  
    This file contains sample data in the exact format required by the tool (multiple domains, competencies, and terms).
    """)

with col2:
    try:
        with open("ImpactData_example.xlsx", "rb") as file:
            st.download_button(
                label="⬇️ Download ImpactData_example.xlsx",
                data=file,
                file_name="ImpactData_example.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Sample data to test the tool"
            )
    except FileNotFoundError:
        st.error("Example file not found. Please place it in the `static/` folder.")

st.markdown("---")
# =====================================================================

# Custom CSS to override Streamlit's default table styles (no longer needed, but keeping for consistency)
st.markdown("""
    <style>
    /* Override Streamlit's default table styles to remove grey gridlines */
    .stMarkdown table, .stMarkdown th, .stMarkdown td {
        border: none !important;
    }
    .stMarkdown table tr, .stMarkdown table td {
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Default Likert labels
default_likert_labels = {
    '1': 'Not developed',
    '2': 'Developing towards',
    '3': 'Developed',
    '4': 'Well developed'
}

# Sidebar for file upload and custom Likert labels
st.sidebar.title("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])

st.sidebar.title("Custom Likert Labels")
label1 = st.sidebar.text_input("Label for Rating 1", value=default_likert_labels['1'])
label2 = st.sidebar.text_input("Label for Rating 2", value=default_likert_labels['2'])
label3 = st.sidebar.text_input("Label for Rating 3", value=default_likert_labels['3'])
label4 = st.sidebar.text_input("Label for Rating 4", value=default_likert_labels['4'])

# Dynamically create likert_labels based on user input
likert_labels = {
    '1': label1 if label1 else default_likert_labels['1'],
    '2': label2 if label2 else default_likert_labels['2'],
    '3': label3 if label3 else default_likert_labels['3'],
    '4': label4 if label4 else default_likert_labels['4']
}

# Domain-specific color palettes (tied to rating values 1-4)
color_maps = {
    0: {  # Domain 1 (e.g., Instructional Skills) - Cyan palette
        1: '#E6F5F6',
        2: '#C8EAED',
        3: '#A6E0E4',
        4: '#83D5DA'
    },
    1: {  # Domain 2 (e.g., Management Skills) - Blue palette
        1: '#D8DEEF',
        2: '#A8B5DB',
        3: '#728AC3',
        4: '#334F9E'
    },
    2: {  # Domain 3 (e.g., Domain 3) - Green palette
        1: '#E8F5C8',
        2: '#D6F0A0',
        3: '#C2EA75',
        4: '#B8E34D'
    },
    3: {  # Domain 4 (e.g., Domain 4) - Purple palette
        1: '#ECE9F8',
        2: '#DAD3F1',
        3: '#C8BCE9',
        4: '#B2A5E4'
    }
}

# Function to wrap text at approximately n characters, preserving spaces
def wrap_text(text, width=30):
    if len(text) <= width:
        return text
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    for word in words:
        word_length = len(word)
        if current_length + word_length + (1 if current_line else 0) <= width:
            current_line.append(word)
            current_length += word_length + (1 if current_line else 0)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length
    if current_line:
        lines.append(" ".join(current_line))
    return "<br>".join(lines)

# Process uploaded file and store in session state
if uploaded_file:
    # Read Excel file
    df = pd.read_excel(uploaded_file)
    
    # Validate required columns
    required_cols = ['Comptnum', 'Competency', 'Domain', 'Term', 'Count_1', 'Count_2', 'Count_3', 'Count_4']
    if not all(col in df.columns for col in required_cols):
        st.error("Excel file must contain columns: Comptnum, Competency, Domain, Term, Count_1, Count_2, Count_3, Count_4")
    else:
        # Data preprocessing
        # Identify competencies with no counts (all Count_1 to Count_4 are NaN for all terms)
        competency_counts = df.groupby('Competency')[['Count_1', 'Count_2', 'Count_3', 'Count_4']].agg(lambda x: x.notna().any())
        valid_competencies = competency_counts[competency_counts.any(axis=1)].index.tolist()
        excluded_competencies = [comp for comp in df['Competency'].unique() if comp not in valid_competencies]
        
        # Filter data to include only valid competencies
        df = df[df['Competency'].isin(valid_competencies)]
        competencies = df[['Comptnum', 'Competency']].drop_duplicates().sort_values('Comptnum')
        domains = df['Domain'].unique()
        
        # Dynamically extract valid terms from the data
        # First, check for non-numeric values in the Term column
        term_values = df['Term'].dropna()
        try:
            # Convert to float first to handle both integers and floats, then to int
            term_values_numeric = term_values.astype(float)
            # Check if all values are whole numbers (i.e., integers)
            if not (term_values_numeric == term_values_numeric.astype(int)).all():
                st.error("Term column must contain only integer values (no decimals)")
            else:
                valid_terms = sorted(term_values_numeric.astype(int).unique())
                # Check if all terms are positive
                if not all(term > 0 for term in valid_terms):
                    st.error("Term column must contain only positive integers")
                elif df.empty:
                    st.error("No valid competencies with counts found in the dataset.")
                else:
                    # Store processed data in session state
                    st.session_state['df'] = df
                    st.session_state['competencies'] = competencies
                    st.session_state['domains'] = domains
                    st.session_state['excluded_competencies'] = excluded_competencies
                    st.session_state['valid_terms'] = valid_terms
        except (ValueError, TypeError) as e:
            st.error(f"Term column contains non-numeric values or cannot be converted to integers: {str(e)}")

# Check if data is available in session state
if 'df' not in st.session_state:
    st.info("Please upload an Excel file to begin analysis.")
else:
    # Retrieve data from session state
    df = st.session_state['df']
    competencies = st.session_state['competencies']
    domains = st.session_state['domains']
    excluded_competencies = st.session_state['excluded_competencies']
    valid_terms = st.session_state['valid_terms']
    
    # Display exclusion message if any competencies are excluded
    if excluded_competencies:
        st.warning(f"The following competencies were excluded from analysis due to missing counts: {', '.join(excluded_competencies)}")
    
    # Main Analysis Page (only page now)
    st.header("Data Analysis: Summary and Visualizations")
    
    # Clustered Stacked Bar Charts
    st.subheader("Clustered Stacked Bar Charts")
    for domain_idx, domain in enumerate(domains):
        st.markdown("<br>", unsafe_allow_html=True)  # Add vertical space
        st.markdown(f'<div style="font-size: 20px; font-weight: bold;">Domain: {domain}</div>', unsafe_allow_html=True)
        domain_comps = df[df['Domain'] == domain][['Comptnum', 'Competency']].drop_duplicates().sort_values('Comptnum')
        
        # Prepare data for all competencies in this domain
        plot_data = []
        for _, row in domain_comps.iterrows():
            comptnum, competency = row['Comptnum'], row['Competency']
            wrapped_competency = wrap_text(f"{competency} (#{comptnum})", width=30)
            comp_data = df[df['Competency'] == competency]
            for term in valid_terms:
                term_data = comp_data[comp_data['Term'] == term]
                if not term_data.empty:
                    counts = term_data[['Count_1', 'Count_2', 'Count_3', 'Count_4']].iloc[0].values
                    total = sum(counts)
                    percentages = [float("{:.2f}".format((c / total * 100))) if total > 0 else "0.00" for c in counts]
                    for value in range(1, 5):
                        plot_data.append({
                            'Competency': wrapped_competency,
                            'Term': f"Term {term}",
                            'Likert Value': likert_labels[str(value)],
                            'Percentage': percentages[value-1]
                        })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create clustered stacked bar chart with data labels
        fig = go.Figure()
        for value in range(1, 5):
            label = likert_labels[str(value)]
            value_data = plot_df[plot_df['Likert Value'] == label]
            fig.add_trace(go.Bar(
                x=[value_data['Competency'], value_data['Term']],
                y=value_data['Percentage'],
                name=label,
                marker_color=color_maps.get(domain_idx % len(color_maps), color_maps[0])[value],
                hovertemplate='%{y:.2f}%',
                text=value_data['Percentage'].apply(lambda x: f"{int(x)}%" if x > 0 else ""),
                textposition='inside',
                textfont=dict(size=12, color='#333333')
            ))
        
        fig.update_layout(
            barmode='stack',
            xaxis_title="",
            xaxis_title_font=dict(size=12, color='#333333'),
            yaxis_title="Percentage (%)",
            yaxis_title_font=dict(size=12, color='#333333'),
            yaxis=dict(range=[0, 100], title="Percentage (%)", tickfont=dict(size=12, color='#333333')),
            legend_title="",
            title=f"Competency Ratings by Term (Domain: {domain})",
            font=dict(size=12, color='#333333'),
            xaxis_tickangle=45,
            xaxis_tickfont=dict(size=12, color='#333333'),
            height=600,
            bargap=0.3,  # Gap between competency clusters
            bargroupgap=0.05  # Gap within competency clusters
        )
        fig.update_xaxes(type='multicategory')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation of the Chart
        st.subheader(f"Interpretation of Competency Ratings for {domain}")
        # Calculate average ratings by term and Likert value
        avg_ratings = plot_df.groupby(['Term', 'Likert Value'])['Percentage'].mean().unstack().fillna(0)
        
        # Initialize bullet points
        insights = []
        
        # Insight 1: Trend in "Well developed" ratings across terms
        well_developed = avg_ratings.get(likert_labels['4'], pd.Series(0))
        if len(well_developed) > 1:
            first_term = well_developed.index[0]
            last_term = well_developed.index[-1]
            change = well_developed[last_term] - well_developed[first_term]
            if change > 0:
                insights.append(f"- **Improvement in '{likert_labels['4']}' ratings**: The average percentage of '{likert_labels['4']}' ratings increased by {change:.1f}% from {first_term} ({well_developed[first_term]:.1f}%) to {last_term} ({well_developed[last_term]:.1f}%).")
            elif change < 0:
                insights.append(f"- **Decline in '{likert_labels['4']}' ratings**: The average percentage of '{likert_labels['4']}' ratings decreased by {abs(change):.1f}% from {first_term} ({well_developed[first_term]:.1f}%) to {last_term} ({well_developed[last_term]:.1f}%).")
            else:
                insights.append(f"- **Stable '{likert_labels['4']}' ratings**: The average percentage of '{likert_labels['4']}' ratings remained steady at {well_developed[first_term]:.1f}% from {first_term} to {last_term}.")
        
        # Insight 2: Dominant rating category in the latest term
        latest_term = plot_df['Term'].max()
        latest_ratings = avg_ratings.loc[latest_term]
        dominant_rating = latest_ratings.idxmax()
        dominant_percentage = latest_ratings[dominant_rating]
        insights.append(f"- **Dominant rating in {latest_term}**: '{dominant_rating}' is the most common rating, averaging {dominant_percentage:.1f}% across competencies.")
        
        # Insight 3: Competency with the highest "Well developed" rating in the latest term
        latest_term_data = plot_df[plot_df['Term'] == latest_term]
        well_developed_data = latest_term_data[latest_term_data['Likert Value'] == likert_labels['4']]
        if not well_developed_data.empty:
            top_competency = well_developed_data.loc[well_developed_data['Percentage'].idxmax()]
            insights.append(f"- **Top performer in {latest_term}**: '{top_competency['Competency']}' has the highest '{likert_labels['4']}' rating at {top_competency['Percentage']:.1f}%.")
        
        # Insight 4: Largest improvement in reducing "Not developed" ratings
        not_developed = avg_ratings.get(likert_labels['1'], pd.Series(0))
        if len(not_developed) > 1:
            first_term = not_developed.index[0]
            last_term = not_developed.index[-1]
            reduction = not_developed[first_term] - not_developed[last_term]
            if reduction > 10:  # Only highlight significant reductions (>10%)
                # Find the competency with the largest reduction
                not_developed_pivot = plot_df[plot_df['Likert Value'] == likert_labels['1']].pivot(index='Competency', columns='Term', values='Percentage').fillna(0)
                if first_term in not_developed_pivot.columns and last_term in not_developed_pivot.columns:
                    not_developed_pivot['Reduction'] = not_developed_pivot[first_term] - not_developed_pivot[last_term]
                    if not not_developed_pivot.empty:
                        top_reducer = not_developed_pivot['Reduction'].idxmax()
                        reduction_value = not_developed_pivot.loc[top_reducer, 'Reduction']
                        if reduction_value > 0:
                            insights.append(f"- **Largest reduction in '{likert_labels['1']}' ratings**: '{top_reducer}' reduced its '{likert_labels['1']}' percentage by {reduction_value:.1f}% from {first_term} to {last_term}.")
        
        # Insight 5: Overall trend (if not already covered)
        if len(insights) < 3 and len(valid_terms) > 1:
            # Check for overall improvement by looking at the shift from lower to higher ratings
            lower_ratings = avg_ratings[[likert_labels['1'], likert_labels['2']]].sum(axis=1)
            higher_ratings = avg_ratings[[likert_labels['3'], likert_labels['4']]].sum(axis=1)
            first_term = lower_ratings.index[0]
            last_term = lower_ratings.index[-1]
            lower_change = lower_ratings[last_term] - lower_ratings[first_term]
            higher_change = higher_ratings[last_term] - higher_ratings[first_term]
            if higher_change > 0 and lower_change < 0:
                insights.append(f"- **Overall improvement**: Higher ratings ('{likert_labels['3']}' and '{likert_labels['4']}') increased by {higher_change:.1f}% on average, while lower ratings ('{likert_labels['1']}' and '{likert_labels['2']}') decreased by {abs(lower_change):.1f}% from {first_term} to {last_term}.")
        
        # Ensure at least 3 bullet points by adding a generic insight if needed
        if len(insights) < 3:
            insights.append(f"- **Number of terms analyzed**: This domain includes data for {len(valid_terms)} terms ({', '.join([f'Term {t}' for t in valid_terms])}).")
        
        # Display the insights (limit to 5 bullet points)
        for insight in insights[:5]:
            st.markdown(insight)
