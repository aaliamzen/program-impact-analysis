import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Streamlit app configuration
st.set_page_config(page_title="Teaching Competency Analysis", layout="wide")

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

# Likert scale labels
likert_labels = {
    '1': 'Not developed',
    '2': 'Developing towards',
    '3': 'Developed',
    '4': 'Well developed'
}

# Domain-specific color palettes
color_maps = {
    0: {  # Domain 1 (e.g., Instructional Skills)
        'Not developed': '#E6F5F6',
        'Developing towards': '#C8EAED',
        'Developed': '#A6E0E4',
        'Well developed': '#83D5DA'
    },
    1: {  # Domain 2 (e.g., Management Skills)
        'Not developed': '#D8DEEF',
        'Developing towards': '#A8B5DB',
        'Developed': '#728AC3',
        'Well developed': '#334F9E'
    },
    2: {  # Domain 3 (darker lime green)
        'Not developed': '#E8F5C8',
        'Developing towards': '#D6F0A0',
        'Developed': '#C2EA75',
        'Well developed': '#B8E34D'
    },
    3: {  # Domain 4
        'Not developed': '#ECE9F8',
        'Developing towards': '#DAD3F1',
        'Developed': '#C8BCE9',
        'Well developed': '#B2A5E4'
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

# Sidebar for file upload
st.sidebar.title("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])

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
        valid_terms = [1, 2, 3]
        
        if not all(df['Term'].isin(valid_terms)):
            st.error("Term column must contain only: 1, 2, 3")
        elif df.empty:
            st.error("No valid competencies with counts found in the dataset.")
        else:
            # Store processed data in session state
            st.session_state['df'] = df
            st.session_state['competencies'] = competencies
            st.session_state['domains'] = domains
            st.session_state['excluded_competencies'] = excluded_competencies
            st.session_state['valid_terms'] = valid_terms

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
        for value in likert_labels.values():
            value_data = plot_df[plot_df['Likert Value'] == value]
            fig.add_trace(go.Bar(
                x=[value_data['Competency'], value_data['Term']],
                y=value_data['Percentage'],
                name=value,
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