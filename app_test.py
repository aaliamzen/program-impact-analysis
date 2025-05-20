import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
from IPython.display import HTML

# Streamlit app configuration
st.set_page_config(page_title="Teaching Competency Analysis", layout="wide")

# Custom CSS to override Streamlit's default table styles
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

# Sidebar for file upload and navigation
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

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Main Analysis", "Competency Progress"])

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
    
    if page == "Main Analysis":
        # Main Analysis Page
        st.header("Data Analysis: Summary and Visualizations")
        
        # Summary Tables (one per row)
        st.subheader("Summary of Competency Ratings")
        for i in range(len(competencies)):
            row = competencies.iloc[i]
            comptnum, competency = row['Comptnum'], row['Competency']
            st.markdown(f"**{competency} (Competency #{comptnum})**")
            comp_data = df[df['Competency'] == competency]
            table_data = {
                'Term': [1, 2, 3],
                'Not developed (%)': ['0.00', '0.00', '0.00'],
                'Developing towards (%)': ['0.00', '0.00', '0.00'],
                'Developed (%)': ['0.00', '0.00', '0.00'],
                'Well developed (%)': ['0.00', '0.00', '0.00']
            }
            for term in valid_terms:
                term_data = comp_data[comp_data['Term'] == term]
                if not term_data.empty:
                    counts = term_data[['Count_1', 'Count_2', 'Count_3', 'Count_4']].iloc[0].values
                    total = sum(counts)
                    percentages = ["{:.2f}".format((c / total * 100)) if total > 0 else "0.00" for c in counts]
                    table_data['Not developed (%)'][term-1] = percentages[0]
                    table_data['Developing towards (%)'][term-1] = percentages[1]
                    table_data['Developed (%)'][term-1] = percentages[2]
                    table_data['Well developed (%)'][term-1] = percentages[3]
            
            summary_df = pd.DataFrame(table_data)
            summary_df.set_index('Term', inplace=True)

            # Construct the HTML table with embedded CSS
            html = '''
            <style>
            .styled-table {
                border-collapse: collapse;
                margin: 10px 0;
                font-size: 14px;
                width: 100%;
            }
            .styled-table thead tr {
                border-top: 1px solid black;
                border-bottom: 1px solid black;
            }
            .styled-table th {
                padding: 8px 12px;
                text-align: center;
                border: none;
                font-weight: normal;
            }
            .styled-table td {
                padding: 8px 12px;
                text-align: center;
                border: none;
            }
            .styled-table td:first-child {
                text-align: left;
                font-weight: normal;
            }
            .styled-table td:not(:first-child) {
                text-align: right;
            }
            .styled-table tbody tr:last-child {
                border-top: none;
            }
            .styled-table tbody tr:last-child td {
                border-top: none;
            }
            .progress-table-container {
                overflow-x: auto;
                border-bottom: 1px solid black;
            }
            </style>
            <div class="progress-table-container">
            <table class="styled-table">
            <thead>
            <tr style="border-top: 1px solid black; border-bottom: 1px solid black;">
            '''
            # Add headers with inline font-weight
            html += '<th style="font-weight: normal;">Term</th>'
            for col in summary_df.columns:
                html += f'<th style="font-weight: normal;">{col}</th>'
            html += '</tr>\n</thead>\n<tbody>\n'
            # Add rows
            for idx, row in summary_df.iterrows():
                html += '<tr>'
                html += f'<td>{row.name}</td>'
                for col in summary_df.columns:
                    html += f'<td>{row[col]}</td>'
                html += '</tr>\n'
            html += '</tbody>\n</table>\n</div>'

            # Render table using custom component
            components.html(html, height=150)
        
        # Clustered, Stacked Bar Charts
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
    
    elif page == "Competency Progress":
        # Competency Progress Page
        st.header("Competency Progress: Term 1 to Term 3")
        
        # Calculate means for Term 1 and Term 3, and progress
        progress_data = []
        for _, row in competencies.iterrows():
            comptnum, competency = row['Comptnum'], row['Competency']
            comp_data = df[df['Competency'] == competency]
            domain = comp_data['Domain'].iloc[0] if not comp_data.empty else ""
            
            # Calculate mean for Term 1
            term1_data = comp_data[comp_data['Term'] == 1]
            mean_term1 = float('nan')
            if not term1_data.empty:
                counts = term1_data[['Count_1', 'Count_2', 'Count_3', 'Count_4']].iloc[0].values
                total = sum(counts)
                if total > 0:
                    mean_term1 = (1 * counts[0] + 2 * counts[1] + 3 * counts[2] + 4 * counts[3]) / total
            
            # Calculate mean for Term 3
            term3_data = comp_data[comp_data['Term'] == 3]
            mean_term3 = float('nan')
            if not term3_data.empty:
                counts = term3_data[['Count_1', 'Count_2', 'Count_3', 'Count_4']].iloc[0].values
                total = sum(counts)
                if total > 0:
                    mean_term3 = (1 * counts[0] + 2 * counts[1] + 3 * counts[2] + 4 * counts[3]) / total
            
            # Calculate progress
            progress = float('nan')
            if not pd.isna(mean_term1) and not pd.isna(mean_term3):
                progress = mean_term3 - mean_term1
            
            progress_data.append({
                'Comptnum': comptnum,
                'Competency': competency,
                'Domain': domain,
                'Term 1 Mean': mean_term1,
                'Term 3 Mean': mean_term3,
                'Progress': progress
            })
        
        progress_df = pd.DataFrame(progress_data)

        # Debug: Print the number of rows in progress_df
        st.write(f"Number of competencies in progress_df: {len(progress_df)}")
        
        print(progress_df.shape)  
                    
        # APA-Style Table for Means and Progress
        st.subheader("Competency Progress Table")

        # Format the data  
        table_data = progress_df[['Competency', 'Term 1 Mean', 'Term 3 Mean', 'Progress']].copy()  
        table_data['Term 1 Mean'] = table_data['Term 1 Mean'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "–")  
        table_data['Term 3 Mean'] = table_data['Term 3 Mean'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "–")  
        table_data['Progress'] = table_data['Progress'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "–")  
        table_data = table_data.rename(columns={'Competency': 'Competency Name'})  
          
        # Show the first few rows to confirm  
        print(table_data.head())  
        print("Number of rows in table_data:", len(table_data))  

        # The error occurred because the column names in table_data are 'Term 1' and 'Term 3', not 'Term 1 Mean' and 'Term 3 Mean'.
        # Let's fix the column names and re-run the HTML table rendering, showing all rows.

        # Use the correct column names
        html = '''
        <style>
        .styled-table {
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 14px;
            width: 100%;
            display: table;
        }
        .progress-table-container {
            overflow-y: visible;
            overflow-x: auto;
            border-bottom: 1px solid black;
            max-height: none !important;
            height: auto !important;
        }
        .styled-table thead tr {
            border-top: 1px solid black;
            border-bottom: 1px solid black;
        }
        .styled-table th {
            padding: 8px 12px;
            text-align: center;
            border: none;
            font-weight: normal;
        }
        .styled-table td {
            padding: 8px 12px;
            text-align: center;
            border: none;
        }
        .styled-table td:first-child {
            text-align: left;
            font-weight: normal;
        }
        .styled-table td:not(:first-child) {
            text-align: right;
        }
        </style>
        <div class="progress-table-container">
        <table class="styled-table">
        <thead>
        <tr style="border-top: 1px solid black; border-bottom: 1px solid black;">
        '''

        # Add headers
        for col in progress_df.columns:
            html += '<th style="font-weight: normal;">' + col + '</th>'
        html += '</tr>\
        </thead>\
        <tbody>\
        '

        # Add all rows
        row_count = 0
        for idx, row in progress_df.iterrows():
            row_count += 1
            html += '<tr>'
            html += '<td>' + str(row['Competency']) + '</td>'
            html += '<td>' + str(format(row['Term 1 Mean'], '.2f')) + '</td>'
            html += '<td>' + str(format(row['Term 3 Mean'], '.2f')) + '</td>'
            html += '<td>' + str(format(row['Progress'], '.2f')) + '</td>'
            html += '</tr>\
        '
        html += '</tbody>\
        </table>\
        </div>'

        print('Number of rows processed:', row_count)

        #st.components.v1.html(html, height=None)  

        #display(HTML(html))

        # Line Charts for Competency Trends
        st.subheader("Competency Trends (Line Charts)")
        # Limit to first 13 valid competencies
        selected_competencies = competencies.head(13)
        num_charts = len(selected_competencies)
        rows = (num_charts + 3) // 4  # Ceiling division for number of rows
        
        for row_idx in range(rows):
            cols = st.columns(4)  # 4 charts per row
            for col_idx in range(4):
                chart_idx = row_idx * 4 + col_idx
                if chart_idx < num_charts:
                    row = selected_competencies.iloc[chart_idx]
                    comptnum, competency = row['Comptnum'], row['Competency']
                    comp_data = df[df['Competency'] == competency]
                    domain = comp_data['Domain'].iloc[0] if not comp_data.empty else ""
                    domain_idx = list(domains).index(domain) if domain in domains else 0
                    
                    # Calculate means for Terms 1, 2, 3
                    means = []
                    for term in valid_terms:
                        term_data = comp_data[comp_data['Term'] == term]
                        mean = float('nan')
                        if not term_data.empty:
                            counts = term_data[['Count_1', 'Count_2', 'Count_3', 'Count_4']].iloc[0].values
                            total = sum(counts)
                            if total > 0:
                                mean = (1 * counts[0] + 2 * counts[1] + 3 * counts[2] + 4 * counts[3]) / total
                        means.append(mean)
                    
                    # Create line chart
                    with cols[col_idx]:
                        fig = go.Figure()
                        if not all(pd.isna(means)):
                            fig.add_trace(go.Scatter(
                                x=["Term 1", "Term 2", "Term 3"],
                                y=means,
                                mode='lines+markers',
                                line=dict(color=color_maps.get(domain_idx % len(color_maps), color_maps[0])['Well developed']),
                                marker=dict(size=8),
                                hovertemplate='Term: %{x}<br>Mean: %{y:.2f}<extra></extra>'
                            ))
                        
                        # Wrap title at 40 characters, use only competency name
                        wrapped_title = wrap_text(competency, width=40)
                        
                        # Add border around plot area
                        fig.add_shape(
                            type='rect',
                            xref='paper',
                            yref='paper',
                            x0=0,
                            y0=0,
                            x1=1,
                            y1=1,
                            line=dict(color='black', width=1)
                        )
                        
                        fig.update_layout(
                            xaxis_title="",
                            yaxis_title="Mean Score",
                            xaxis_title_font=dict(size=12, color='#333333'),
                            yaxis_title_font=dict(size=12, color='#333333'),
                            xaxis=dict(tickfont=dict(size=12, color='#333333')),
                            yaxis=dict(tickfont=dict(size=12, color='#333333'), range=[1, 4]),
                            title={
                                "text": wrapped_title,
                                "x": 0.5,
                                "xanchor": "center",
                                "font": {
                                    "family": "Verdana",
                                    "size": 12,
                                    "color": "#333333"
                                }
                            },
                            font=dict(size=6, color='#333333'),  # General font size for other elements
                            height=300,
                            showlegend=False,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)