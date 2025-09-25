"""HTML report generation system.

This module provides HTML-based report generation with interactive elements,
responsive design, and comprehensive formatting for morphogenesis simulation results.
"""

import base64
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import io

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .generator import ReportConfig, ReportSection, SectionType

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """HTML report generator with interactive features and responsive design."""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.template_dir = Path(__file__).parent / "templates"

    async def generate(
        self,
        sections: List[ReportSection],
        report_data: Dict[str, Any],
        figures: Dict[str, plt.Figure],
        tables: Dict[str, pd.DataFrame]
    ) -> Path:
        """Generate HTML report.

        Args:
            sections: List of report sections to include
            report_data: Report data dictionary
            figures: Dictionary of matplotlib figures
            tables: Dictionary of pandas DataFrames

        Returns:
            Path to generated HTML report
        """
        logger.info("Generating HTML report")

        # Prepare output path
        if not self.config.filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"morphogenesis_report_{timestamp}.html"
        else:
            filename = self.config.filename
            if not filename.endswith('.html'):
                filename += '.html'

        output_path = self.config.output_dir / filename

        # Generate HTML content
        html_content = await self._generate_html_content(sections, report_data, figures, tables)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {output_path}")
        return output_path

    async def _generate_html_content(
        self,
        sections: List[ReportSection],
        report_data: Dict[str, Any],
        figures: Dict[str, plt.Figure],
        tables: Dict[str, pd.DataFrame]
    ) -> str:
        """Generate complete HTML content."""
        # Convert figures to base64 images
        figure_images = await self._convert_figures_to_images(figures)

        # Generate HTML components
        html_head = self._generate_html_head()
        html_header = self._generate_html_header(report_data)
        html_nav = self._generate_navigation(sections)
        html_sections = await self._generate_html_sections(sections, figure_images, tables)
        html_footer = self._generate_html_footer()

        # Combine into complete HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
{html_head}
<body>
    {html_header}
    <div class="container">
        <div class="row">
            <div class="col-md-3">
                {html_nav}
            </div>
            <div class="col-md-9">
                {html_sections}
            </div>
        </div>
    </div>
    {html_footer}
    {self._generate_javascript()}
</body>
</html>"""

        return html_content

    def _generate_html_head(self) -> str:
        """Generate HTML head section with styles and meta tags."""
        return f"""<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>

    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">

    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <!-- Plotly.js for interactive plots -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

    <!-- Custom styles -->
    <style>
        {self._get_custom_css()}
    </style>
</head>"""

    def _get_custom_css(self) -> str:
        """Get custom CSS styles."""
        theme_colors = self._get_theme_colors()

        return f"""
        :root {{
            --primary-color: {theme_colors['primary']};
            --secondary-color: {theme_colors['secondary']};
            --accent-color: {theme_colors['accent']};
            --background-color: {theme_colors['background']};
            --text-color: {theme_colors['text']};
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
        }}

        .report-header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .report-title {{
            font-size: 2.5rem;
            font-weight: 300;
            margin-bottom: 0.5rem;
        }}

        .report-subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
        }}

        .nav-pills .nav-link {{
            color: var(--primary-color);
            border-radius: 0.5rem;
            margin-bottom: 0.25rem;
        }}

        .nav-pills .nav-link.active {{
            background-color: var(--primary-color);
        }}

        .nav-pills .nav-link:hover {{
            background-color: var(--accent-color);
            color: white;
        }}

        .report-section {{
            background: white;
            border-radius: 0.5rem;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border-left: 4px solid var(--primary-color);
        }}

        .section-title {{
            color: var(--primary-color);
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }}

        .metric-card {{
            background: var(--background-color);
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid #e0e0e0;
            text-align: center;
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
            margin-bottom: 0.25rem;
        }}

        .metric-label {{
            color: var(--text-color);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .figure-container {{
            text-align: center;
            margin: 2rem 0;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 0.5rem;
        }}

        .figure-image {{
            max-width: 100%;
            height: auto;
            border-radius: 0.25rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .figure-caption {{
            font-style: italic;
            color: #666;
            margin-top: 1rem;
            font-size: 0.9rem;
        }}

        .data-table {{
            margin: 1.5rem 0;
        }}

        .table {{
            font-size: 0.9rem;
        }}

        .table th {{
            background-color: var(--primary-color);
            color: white;
            border: none;
            font-weight: 500;
        }}

        .table-hover tbody tr:hover {{
            background-color: rgba(var(--primary-color), 0.05);
        }}

        .highlight-box {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-left: 4px solid var(--accent-color);
            padding: 1.5rem;
            margin: 1.5rem 0;
            border-radius: 0.25rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }}

        .collapsible-section {{
            border: 1px solid #e0e0e0;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }}

        .collapsible-header {{
            background: #f8f9fa;
            padding: 1rem;
            cursor: pointer;
            border-radius: 0.5rem;
            transition: background-color 0.3s;
        }}

        .collapsible-header:hover {{
            background: #e9ecef;
        }}

        .collapsible-content {{
            padding: 1rem;
            display: none;
        }}

        .collapsible-content.active {{
            display: block;
        }}

        .footer {{
            background: #343a40;
            color: white;
            padding: 2rem 0;
            margin-top: 3rem;
        }}

        .interactive-plot {{
            margin: 2rem 0;
            padding: 1rem;
            background: white;
            border-radius: 0.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        @media (max-width: 768px) {{
            .report-title {{
                font-size: 2rem;
            }}

            .col-md-3, .col-md-9 {{
                width: 100%;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        /* Print styles */
        @media print {{
            .nav-pills {{
                display: none;
            }}

            .report-section {{
                break-inside: avoid;
                box-shadow: none;
            }}

            .figure-container {{
                break-inside: avoid;
            }}
        }}
        """

    def _get_theme_colors(self) -> Dict[str, str]:
        """Get theme colors based on configuration."""
        themes = {
            'default': {
                'primary': '#007bff',
                'secondary': '#6c757d',
                'accent': '#28a745',
                'background': '#f8f9fa',
                'text': '#212529'
            },
            'scientific': {
                'primary': '#2c3e50',
                'secondary': '#34495e',
                'accent': '#3498db',
                'background': '#ecf0f1',
                'text': '#2c3e50'
            },
            'nature': {
                'primary': '#27ae60',
                'secondary': '#2ecc71',
                'accent': '#f39c12',
                'background': '#f8fffe',
                'text': '#2c3e50'
            }
        }
        return themes.get(self.config.theme, themes['default'])

    def _generate_html_header(self, report_data: Dict[str, Any]) -> str:
        """Generate report header section."""
        generation_time = report_data.get('generation_timestamp', datetime.now())
        cell_count = report_data.get('cell_count', 0)
        timestep_count = report_data.get('timestep_count', 0)

        return f"""
        <header class="report-header">
            <div class="container">
                <div class="row">
                    <div class="col-md-8">
                        <h1 class="report-title">{self.config.title}</h1>
                        <p class="report-subtitle">By {self.config.author}</p>
                        {f'<p class="report-subtitle">{self.config.description}</p>' if self.config.description else ''}
                    </div>
                    <div class="col-md-4 text-md-end">
                        <div class="mt-3">
                            <small class="text-light">
                                Generated: {generation_time.strftime("%Y-%m-%d %H:%M:%S")}<br>
                                Cells: {cell_count:,} | Timesteps: {timestep_count:,}
                            </small>
                        </div>
                    </div>
                </div>
            </div>
        </header>"""

    def _generate_navigation(self, sections: List[ReportSection]) -> str:
        """Generate navigation sidebar."""
        toc_sections = [s for s in sections if s.include_in_toc]
        toc_sections.sort(key=lambda s: s.order)

        nav_items = []
        for section in toc_sections:
            section_id = self._generate_section_id(section.title)
            nav_items.append(f'<li class="nav-item">'
                           f'<a class="nav-link" href="#{section_id}">{section.title}</a>'
                           f'</li>')

        return f"""
        <nav class="sticky-top" style="top: 1rem;">
            <h5 class="mb-3">Table of Contents</h5>
            <ul class="nav nav-pills flex-column">
                {chr(10).join(nav_items)}
            </ul>

            <div class="mt-4">
                <h6>Quick Actions</h6>
                <button class="btn btn-outline-primary btn-sm mb-2 w-100" onclick="printReport()">
                    Print Report
                </button>
                <button class="btn btn-outline-secondary btn-sm w-100" onclick="toggleAllSections()">
                    Toggle All Sections
                </button>
            </div>
        </nav>"""

    async def _generate_html_sections(
        self,
        sections: List[ReportSection],
        figure_images: Dict[str, str],
        tables: Dict[str, pd.DataFrame]
    ) -> str:
        """Generate HTML for all report sections."""
        sections.sort(key=lambda s: s.order)

        html_sections = []
        for section in sections:
            section_html = await self._generate_section_html(section, figure_images, tables)
            html_sections.append(section_html)

        return chr(10).join(html_sections)

    async def _generate_section_html(
        self,
        section: ReportSection,
        figure_images: Dict[str, str],
        tables: Dict[str, pd.DataFrame]
    ) -> str:
        """Generate HTML for a single section."""
        section_id = self._generate_section_id(section.title)

        # Generate section content
        content_html = self._markdown_to_html(section.content or "")

        # Add visualizations
        viz_html = ""
        if section.visualizations:
            viz_html = self._generate_visualizations_html(section.visualizations, figure_images)

        # Add data tables
        table_html = ""
        if section.data and 'tables' in section.data:
            table_html = self._generate_tables_html(section.data['tables'])

        # Add metrics if this is a summary section
        metrics_html = ""
        if section.section_type == SectionType.SUMMARY:
            metrics_html = self._generate_summary_metrics(section.data)

        # Add interactive elements for analysis sections
        interactive_html = ""
        if section.section_type == SectionType.ANALYSIS and self.config.include_interactive_plots:
            interactive_html = self._generate_interactive_elements(section)

        return f"""
        <section id="{section_id}" class="report-section">
            <h2 class="section-title">{section.title}</h2>
            {metrics_html}
            {content_html}
            {viz_html}
            {interactive_html}
            {table_html}
        </section>"""

    def _generate_summary_metrics(self, section_data: Optional[Dict[str, Any]]) -> str:
        """Generate summary metrics cards."""
        if not section_data:
            return ""

        # Extract key metrics for display
        metrics = []

        # Try to get sorting analysis data
        sorting_data = section_data.get('sorting_analysis', {}) if section_data else {}
        if sorting_data:
            improvement_rate = sorting_data.get('improvement_rate', 0)
            metrics.append({
                'value': f"{improvement_rate:.1%}",
                'label': "Improvement Rate"
            })

            convergence_time = sorting_data.get('convergence_timestep', 0)
            metrics.append({
                'value': str(convergence_time),
                'label': "Convergence Time"
            })

        # Try to get performance data
        performance_data = section_data.get('performance_analysis', {}) if section_data else {}
        if performance_data:
            avg_metrics = performance_data.get('avg_metrics', {})
            for metric, value in list(avg_metrics.items())[:2]:  # Show first 2 metrics
                if isinstance(value, (int, float)):
                    metrics.append({
                        'value': f"{value:.3f}",
                        'label': metric.replace('_', ' ').title()
                    })

        if not metrics:
            return ""

        metrics_html = []
        for metric in metrics:
            metrics_html.append(f"""
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-value">{metric['value']}</div>
                        <div class="metric-label">{metric['label']}</div>
                    </div>
                </div>""")

        return f"""
        <div class="row stats-grid mb-4">
            {chr(10).join(metrics_html)}
        </div>"""

    def _generate_visualizations_html(self, viz_names: List[str], figure_images: Dict[str, str]) -> str:
        """Generate HTML for visualizations."""
        viz_html = []

        for viz_name in viz_names:
            if viz_name in figure_images:
                image_data = figure_images[viz_name]
                caption = self._get_figure_caption(viz_name)

                viz_html.append(f"""
                <div class="figure-container">
                    <img src="data:image/png;base64,{image_data}"
                         alt="{viz_name}"
                         class="figure-image">
                    <div class="figure-caption">{caption}</div>
                </div>""")

        return chr(10).join(viz_html)

    def _generate_tables_html(self, tables_dict: Dict[str, pd.DataFrame]) -> str:
        """Generate HTML for data tables."""
        tables_html = []

        for table_name, df in tables_dict.items():
            if df.empty:
                continue

            # Limit table size for display
            display_df = df.head(100)  # Show first 100 rows
            table_html = display_df.to_html(
                classes="table table-striped table-hover",
                table_id=f"table-{table_name.replace('_', '-')}",
                escape=False
            )

            # Add collapsible wrapper for large tables
            if len(df) > 20:
                tables_html.append(f"""
                <div class="collapsible-section">
                    <div class="collapsible-header" onclick="toggleCollapsible('{table_name}')">
                        <strong>{table_name.replace('_', ' ').title()}</strong>
                        <small class="text-muted">({len(df)} rows, click to expand)</small>
                    </div>
                    <div id="content-{table_name}" class="collapsible-content">
                        <div class="data-table">
                            {table_html}
                        </div>
                    </div>
                </div>""")
            else:
                tables_html.append(f"""
                <div class="data-table">
                    <h5>{table_name.replace('_', ' ').title()}</h5>
                    {table_html}
                </div>""")

        return chr(10).join(tables_html)

    def _generate_interactive_elements(self, section: ReportSection) -> str:
        """Generate interactive elements for analysis sections."""
        if section.section_type != SectionType.ANALYSIS:
            return ""

        # This is a placeholder for interactive elements
        # In a full implementation, you might generate interactive charts here
        return """
        <div class="interactive-plot">
            <div id="interactive-chart-placeholder">
                <p class="text-muted text-center">
                    Interactive visualizations can be added here using Chart.js or Plotly.js
                </p>
            </div>
        </div>"""

    def _generate_html_footer(self) -> str:
        """Generate HTML footer."""
        return f"""
        <footer class="footer">
            <div class="container">
                <div class="row">
                    <div class="col-md-6">
                        <h5>Morphogenesis Simulation Report</h5>
                        <p class="text-light">
                            Generated by the Enhanced Morphogenesis Framework<br>
                            <small>Async cell agent simulation system</small>
                        </p>
                    </div>
                    <div class="col-md-6 text-md-end">
                        <h6>Report Information</h6>
                        <p class="text-light">
                            <small>
                                Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
                                Format: HTML Interactive Report<br>
                                Version: 1.0
                            </small>
                        </p>
                    </div>
                </div>
            </div>
        </footer>"""

    def _generate_javascript(self) -> str:
        """Generate JavaScript for interactive features."""
        return """
        <!-- Bootstrap JS -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>

        <script>
            // Smooth scrolling for navigation
            document.querySelectorAll('a.nav-link').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                });
            });

            // Toggle collapsible sections
            function toggleCollapsible(sectionName) {
                const content = document.getElementById(`content-${sectionName}`);
                if (content) {
                    content.classList.toggle('active');
                }
            }

            // Toggle all sections
            function toggleAllSections() {
                const sections = document.querySelectorAll('.collapsible-content');
                const anyActive = Array.from(sections).some(s => s.classList.contains('active'));

                sections.forEach(section => {
                    if (anyActive) {
                        section.classList.remove('active');
                    } else {
                        section.classList.add('active');
                    }
                });
            }

            // Print report
            function printReport() {
                window.print();
            }

            // Highlight active section in navigation
            window.addEventListener('scroll', function() {
                const sections = document.querySelectorAll('.report-section');
                const navLinks = document.querySelectorAll('.nav-link');

                let current = '';
                sections.forEach(section => {
                    const sectionTop = section.offsetTop;
                    const sectionHeight = section.clientHeight;
                    if (scrollY >= sectionTop - 200) {
                        current = section.getAttribute('id');
                    }
                });

                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${current}`) {
                        link.classList.add('active');
                    }
                });
            });

            // Initialize tooltips
            var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });

            console.log('Morphogenesis Report loaded successfully');
        </script>"""

    async def _convert_figures_to_images(self, figures: Dict[str, plt.Figure]) -> Dict[str, str]:
        """Convert matplotlib figures to base64 encoded images."""
        figure_images = {}

        for fig_name, figure in figures.items():
            try:
                # Save figure to bytes buffer
                buffer = io.BytesIO()
                figure.savefig(
                    buffer,
                    format='png',
                    dpi=self.config.dpi,
                    bbox_inches='tight',
                    facecolor='white'
                )
                buffer.seek(0)

                # Encode as base64
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                figure_images[fig_name] = image_base64

                buffer.close()

            except Exception as e:
                logger.warning(f"Failed to convert figure {fig_name} to image: {e}")

        return figure_images

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert simple markdown to HTML."""
        if not markdown_text:
            return ""

        html = markdown_text

        # Headers
        html = html.replace('### ', '<h4>').replace('\n', '</h4>\n', 1) if '### ' in html else html
        html = html.replace('## ', '<h3>').replace('\n', '</h3>\n', 1) if '## ' in html else html
        html = html.replace('# ', '<h2>').replace('\n', '</h2>\n', 1) if '# ' in html else html

        # Bold and italic
        html = html.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
        html = html.replace('*', '<em>', 1).replace('*', '</em>', 1)

        # Lists
        lines = html.split('\n')
        in_list = False
        processed_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                if not in_list:
                    processed_lines.append('<ul>')
                    in_list = True
                processed_lines.append(f'<li>{line[2:]}</li>')
            else:
                if in_list:
                    processed_lines.append('</ul>')
                    in_list = False
                if line:
                    processed_lines.append(f'<p>{line}</p>')
                else:
                    processed_lines.append('')

        if in_list:
            processed_lines.append('</ul>')

        return '\n'.join(processed_lines)

    def _generate_section_id(self, title: str) -> str:
        """Generate HTML ID from section title."""
        return title.lower().replace(' ', '-').replace('&', 'and')

    def _get_figure_caption(self, viz_name: str) -> str:
        """Get caption for a figure."""
        captions = {
            'sorting_progress': 'Evolution of sorting progress showing inversions and completion percentage over time.',
            'cell_trajectories': 'Sample cell trajectories showing movement patterns during simulation.',
            'performance_metrics': 'Key performance metrics tracked throughout the simulation.',
            'state_distribution': 'Distribution of cell states over the course of the simulation.',
            'efficiency_comparison': 'Comparison of sorting efficiency metrics and algorithm performance.'
        }
        return captions.get(viz_name, f'Visualization: {viz_name.replace("_", " ").title()}')