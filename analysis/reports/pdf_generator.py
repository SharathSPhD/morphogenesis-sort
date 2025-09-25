"""PDF report generation system.

This module provides PDF-based report generation with professional formatting,
charts, tables, and publication-ready output for morphogenesis simulation results.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import tempfile
import io

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
        PageBreak, KeepTogether, Frame, PageTemplate
    )
    from reportlab.platypus.tableofcontents import TableOfContents
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from .generator import ReportConfig, ReportSection, SectionType

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """PDF report generator with professional formatting and charts."""

    def __init__(self, config: ReportConfig):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")

        self.config = config
        self.doc = None
        self.styles = None
        self.story = []
        self.toc = None

        # Page settings
        self.page_size = A4
        self.margins = {
            'left': 1.5 * cm,
            'right': 1.5 * cm,
            'top': 2 * cm,
            'bottom': 2 * cm
        }

    async def generate(
        self,
        sections: List[ReportSection],
        report_data: Dict[str, Any],
        figures: Dict[str, plt.Figure],
        tables: Dict[str, pd.DataFrame]
    ) -> Path:
        """Generate PDF report.

        Args:
            sections: List of report sections to include
            report_data: Report data dictionary
            figures: Dictionary of matplotlib figures
            tables: Dictionary of pandas DataFrames

        Returns:
            Path to generated PDF report
        """
        logger.info("Generating PDF report")

        # Prepare output path
        if not self.config.filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"morphogenesis_report_{timestamp}.pdf"
        else:
            filename = self.config.filename
            if not filename.endswith('.pdf'):
                filename += '.pdf'

        output_path = self.config.output_dir / filename

        # Initialize document
        self._initialize_document(output_path, report_data)

        # Generate content
        await self._generate_pdf_content(sections, report_data, figures, tables)

        # Build PDF
        self.doc.build(self.story, onFirstPage=self._first_page_layout, onLaterPages=self._later_page_layout)

        logger.info(f"PDF report generated: {output_path}")
        return output_path

    def _initialize_document(self, output_path: Path, report_data: Dict[str, Any]):
        """Initialize PDF document with styles and settings."""
        self.doc = SimpleDocTemplate(
            str(output_path),
            pagesize=self.page_size,
            leftMargin=self.margins['left'],
            rightMargin=self.margins['right'],
            topMargin=self.margins['top'],
            bottomMargin=self.margins['bottom'],
            title=self.config.title,
            author=self.config.author
        )

        # Initialize styles
        self._initialize_styles()

        # Initialize table of contents
        self.toc = TableOfContents()
        self.toc.levelStyles = [
            ParagraphStyle(fontSize=14, name='TOCHeading1', leftIndent=20, firstLineIndent=-20, spaceBefore=10, leading=16),
            ParagraphStyle(fontSize=12, name='TOCHeading2', leftIndent=40, firstLineIndent=-20, spaceBefore=5, leading=14),
        ]

    def _initialize_styles(self):
        """Initialize paragraph and text styles."""
        self.styles = getSampleStyleSheet()

        # Custom styles
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50')
        ))

        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            alignment=TA_CENTER,
            spaceAfter=20,
            textColor=colors.HexColor('#7f8c8d')
        ))

        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceBefore=30,
            spaceAfter=12,
            textColor=colors.HexColor('#2980b9'),
            borderWidth=1,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=5
        ))

        self.styles.add(ParagraphStyle(
            name='SubsectionTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=8,
            textColor=colors.HexColor('#34495e')
        ))

        self.styles.add(ParagraphStyle(
            name='MetricValue',
            fontSize=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#27ae60'),
            spaceBefore=10,
            spaceAfter=5
        ))

        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#7f8c8d'),
            spaceAfter=20
        ))

        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#7f8c8d'),
            fontName='Helvetica-Oblique',
            spaceBefore=5,
            spaceAfter=15
        ))

        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#95a5a6')
        ))

    async def _generate_pdf_content(
        self,
        sections: List[ReportSection],
        report_data: Dict[str, Any],
        figures: Dict[str, plt.Figure],
        tables: Dict[str, pd.DataFrame]
    ):
        """Generate all PDF content."""
        # Title page
        self._add_title_page(report_data)

        # Table of contents
        self.story.append(PageBreak())
        self.story.append(Paragraph("Table of Contents", self.styles['SectionTitle']))
        self.story.append(self.toc)
        self.story.append(PageBreak())

        # Content sections
        sections.sort(key=lambda s: s.order)
        for section in sections:
            await self._add_section(section, figures, tables)

        # Appendix with technical details
        self._add_technical_appendix(report_data)

    def _add_title_page(self, report_data: Dict[str, Any]):
        """Add title page to PDF."""
        # Title
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(self.config.title, self.styles['ReportTitle']))

        # Subtitle
        if self.config.description:
            self.story.append(Paragraph(self.config.description, self.styles['ReportSubtitle']))

        # Author and date
        self.story.append(Spacer(1, 1*inch))
        author_info = f"<b>Author:</b> {self.config.author}<br/>"
        generation_time = report_data.get('generation_timestamp', datetime.now())
        author_info += f"<b>Generated:</b> {generation_time.strftime('%Y-%m-%d %H:%M:%S')}<br/>"

        # Experiment info
        cell_count = report_data.get('cell_count', 0)
        timestep_count = report_data.get('timestep_count', 0)
        author_info += f"<b>Simulation Scale:</b> {cell_count:,} cells, {timestep_count:,} timesteps"

        self.story.append(Paragraph(author_info, self.styles['Normal']))

        # Summary metrics if available
        sorting_analysis = report_data.get('sorting_analysis', {})
        if sorting_analysis:
            self.story.append(Spacer(1, 0.5*inch))
            self.story.append(Paragraph("<b>Key Results Summary</b>", self.styles['SubsectionTitle']))

            metrics_data = [
                ['Metric', 'Value'],
                ['Improvement Rate', f"{sorting_analysis.get('improvement_rate', 0):.1%}"],
                ['Convergence Time', f"{sorting_analysis.get('convergence_timestep', 0)} timesteps"],
                ['Initial Inversions', str(sorting_analysis.get('initial_inversions', 0))],
                ['Final Inversions', str(sorting_analysis.get('final_inversions', 0))]
            ]

            metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
            ]))

            self.story.append(metrics_table)

    async def _add_section(
        self,
        section: ReportSection,
        figures: Dict[str, plt.Figure],
        tables: Dict[str, pd.DataFrame]
    ):
        """Add a section to the PDF."""
        # Section title
        self.story.append(Paragraph(section.title, self.styles['SectionTitle']))
        self.toc.addEntry(0, section.title, self.doc.page)

        # Add summary metrics for summary sections
        if section.section_type == SectionType.SUMMARY and section.data:
            self._add_summary_metrics(section.data)

        # Section content
        if section.content:
            content_paras = self._convert_markdown_to_paragraphs(section.content)
            for para in content_paras:
                self.story.append(para)

        # Add visualizations
        if section.visualizations:
            await self._add_visualizations(section.visualizations, figures)

        # Add tables
        if section.data and 'tables' in section.data:
            self._add_data_tables(section.data['tables'])

        # Add space between sections
        self.story.append(Spacer(1, 20))

    def _add_summary_metrics(self, section_data: Dict[str, Any]):
        """Add summary metrics display."""
        sorting_data = section_data.get('sorting_analysis', {})
        performance_data = section_data.get('performance_analysis', {})

        if not sorting_data and not performance_data:
            return

        # Create metrics grid
        metrics = []
        labels = []

        if sorting_data:
            improvement_rate = sorting_data.get('improvement_rate', 0)
            convergence_time = sorting_data.get('convergence_timestep', 0)

            metrics.extend([f"{improvement_rate:.1%}", str(convergence_time)])
            labels.extend(["Improvement Rate", "Convergence Time"])

        if performance_data and len(metrics) < 4:
            avg_metrics = performance_data.get('avg_metrics', {})
            for metric, value in list(avg_metrics.items())[:2]:
                if isinstance(value, (int, float)) and len(metrics) < 4:
                    metrics.append(f"{value:.3f}")
                    labels.append(metric.replace('_', ' ').title())

        # Create metrics table
        if metrics:
            metrics_rows = []
            for i in range(0, len(metrics), 2):
                row = []
                for j in range(2):
                    if i + j < len(metrics):
                        row.extend([metrics[i + j], labels[i + j]])
                    else:
                        row.extend(['', ''])
                metrics_rows.append(row)

            metrics_table = Table(metrics_rows, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (0, -1), 16),
                ('FONTSIZE', (2, 0), (2, -1), 16),
                ('FONTSIZE', (1, 0), (1, -1), 10),
                ('FONTSIZE', (3, 0), (3, -1), 10),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#27ae60')),
                ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#27ae60')),
                ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#7f8c8d')),
                ('TEXTCOLOR', (3, 0), (3, -1), colors.HexColor('#7f8c8d')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
                ('LINEBELOW', (0, 0), (1, -1), 1, colors.HexColor('#bdc3c7')),
                ('LINEBELOW', (2, 0), (3, -1), 1, colors.HexColor('#bdc3c7')),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa'))
            ]))

            self.story.append(metrics_table)
            self.story.append(Spacer(1, 20))

    async def _add_visualizations(self, viz_names: List[str], figures: Dict[str, plt.Figure]):
        """Add visualizations to PDF."""
        for viz_name in viz_names:
            if viz_name in figures:
                # Save figure to temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    figures[viz_name].savefig(
                        tmp_file.name,
                        dpi=300,
                        bbox_inches='tight',
                        facecolor='white'
                    )

                    # Add image to PDF
                    img = Image(tmp_file.name, width=6*inch, height=4*inch)
                    self.story.append(img)

                    # Add caption
                    caption = self._get_figure_caption(viz_name)
                    self.story.append(Paragraph(caption, self.styles['Caption']))
                    self.story.append(Spacer(1, 20))

                # Clean up temporary file
                Path(tmp_file.name).unlink(missing_ok=True)

    def _add_data_tables(self, tables_dict: Dict[str, pd.DataFrame]):
        """Add data tables to PDF."""
        for table_name, df in tables_dict.items():
            if df.empty:
                continue

            # Add table title
            self.story.append(Paragraph(
                table_name.replace('_', ' ').title(),
                self.styles['SubsectionTitle']
            ))

            # Limit table size for PDF
            display_df = df.head(20)  # Show first 20 rows

            # Prepare table data
            table_data = [display_df.columns.tolist()]
            for _, row in display_df.iterrows():
                formatted_row = []
                for value in row:
                    if isinstance(value, float):
                        formatted_row.append(f"{value:.3f}")
                    else:
                        formatted_row.append(str(value))
                table_data.append(formatted_row)

            # Calculate column widths
            col_count = len(table_data[0])
            col_width = (7 * inch) / col_count  # Distribute across page width

            # Create table
            table = Table(table_data, colWidths=[col_width] * col_count)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
            ]))

            self.story.append(table)

            # Add note if table was truncated
            if len(df) > 20:
                self.story.append(Paragraph(
                    f"<i>Table truncated. Showing first 20 of {len(df)} rows.</i>",
                    self.styles['Caption']
                ))

            self.story.append(Spacer(1, 20))

    def _add_technical_appendix(self, report_data: Dict[str, Any]):
        """Add technical appendix with detailed information."""
        self.story.append(PageBreak())
        self.story.append(Paragraph("Technical Appendix", self.styles['SectionTitle']))

        # System information
        self.story.append(Paragraph("System Information", self.styles['SubsectionTitle']))
        system_info = f"""
        <b>Report Generation Details:</b><br/>
        • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        • Format: PDF Report<br/>
        • Version: 1.0<br/>
        • Framework: Enhanced Morphogenesis Simulation<br/><br/>

        <b>Simulation Configuration:</b><br/>
        • Total Cells: {report_data.get('cell_count', 'N/A'):,}<br/>
        • Timesteps: {report_data.get('timestep_count', 'N/A'):,}<br/>
        • Data Points: {report_data.get('cell_count', 0) * report_data.get('timestep_count', 0):,}<br/>
        """

        self.story.append(Paragraph(system_info, self.styles['Normal']))

        # Methodology
        self.story.append(Paragraph("Methodology", self.styles['SubsectionTitle']))
        methodology_text = """
        This report analyzes morphogenesis simulation data using statistical methods
        and visualizations to understand emergent cell behavior patterns. The analysis
        includes:

        • Descriptive statistics of cell positions and sorting values
        • Performance metrics tracking over time
        • Statistical significance testing of behavioral changes
        • Trajectory analysis of individual cell movements
        • Comparative analysis of different behavioral strategies

        All statistical analyses use standard methods with appropriate confidence
        intervals and significance testing where applicable.
        """

        self.story.append(Paragraph(methodology_text, self.styles['Normal']))

    def _convert_markdown_to_paragraphs(self, markdown_text: str) -> List[Paragraph]:
        """Convert simple markdown to ReportLab paragraphs."""
        if not markdown_text:
            return []

        paragraphs = []
        lines = markdown_text.split('\n')
        current_list = []
        in_list = False

        for line in lines:
            line = line.strip()

            if not line:
                if in_list:
                    # End list
                    list_text = "<br/>".join(f"• {item[2:]}" for item in current_list)
                    paragraphs.append(Paragraph(list_text, self.styles['Normal']))
                    current_list = []
                    in_list = False
                continue

            if line.startswith('### '):
                if in_list:
                    list_text = "<br/>".join(f"• {item[2:]}" for item in current_list)
                    paragraphs.append(Paragraph(list_text, self.styles['Normal']))
                    current_list = []
                    in_list = False
                paragraphs.append(Paragraph(line[4:], self.styles['SubsectionTitle']))

            elif line.startswith('## '):
                if in_list:
                    list_text = "<br/>".join(f"• {item[2:]}" for item in current_list)
                    paragraphs.append(Paragraph(list_text, self.styles['Normal']))
                    current_list = []
                    in_list = False
                paragraphs.append(Paragraph(line[3:], self.styles['SubsectionTitle']))

            elif line.startswith('- '):
                current_list.append(line)
                in_list = True

            else:
                if in_list:
                    list_text = "<br/>".join(f"• {item[2:]}" for item in current_list)
                    paragraphs.append(Paragraph(list_text, self.styles['Normal']))
                    current_list = []
                    in_list = False

                # Process inline formatting
                formatted_line = line.replace('**', '<b>', 1).replace('**', '</b>', 1)
                formatted_line = formatted_line.replace('*', '<i>', 1).replace('*', '</i>', 1)

                paragraphs.append(Paragraph(formatted_line, self.styles['Normal']))

        # Handle remaining list
        if in_list:
            list_text = "<br/>".join(f"• {item[2:]}" for item in current_list)
            paragraphs.append(Paragraph(list_text, self.styles['Normal']))

        return paragraphs

    def _get_figure_caption(self, viz_name: str) -> str:
        """Get caption for a figure."""
        captions = {
            'sorting_progress': 'Figure: Evolution of sorting progress showing inversions and completion percentage over time.',
            'cell_trajectories': 'Figure: Sample cell trajectories showing movement patterns during simulation.',
            'performance_metrics': 'Figure: Key performance metrics tracked throughout the simulation.',
            'state_distribution': 'Figure: Distribution of cell states over the course of the simulation.',
            'efficiency_comparison': 'Figure: Comparison of sorting efficiency metrics and algorithm performance.'
        }
        return captions.get(viz_name, f'Figure: {viz_name.replace("_", " ").title()}')

    def _first_page_layout(self, canvas, doc):
        """Layout for first page (title page)."""
        canvas.saveState()

        # Add header line
        canvas.setStrokeColor(colors.HexColor('#3498db'))
        canvas.setLineWidth(3)
        canvas.line(doc.leftMargin, doc.height + doc.topMargin - 20,
                   doc.width + doc.leftMargin, doc.height + doc.topMargin - 20)

        # Add footer
        footer_text = f"Morphogenesis Simulation Report • Generated {datetime.now().strftime('%Y-%m-%d')}"
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#7f8c8d'))
        canvas.drawCentredText(doc.width/2 + doc.leftMargin, doc.bottomMargin/2, footer_text)

        canvas.restoreState()

    def _later_page_layout(self, canvas, doc):
        """Layout for subsequent pages."""
        canvas.saveState()

        # Header
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(colors.HexColor('#2c3e50'))
        canvas.drawString(doc.leftMargin, doc.height + doc.topMargin - 20, self.config.title[:50])

        # Page number
        canvas.setFont('Helvetica', 10)
        canvas.drawRightString(doc.width + doc.leftMargin, doc.height + doc.topMargin - 20,
                             f"Page {canvas.getPageNumber()}")

        # Header line
        canvas.setStrokeColor(colors.HexColor('#3498db'))
        canvas.setLineWidth(1)
        canvas.line(doc.leftMargin, doc.height + doc.topMargin - 30,
                   doc.width + doc.leftMargin, doc.height + doc.topMargin - 30)

        # Footer
        footer_text = f"Morphogenesis Framework • {datetime.now().strftime('%Y-%m-%d')}"
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#7f8c8d'))
        canvas.drawCentredText(doc.width/2 + doc.leftMargin, doc.bottomMargin/2, footer_text)

        canvas.restoreState()