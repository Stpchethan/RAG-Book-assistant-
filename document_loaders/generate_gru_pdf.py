from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime

# Create PDF
pdf_path = r"c:\Users\cheth\Downloads\RAG Proejct\document loaders\GRU.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                        rightMargin=72, leftMargin=72,
                        topMargin=72, bottomMargin=18)

# Container for the 'Flowable' objects
elements = []

# Define styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=28,
    textColor=colors.HexColor('#1f4788'),
    spaceAfter=30,
    alignment=1  # Center
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=16,
    textColor=colors.HexColor('#2e5c9a'),
    spaceAfter=12,
    spaceBefore=12
)

body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['BodyText'],
    fontSize=11,
    spaceAfter=12,
    alignment=4  # Justify
)

# Title
elements.append(Paragraph("Gated Recurrent Unit (GRU)", title_style))
elements.append(Spacer(1, 0.3*inch))

# Table of Contents
elements.append(Paragraph("Table of Contents", heading_style))
toc_items = [
    "1. Introduction",
    "2. GRU Architecture",
    "3. Mathematical Foundations",
    "4. GRU vs LSTM",
    "5. Applications",
    "6. Implementation",
    "7. Training Tips"
]
for item in toc_items:
    elements.append(Paragraph(item, body_style))
elements.append(Spacer(1, 0.2*inch))

elements.append(PageBreak())

# Section 1: Introduction
elements.append(Paragraph("1. Introduction", heading_style))
intro_text = """
A Gated Recurrent Unit (GRU) is a type of Recurrent Neural Network (RNN) designed to capture 
long-term dependencies in sequential data. GRUs address the vanishing gradient problem that occurs in 
standard RNNs by introducing gating mechanisms that control information flow. Unlike LSTM cells which 
have three gates, GRUs use only two gates (reset and update), making them more computationally efficient 
while maintaining similar performance.
"""
elements.append(Paragraph(intro_text, body_style))
elements.append(Spacer(1, 0.2*inch))

# Section 2: GRU Architecture
elements.append(Paragraph("2. GRU Architecture", heading_style))
arch_text = """
The GRU architecture consists of three main components:
"""
elements.append(Paragraph(arch_text, body_style))

# GRU components table
gru_components = [
    ['Component', 'Purpose', 'Function'],
    ['Reset Gate (r)', 'Controls how much past information to forget', 'r = σ(W_r·[h_{t-1}, x_t] + b_r)'],
    ['Update Gate (z)', 'Controls how much past and new info to combine', 'z = σ(W_z·[h_{t-1}, x_t] + b_z)'],
    ['Candidate Hidden State (h̃)', 'Candidate for new hidden state', 'h̃ = tanh(W·[r⊙h_{t-1}, x_t] + b)'],
]

t = Table(gru_components, colWidths=[1.5*inch, 2.5*inch, 2.5*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c9a')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
]))
elements.append(t)
elements.append(Spacer(1, 0.2*inch))

# Section 3: Mathematical Foundations
elements.append(Paragraph("3. Mathematical Foundations", heading_style))
math_text = """
<b>GRU Equations:</b><br/>
<br/>
The following equations define the GRU cell operations at each time step t:<br/>
<br/>
<b>Reset Gate:</b><br/>
r_t = σ(W_ir·x_t + U_ir·h_{t-1} + b_r)<br/>
<br/>
<b>Update Gate:</b><br/>
z_t = σ(W_iz·x_t + U_iz·h_{t-1} + b_z)<br/>
<br/>
<b>Candidate Hidden State:</b><br/>
h̃_t = tanh(W_ih·x_t + U_ih·(r_t ⊙ h_{t-1}) + b_h)<br/>
<br/>
<b>Output Hidden State:</b><br/>
h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}<br/>
<br/>
Where:<br/>
• σ denotes the sigmoid activation function<br/>
• tanh is the hyperbolic tangent activation<br/>
• ⊙ represents element-wise multiplication<br/>
• W and U are weight matrices<br/>
• b represents bias terms<br/>
• h_t is the hidden state at time t<br/>
• x_t is the input at time t
"""
elements.append(Paragraph(math_text, body_style))
elements.append(Spacer(1, 0.2*inch))

elements.append(PageBreak())

# Section 4: GRU vs LSTM
elements.append(Paragraph("4. GRU vs LSTM", heading_style))

comparison_data = [
    ['Aspect', 'GRU', 'LSTM'],
    ['Number of Gates', '2 (Reset, Update)', '3 (Input, Forget, Output)'],
    ['Complexity', 'Lower', 'Higher'],
    ['Parameters', 'Fewer', 'More'],
    ['Training Speed', 'Faster', 'Slower'],
    ['Performance', 'Comparable', 'Slightly Better on Large Datasets'],
    ['Memory Usage', 'Lower', 'Higher'],
    ['Use Case', 'Smaller datasets, real-time', 'Large datasets, complex tasks'],
]

t2 = Table(comparison_data, colWidths=[1.8*inch, 2.1*inch, 2.1*inch])
t2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c9a')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
]))
elements.append(t2)
elements.append(Spacer(1, 0.2*inch))

# Section 5: Applications
elements.append(Paragraph("5. Applications", heading_style))
app_text = """
GRUs are widely used in various machine learning applications:<br/>
<br/>
<b>• Machine Translation:</b> Converting text from one language to another<br/>
<b>• Speech Recognition:</b> Converting audio to text<br/>
<b>• Time Series Forecasting:</b> Predicting future values in sequential data<br/>
<b>• Sentiment Analysis:</b> Determining sentiment from text<br/>
<b>• Video Analysis:</b> Understanding sequences of video frames<br/>
<b>• Document Generation:</b> Creating coherent text sequences<br/>
<b>• Weather Prediction:</b> Forecasting meteorological patterns<br/>
<b>• Stock Price Prediction:</b> Analyzing financial time series
"""
elements.append(Paragraph(app_text, body_style))
elements.append(Spacer(1, 0.2*inch))

elements.append(PageBreak())

# Section 6: Implementation
elements.append(Paragraph("6. Implementation", heading_style))
impl_text = """
<b>Using PyTorch:</b><br/>
<br/>
import torch<br/>
import torch.nn as nn<br/>
<br/>
# Create a GRU layer<br/>
gru = nn.GRU(input_size=50, hidden_size=100, num_layers=2, batch_first=True)<br/>
<br/>
# Forward pass<br/>
x = torch.randn(32, 10, 50)  # (batch, seq_len, input_size)<br/>
output, hidden = gru(x)<br/>
<br/>
<b>Key Parameters:</b><br/>
• input_size: Size of input features<br/>
• hidden_size: Dimension of hidden state<br/>
• num_layers: Number of stacked GRU layers<br/>
• batch_first: If True, input is (batch, seq, features)<br/>
• dropout: Dropout applied to outputs between layers<br/>
• bidirectional: Create bidirectional GRU if True
"""
elements.append(Paragraph(impl_text, body_style))
elements.append(Spacer(1, 0.2*inch))

# Section 7: Training Tips
elements.append(Paragraph("7. Training Tips", heading_style))
tips_text = """
<b>Best Practices for Training GRU Networks:</b><br/>
<br/>
<b>1. Initialization:</b> Use proper weight initialization (Xavier/He) to avoid exploding/vanishing gradients<br/>
<br/>
<b>2. Learning Rate:</b> Start with smaller learning rates (0.001 to 0.0001) and adjust based on convergence<br/>
<br/>
<b>3. Gradient Clipping:</b> Clip gradients to prevent explosion during training<br/>
<br/>
<b>4. Batch Normalization:</b> Apply batch normalization between layers for stability<br/>
<br/>
<b>5. Sequence Padding:</b> Use padding for variable-length sequences in batches<br/>
<br/>
<b>6. Bidirectional Processing:</b> Use bidirectional GRUs when full sequence context is available<br/>
<br/>
<b>7. Regularization:</b> Apply dropout and L2 regularization to prevent overfitting<br/>
<br/>
<b>8. Monitoring:</b> Track training/validation loss to detect overfitting early<br/>
<br/>
<b>9. Early Stopping:</b> Stop training when validation loss plateaus<br/>
<br/>
<b>10. Architecture Tuning:</b> Experiment with different hidden sizes and number of layers
"""
elements.append(Paragraph(tips_text, body_style))

# Footer
elements.append(Spacer(1, 0.3*inch))
footer_text = f"""
<b>Document Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
<b>Version:</b> 1.0<br/>
<b>Subject:</b> Gated Recurrent Unit (GRU) - Comprehensive Guide
"""
elements.append(Paragraph(footer_text, body_style))

# Build PDF
doc.build(elements)
print(f"PDF generated successfully at: {pdf_path}")
