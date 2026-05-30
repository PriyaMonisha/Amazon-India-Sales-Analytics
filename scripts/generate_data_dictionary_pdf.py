"""
Generate a production-grade Data Dictionary PDF for the Amazon India Sales Analytics project.
Output: artifacts/reports/Data_Dictionary_Amazon_India_Sales_Analytics.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── Brand colours ──────────────────────────────────────────────────────────────
AMAZON_ORANGE = colors.HexColor("#FF9900")
AMAZON_DARK   = colors.HexColor("#232F3E")
AMAZON_LIGHT  = colors.HexColor("#F5F5F5")
ACCENT_BLUE   = colors.HexColor("#1565C0")
TEXT_DARK     = colors.HexColor("#212121")
TEXT_GREY     = colors.HexColor("#616161")
WHITE         = colors.white

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "Data_Dictionary_Amazon_India_Sales_Analytics.pdf"

# ── Styles ─────────────────────────────────────────────────────────────────────
base_styles = getSampleStyleSheet()

def _style(name, parent="Normal", **kwargs):
    s = ParagraphStyle(name, parent=base_styles[parent], **kwargs)
    return s

S_TITLE       = _style("DocTitle",    fontSize=28, textColor=WHITE,       alignment=TA_CENTER, spaceAfter=6, fontName="Helvetica-Bold")
S_SUBTITLE    = _style("DocSub",      fontSize=13, textColor=AMAZON_ORANGE, alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica")
S_COVER_META  = _style("CoverMeta",  fontSize=10, textColor=colors.HexColor("#BDBDBD"), alignment=TA_CENTER, fontName="Helvetica")
S_H1          = _style("H1",          fontSize=16, textColor=AMAZON_DARK,  spaceAfter=6, spaceBefore=18, fontName="Helvetica-Bold")
S_H2          = _style("H2",          fontSize=12, textColor=ACCENT_BLUE,  spaceAfter=4, spaceBefore=12, fontName="Helvetica-Bold")
S_BODY        = _style("Body",        fontSize=9,  textColor=TEXT_DARK,    spaceAfter=4, leading=13, fontName="Helvetica")
S_NOTE        = _style("Note",        fontSize=8,  textColor=TEXT_GREY,    spaceAfter=4, leading=11, fontName="Helvetica-Oblique")
S_TOC_ITEM    = _style("TOCItem",     fontSize=10, textColor=TEXT_DARK,    spaceAfter=3, fontName="Helvetica")
S_TOC_PAGE    = _style("TOCPage",     fontSize=10, textColor=TEXT_GREY,    alignment=TA_RIGHT, fontName="Helvetica")

# ── Header / Footer ────────────────────────────────────────────────────────────
def _on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    # Header bar
    canvas.setFillColor(AMAZON_DARK)
    canvas.rect(0, h - 1.2*cm, w, 1.2*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(WHITE)
    canvas.drawString(1.5*cm, h - 0.8*cm, "Amazon India Sales Analytics")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(AMAZON_ORANGE)
    canvas.drawRightString(w - 1.5*cm, h - 0.8*cm, "Data Dictionary  |  Confidential")
    # Footer
    canvas.setFillColor(AMAZON_LIGHT)
    canvas.rect(0, 0, w, 0.9*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(TEXT_GREY)
    canvas.drawString(1.5*cm, 0.3*cm, f"Generated {datetime.now().strftime('%d %B %Y')}")
    canvas.drawCentredString(w/2, 0.3*cm, "INTERNAL USE — NOT FOR DISTRIBUTION")
    canvas.setFillColor(AMAZON_ORANGE)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawRightString(w - 1.5*cm, 0.3*cm, f"Page {doc.page}")
    canvas.restoreState()

def _on_cover(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(AMAZON_DARK)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(AMAZON_ORANGE)
    canvas.rect(0, h * 0.35, w, 3, fill=1, stroke=0)
    canvas.rect(0, h * 0.35 - 6, w, 3, fill=1, stroke=0)
    canvas.restoreState()

# ── Table helpers ──────────────────────────────────────────────────────────────
_HDR_STYLE = TableStyle([
    ("BACKGROUND",   (0,0), (-1,0),  AMAZON_DARK),
    ("TEXTCOLOR",    (0,0), (-1,0),  WHITE),
    ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,0),  8),
    ("ALIGN",        (0,0), (-1,0),  "LEFT"),
    ("TOPPADDING",   (0,0), (-1,0),  6),
    ("BOTTOMPADDING",(0,0), (-1,0),  6),
    ("LEFTPADDING",  (0,0), (-1,-1), 6),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, AMAZON_LIGHT]),
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
    ("FONTSIZE",     (0,1), (-1,-1), 8),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING",   (0,1), (-1,-1), 4),
    ("BOTTOMPADDING",(0,1), (-1,-1), 4),
    ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#E0E0E0")),
    ("BOX",          (0,0), (-1,-1), 0.6, AMAZON_DARK),
])

def _cell(text, bold=False, color=TEXT_DARK, size=8):
    font = "Helvetica-Bold" if bold else "Helvetica"
    return Paragraph(f'<font name="{font}" size="{size}" color="{color.hexval() if hasattr(color,"hexval") else color}">{text}</font>', S_BODY)


def build():
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=2.0*cm, bottomMargin=1.8*cm,
    )
    story = []

    # ── COVER PAGE ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 5*cm))
    story.append(Paragraph("DATA DICTIONARY", S_TITLE))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Amazon India: A Decade of Sales Analytics", S_SUBTITLE))
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("11 Years of E-Commerce Data  ·  1.1 Million Transactions  ·  PostgreSQL Star Schema", S_COVER_META))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Version 1.0  ·  {datetime.now().strftime('%B %Y')}  ·  GUVI Data Science Programme", S_COVER_META))
    story.append(PageBreak())

    # ── TABLE OF CONTENTS ──────────────────────────────────────────────────────
    doc_template = SimpleDocTemplate.__new__(SimpleDocTemplate)  # placeholder for first page
    story.append(Paragraph("Table of Contents", S_H1))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceAfter=10))
    toc = [
        ("1. Project Overview",                      "3"),
        ("2. Dataset 1 — Sales Transactions",        "3"),
        ("   2.1  Column Reference (36 columns)",    "3"),
        ("   2.2  Data Quality Issues (10 Fixes)",   "6"),
        ("3. Dataset 2 — Product Catalog",           "7"),
        ("   3.1  Column Reference (11 columns)",    "7"),
        ("4. PostgreSQL Star Schema",                 "8"),
        ("   4.1  fact_transactions",                "8"),
        ("   4.2  dim_customers",                    "9"),
        ("   4.3  dim_products",                     "9"),
        ("   4.4  dim_time",                         "9"),
        ("5. Key Business Metrics",                  "10"),
    ]
    for label, page in toc:
        row = Table([[Paragraph(label, S_TOC_ITEM), Paragraph(page, S_TOC_PAGE)]],
                    colWidths=["85%", "15%"])
        row.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("BOTTOMPADDING",(0,0),(-1,-1),2)]))
        story.append(row)
    story.append(PageBreak())

    # ── SECTION 1 — PROJECT OVERVIEW ──────────────────────────────────────────
    story.append(Paragraph("1. Project Overview", S_H1))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceAfter=8))
    overview = [
        ["Attribute", "Detail"],
        ["Project Name",   "Amazon India: A Decade of Sales Analytics"],
        ["Data Period",    "2015 – 2025 (11 years)"],
        ["Raw Transactions","1,127,609 rows across 11 annual CSV files"],
        ["After Cleaning", "1,122,000 rows (99.5% retention)"],
        ["Products",       "2,004 unique SKUs across 6 subcategories"],
        ["Database",       "PostgreSQL 15+ — Star schema with 4 tables"],
        ["Pipeline",       "Python 3.11 · SQLAlchemy · Pandera validation"],
        ["Subcategories",  "Smartphones, Laptops, Tablets, Smart Watch, Audio, TV & Entertainment"],
        ["Geographic Scope","30+ Indian cities — Metro / Tier1 / Tier2 / Rural"],
    ]
    t = Table(overview, colWidths=["35%","65%"])
    t.setStyle(_HDR_STYLE)
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # ── SECTION 2 — SALES TRANSACTIONS ────────────────────────────────────────
    story.append(Paragraph("2. Dataset 1 — Sales Transactions", S_H1))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceAfter=4))
    story.append(Paragraph("Source: <b>data/raw/amazon_india_{year}.csv</b> (11 files, 2015–2025) → cleaned to <b>data/processed/cleaned_df_sales.csv</b>", S_BODY))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("2.1  Column Reference", S_H2))

    tx_cols = [
        ["Column", "Data Type", "Description", "Example Value"],
        # Identifiers
        ["transaction_id",      "VARCHAR",  "Unique transaction identifier",                               "TXN_2019_000001"],
        ["order_date",          "DATE",     "Order date — standardised to YYYY-MM-DD",                   "2019-10-15"],
        ["order_month",         "SMALLINT", "Month number extracted from order_date",                    "10"],
        ["order_year",          "SMALLINT", "Year extracted from order_date",                            "2019"],
        ["order_quarter",       "SMALLINT", "Quarter (1–4)",                                             "4"],
        # Customer
        ["customer_id",         "VARCHAR",  "Unique customer identifier",                                "CUST_000123"],
        ["customer_city",       "VARCHAR",  "Standardised city name (e.g. Bangalore→Bengaluru)",        "Bengaluru"],
        ["customer_state",      "VARCHAR",  "Indian state",                                              "Karnataka"],
        ["customer_tier",       "VARCHAR",  "City tier: Metro / Tier1 / Tier2 / Rural",                 "Metro"],
        ["customer_spending_tier","VARCHAR","Spending segment: High / Medium / Low",                    "High"],
        ["customer_age_group",  "VARCHAR",  "Age bracket: 18-25 / 26-35 / 36-45 / 46-55 / 55+",        "26-35"],
        ["is_prime_member",     "BOOLEAN",  "Prime membership flag (cleaned from Yes/No/1/0)",           "True"],
        # Product
        ["product_id",          "VARCHAR",  "Unique product identifier",                                 "PROD_000456"],
        ["product_name",        "VARCHAR",  "Product display name",                                      "Samsung Galaxy S21"],
        ["category",            "VARCHAR",  "Top-level category — Electronics for all rows",             "Electronics"],
        ["subcategory",         "VARCHAR",  "Product subcategory (6 values in dataset)",                 "Smartphones"],
        ["brand",               "VARCHAR",  "Brand name (100+ brands)",                                  "Samsung"],
        ["is_prime_eligible",   "BOOLEAN",  "Prime delivery eligible flag",                              "True"],
        ["product_weight_kg",   "NUMERIC",  "Product weight in kilograms",                               "0.18"],
        ["product_rating",      "NUMERIC",  "Product catalogue rating 1.0–5.0",                         "4.2"],
        # Pricing
        ["original_price_inr",  "NUMERIC",  "List price before discount — ₹ symbols and commas removed","75000.00"],
        ["discount_percent",    "NUMERIC",  "Discount percentage applied (0–100)",                      "15.0"],
        ["discounted_price_inr","NUMERIC",  "Price after discount applied",                              "63750.00"],
        ["quantity",            "SMALLINT", "Units ordered (typically 1–3)",                             "1"],
        ["subtotal_inr",        "NUMERIC",  "discounted_price_inr × quantity",                          "63750.00"],
        ["delivery_charges",    "NUMERIC",  "Delivery fee in INR (0 for Prime-eligible orders)",        "200.00"],
        ["final_amount_inr",    "NUMERIC",  "Total amount paid (subtotal + delivery charges)",           "63950.00"],
        # Transaction
        ["payment_method",      "VARCHAR",  "Standardised payment method (UPI/COD/Credit Card etc.)",   "UPI"],
        ["payment_category",    "VARCHAR",  "Grouped type: Digital / Card / Cash",                      "Digital"],
        ["delivery_days",       "NUMERIC",  "Days from order to delivery — cleaned from text values",   "3.0"],
        ["delivery_type",       "VARCHAR",  "Speed tier: Same Day / Express / Standard",                "Standard"],
        ["is_festival_sale",    "BOOLEAN",  "Festival period flag (cleaned from True/False/1/0)",       "False"],
        ["festival_name",       "VARCHAR",  "Festival name if is_festival_sale=True, else NULL",        "Diwali"],
        ["sale_type",           "VARCHAR",  "Sale classification: Festival / Regular",                  "Regular"],
        ["return_status",       "VARCHAR",  "Order outcome: Delivered / Returned / Cancelled",          "Delivered"],
        ["customer_rating",     "NUMERIC",  "Post-purchase customer rating 1.0–5.0",                    "4.0"],
        ["rating_missing",      "SMALLINT", "1 if customer_rating was imputed; 0 if original",         "0"],
    ]
    t2 = Table(tx_cols, colWidths=["28%","13%","42%","17%"])
    t2.setStyle(_HDR_STYLE)
    story.append(t2)
    story.append(Spacer(1, 0.5*cm))

    # ── Data Quality ──────────────────────────────────────────────────────────
    story.append(Paragraph("2.2  Data Quality Issues Fixed (10 Cleaning Challenges)", S_H2))
    dq_data = [
        ["#", "Issue", "Method Applied", "Rows Affected"],
        ["1", "Mixed date formats (DD/MM/YYYY, DD-MM-YY, YYYY-MM-DD, invalid)", "Multi-format parse; invalid → sentinel 1900-01-01", "~4,757"],
        ["2", "Price with ₹ symbols, Indian commas, 'Price on Request' text",   "Regex strip; non-numeric → NaN → median imputation", "~2,898"],
        ["3", "Ratings in multiple formats (4 stars, 3/5, 2.5/5.0)",             "Regex extraction; scaled to 1.0–5.0",              "~15,336"],
        ["4", "Inconsistent city names (Bangalore/Bengaluru, spelling errors)",   "Canonical mapping dictionary + lowercase normalisation","~222"],
        ["5", "Boolean columns with True/False/Yes/No/1/0/Y/N",                 "Explicit mapping to Python bool",                   "All boolean cols"],
        ["6", "Category name variations (Electronics/ELECTRONICS/Electronic)",   "Lowercase + canonical map",                        "~0 after standardise"],
        ["7", "Delivery days: negatives, 'Same Day', '1-2 days', >30 outliers",  "Text parse; negatives → 0; IQR cap per tier",       "~110"],
        ["8", "Duplicate transactions (same customer+product+date+amount)",       "Keep first; bulk orders identified by quantity >1", "~244"],
        ["9", "Price outliers (100× due to decimal point data entry errors)",    "IQR per subcategory; corrected decimal shift",      "~1,457"],
        ["10","Payment methods inconsistent (UPI/PhonePe/GooglePay → UPI)",       "Canonical mapping to 7 standard categories",       "All payment rows"],
    ]
    t3 = Table(dq_data, colWidths=["4%","35%","40%","21%"])
    t3.setStyle(_HDR_STYLE)
    story.append(t3)
    story.append(PageBreak())

    # ── SECTION 3 — PRODUCT CATALOG ───────────────────────────────────────────
    story.append(Paragraph("3. Dataset 2 — Product Catalog", S_H1))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceAfter=4))
    story.append(Paragraph("Source: <b>data/raw/amazon_india_products_catalog.csv</b> → <b>data/processed/cleaned_products.csv</b>", S_BODY))
    story.append(Paragraph("2,004 rows × 11 columns. Joined to fact_transactions on product_id.", S_NOTE))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("3.1  Column Reference", S_H2))
    prod_cols = [
        ["Column",              "Data Type", "Description",                                            "Example"],
        ["product_id",          "VARCHAR PK","Unique product identifier — joins to fact_transactions","PROD_000456"],
        ["product_name",        "VARCHAR",   "Full product display name",                              "Samsung Galaxy S21"],
        ["category",            "VARCHAR",   "Top-level category (Electronics for all rows)",          "Electronics"],
        ["subcategory",         "VARCHAR",   "Subcategory (6 distinct values)",                        "Smartphones"],
        ["brand",               "VARCHAR",   "Brand name (100+ brands in dataset)",                    "Samsung"],
        ["base_price_2015",     "NUMERIC",   "Original launch price in 2015 (INR)",                   "45000.00"],
        ["weight_kg",           "NUMERIC",   "Product weight in kilograms",                            "0.18"],
        ["rating",              "NUMERIC",   "Catalogue aggregate rating (avg across all reviews)",   "4.3"],
        ["is_prime_eligible",   "BOOLEAN",   "Whether product qualifies for Prime free delivery",      "True"],
        ["launch_year",         "SMALLINT",  "Year product was first listed on Amazon India",          "2021"],
        ["model",               "VARCHAR",   "Model / variant identifier",                             "SM-G991B"],
    ]
    t4 = Table(prod_cols, colWidths=["22%","13%","45%","20%"])
    t4.setStyle(_HDR_STYLE)
    story.append(t4)
    story.append(PageBreak())

    # ── SECTION 4 — STAR SCHEMA ───────────────────────────────────────────────
    story.append(Paragraph("4. PostgreSQL Star Schema", S_H1))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceAfter=6))
    story.append(Paragraph(
        "After ETL, data is loaded into a PostgreSQL star schema with 4 tables. "
        "The schema is defined in <b>scripts/init_db.sql</b> and created automatically on first run.",
        S_BODY))
    story.append(Spacer(1, 0.3*cm))

    schema_intro = Table(
        [["Table", "Type", "Rows (full dataset)", "Purpose"]],
        colWidths=["30%","15%","25%","30%"]
    )
    schema_intro.setStyle(_HDR_STYLE)
    schema_rows = [
        ["fact_transactions", "Fact",      "~1,122,000", "One row per transaction — all measures"],
        ["dim_customers",     "Dimension", "~354,000",   "Customer master — geography, tier, Prime"],
        ["dim_products",      "Dimension", "2,004",      "Product master — category, brand, price"],
        ["dim_time",          "Dimension", "~4,000",     "Date dimension — year, month, quarter, festival flag"],
    ]
    full_schema = Table(
        [["Table", "Type", "Rows (full dataset)", "Purpose"]] + schema_rows,
        colWidths=["30%","15%","25%","30%"]
    )
    full_schema.setStyle(_HDR_STYLE)
    story.append(full_schema)
    story.append(Spacer(1, 0.4*cm))

    def _schema_table(title, note, rows):
        story.append(Paragraph(title, S_H2))
        if note:
            story.append(Paragraph(note, S_NOTE))
        t = Table([["Column", "Type", "Constraint / Notes"]] + rows,
                  colWidths=["38%","17%","45%"])
        t.setStyle(_HDR_STYLE)
        story.append(t)
        story.append(Spacer(1, 0.4*cm))

    _schema_table("4.1  fact_transactions",
        "Central fact table. Foreign keys to all 3 dimension tables. Upserted on transaction_id.",
        [
            ["transaction_id",       "VARCHAR",  "PRIMARY KEY"],
            ["order_date",           "DATE",     "Indexed — primary filter column"],
            ["order_year",           "SMALLINT", "Indexed — partition-style filtering"],
            ["order_month",          "SMALLINT", ""],
            ["order_quarter",        "SMALLINT", ""],
            ["customer_id",          "VARCHAR",  "FK → dim_customers.customer_id"],
            ["product_id",           "VARCHAR",  "FK → dim_products.product_id"],
            ["original_price_inr",   "NUMERIC",  "NULL allowed — some products have no list price"],
            ["discount_percent",     "NUMERIC",  "0–100"],
            ["discounted_price_inr", "NUMERIC",  ""],
            ["final_amount_inr",     "NUMERIC",  "Indexed — used in all revenue aggregations"],
            ["quantity",             "SMALLINT", "Typically 1–3"],
            ["payment_method",       "VARCHAR",  ""],
            ["delivery_days",        "NUMERIC",  "NULL if not yet delivered"],
            ["delivery_type",        "VARCHAR",  "Same Day / Express / Standard"],
            ["is_festival_sale",     "BOOLEAN",  ""],
            ["festival_name",        "VARCHAR",  "NULL when is_festival_sale=False"],
            ["product_rating",       "NUMERIC",  "1.0–5.0"],
            ["customer_rating",      "NUMERIC",  "1.0–5.0; NULL if not rated"],
            ["return_status",        "VARCHAR",  "Delivered / Returned / Cancelled"],
        ]
    )

    _schema_table("4.2  dim_customers",
        "One row per unique customer. Populated from fact_transactions.customer_id.",
        [
            ["customer_id",           "VARCHAR",  "PRIMARY KEY"],
            ["customer_city",         "VARCHAR",  "Standardised city name"],
            ["customer_state",        "VARCHAR",  "Indian state"],
            ["customer_tier",         "VARCHAR",  "Metro / Tier1 / Tier2 / Rural"],
            ["customer_spending_tier","VARCHAR",  "High / Medium / Low"],
            ["customer_age_group",    "VARCHAR",  "18-25 / 26-35 / 36-45 / 46-55 / 55+"],
            ["is_prime_member",       "BOOLEAN",  ""],
        ]
    )

    _schema_table("4.3  dim_products",
        "One row per unique product. Sourced from product catalogue.",
        [
            ["product_id",       "VARCHAR",  "PRIMARY KEY"],
            ["product_name",     "VARCHAR",  ""],
            ["category",         "VARCHAR",  "Electronics for all rows in this dataset"],
            ["subcategory",      "VARCHAR",  "6 distinct values — use this for analysis"],
            ["brand",            "VARCHAR",  ""],
            ["launch_year",      "SMALLINT", "Used for product lifecycle analysis"],
            ["is_prime_eligible","BOOLEAN",  ""],
        ]
    )

    _schema_table("4.4  dim_time",
        "Date dimension table for time-based joins. One row per calendar date.",
        [
            ["date",              "DATE",     "PRIMARY KEY"],
            ["year",              "SMALLINT", ""],
            ["month",             "SMALLINT", "1–12"],
            ["quarter",           "SMALLINT", "1–4"],
            ["is_festival_period","BOOLEAN",  "True for known festival windows (Diwali, Prime Day, etc.)"],
        ]
    )

    # ── SECTION 5 — KEY METRICS ───────────────────────────────────────────────
    story.append(Paragraph("5. Key Business Metrics", S_H1))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceAfter=8))
    story.append(Paragraph("Reference values from full 1.1M row dataset (confirmed post-ETL):", S_BODY))
    story.append(Spacer(1, 0.2*cm))
    kpi_data = [
        ["Metric",                           "Value"],
        ["Total raw transactions",           "1,127,609"],
        ["After cleaning (ETL output)",      "1,122,000"],
        ["Data retention rate",              "99.5%"],
        ["Unique customers",                 "~354,000"],
        ["Unique products (SKUs)",           "2,004"],
        ["Subcategories",                    "6 (Smartphones, Laptops, Tablets, Smart Watch, Audio, TV & Entertainment)"],
        ["Years covered",                    "2015–2025 (11 years)"],
        ["States covered",                   "15+ Indian states"],
        ["Cities covered",                   "30+ cities"],
        ["Smartphone revenue share",         "73.2% of total electronics revenue"],
        ["Peak revenue year",                "2020 (COVID-driven demand)"],
        ["Top state by revenue",             "Maharashtra"],
        ["Prime vs Non-Prime AOV",           "₹78,127 vs ₹62,296 (+25% for Prime)"],
        ["Avg Customer Lifetime Value",      "₹74,100 (mean) / ₹47,900 (median)"],
        ["Festival AOV vs Regular AOV",      "₹47,400 vs ₹77,700 (festival is LOWER — discount effect)"],
        ["Highest return rate subcategory",  "Audio (8.1%)"],
        ["Lowest return rate subcategory",   "TV & Entertainment (5.1%)"],
        ["Optimal discount range",           "20–30% (highest revenue generation)"],
        ["Avg delivery days (all tiers)",    "3.3–3.5 days"],
        ["Top 36% products",                 "Generate 80% of total revenue (Pareto principle)"],
        ["UPI payment share (2025)",         "Largest and growing — displaced COD since 2016 demonetisation"],
    ]
    t5 = Table(kpi_data, colWidths=["45%","55%"])
    t5.setStyle(_HDR_STYLE)
    story.append(t5)

    # Build
    doc.build(story, onFirstPage=_on_cover, onLaterPages=_on_page)
    print(f"[OK] Data Dictionary PDF saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
