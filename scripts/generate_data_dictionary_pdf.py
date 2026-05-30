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
    PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT

# ── Page metrics ───────────────────────────────────────────────────────────────
PAGE_W = A4[0]           # 595 pts
L_MARGIN = R_MARGIN = 1.8 * cm
USABLE_W = PAGE_W - L_MARGIN - R_MARGIN   # ~17.4 cm

# ── Brand colours ──────────────────────────────────────────────────────────────
AMAZON_ORANGE = colors.HexColor("#FF9900")
AMAZON_DARK   = colors.HexColor("#232F3E")
AMAZON_LIGHT  = colors.HexColor("#F5F5F5")
ACCENT_BLUE   = colors.HexColor("#1565C0")
TEXT_DARK     = colors.HexColor("#212121")
TEXT_GREY     = colors.HexColor("#616161")
WHITE         = colors.white

OUTPUT_DIR  = PROJECT_ROOT / "artifacts" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "Data_Dictionary_Amazon_India_Sales_Analytics.pdf"

# ── Styles ─────────────────────────────────────────────────────────────────────
_BASE = getSampleStyleSheet()

def _sty(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=_BASE[parent], **kw)

S_TITLE      = _sty("DocTitle",  fontSize=28, textColor=WHITE,        alignment=TA_CENTER, spaceAfter=6,  fontName="Helvetica-Bold")
S_SUBTITLE   = _sty("DocSub",   fontSize=13, textColor=AMAZON_ORANGE, alignment=TA_CENTER, spaceAfter=4,  fontName="Helvetica")
S_COVER_META = _sty("CMeta",    fontSize=10, textColor=colors.HexColor("#BDBDBD"), alignment=TA_CENTER, fontName="Helvetica")
S_H1         = _sty("H1",       fontSize=15, textColor=AMAZON_DARK,   spaceAfter=6,  spaceBefore=16, fontName="Helvetica-Bold")
S_H2         = _sty("H2",       fontSize=11, textColor=ACCENT_BLUE,   spaceAfter=4,  spaceBefore=10, fontName="Helvetica-Bold")
S_BODY       = _sty("Body",     fontSize=9,  textColor=TEXT_DARK,     spaceAfter=4,  leading=13, fontName="Helvetica")
S_NOTE       = _sty("Note",     fontSize=8,  textColor=TEXT_GREY,     spaceAfter=4,  leading=11, fontName="Helvetica-Oblique")
S_TOC        = _sty("TOC",      fontSize=10, textColor=TEXT_DARK,     spaceAfter=3,  fontName="Helvetica")
S_TOC_PG     = _sty("TOCPg",   fontSize=10, textColor=TEXT_GREY,     alignment=TA_RIGHT, fontName="Helvetica")

# Cell paragraph styles for table content (tight leading, auto-wrap)
S_CELL_HDR  = _sty("CellHdr",  fontSize=8, textColor=WHITE,      fontName="Helvetica-Bold", leading=10, spaceBefore=0, spaceAfter=0)
S_CELL      = _sty("Cell",     fontSize=8, textColor=TEXT_DARK,  fontName="Helvetica",      leading=11, spaceBefore=0, spaceAfter=0)
S_CELL_CODE = _sty("CellCode", fontSize=7.5, textColor=colors.HexColor("#1A237E"), fontName="Helvetica-Bold", leading=10, spaceBefore=0, spaceAfter=0)
S_CELL_GREY = _sty("CellGrey", fontSize=8, textColor=TEXT_GREY,  fontName="Helvetica",      leading=11, spaceBefore=0, spaceAfter=0)

# ── Cell helpers ───────────────────────────────────────────────────────────────
def _h(text: str) -> Paragraph:
    """Header cell — white bold."""
    return Paragraph(str(text), S_CELL_HDR)

def _p(text: str) -> Paragraph:
    """Normal cell — wraps automatically."""
    return Paragraph(str(text), S_CELL)

def _code(text: str) -> Paragraph:
    """Code/column-name cell — dark blue bold."""
    return Paragraph(str(text), S_CELL_CODE)

def _grey(text: str) -> Paragraph:
    """Grey secondary cell."""
    return Paragraph(str(text), S_CELL_GREY)

def _wrap(rows: list[list]) -> list[list]:
    """
    Convert a 2D list of strings to Paragraphs so ReportLab word-wraps correctly.
    Row 0 = header (white bold), rest = body cells.
    """
    out = []
    for r_idx, row in enumerate(rows):
        new_row = []
        for c_idx, cell in enumerate(row):
            if r_idx == 0:
                new_row.append(_h(cell))
            elif c_idx == 0:
                new_row.append(_code(cell))
            else:
                new_row.append(_p(cell))
        out.append(new_row)
    return out


# ── Table style ────────────────────────────────────────────────────────────────
def _tbl_style(row_count: int) -> TableStyle:
    # ReportLab TableStyle coords: (col, row) — NOT (row, col)
    cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  AMAZON_DARK),   # header row
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("GRID",          (0, 0), (-1, -1), 0.35, colors.HexColor("#DDDDDD")),
        ("BOX",           (0, 0), (-1, -1), 0.6,  AMAZON_DARK),
        ("BACKGROUND",    (0, 1), (-1, -1), WHITE),
    ]
    # Alternate row shading: even body rows (rows 2, 4, 6 …)
    for row_idx in range(2, row_count, 2):
        cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), AMAZON_LIGHT))
    return TableStyle(cmds)


# ── Header / Footer ────────────────────────────────────────────────────────────
def _on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(AMAZON_DARK)
    canvas.rect(0, h - 1.2*cm, w, 1.2*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(WHITE)
    canvas.drawString(1.5*cm, h - 0.8*cm, "Amazon India Sales Analytics")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(AMAZON_ORANGE)
    canvas.drawRightString(w - 1.5*cm, h - 0.8*cm, "Data Dictionary  |  Confidential")
    canvas.setFillColor(AMAZON_LIGHT)
    canvas.rect(0, 0, w, 0.9*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(TEXT_GREY)
    canvas.drawString(1.5*cm, 0.3*cm, f"Generated {datetime.now().strftime('%d %B %Y')}")
    canvas.drawCentredString(w/2, 0.3*cm, "INTERNAL USE — NOT FOR DISTRIBUTION")
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(AMAZON_ORANGE)
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


def _schema_block(story, title: str, note: str, rows: list[list]):
    story.append(Paragraph(title, S_H2))
    if note:
        story.append(Paragraph(note, S_NOTE))
    widths = [USABLE_W * f for f in (0.34, 0.17, 0.49)]
    t = Table(_wrap(rows), colWidths=widths, repeatRows=1)
    t.setStyle(_tbl_style(len(rows)))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))


def build():
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=A4,
        leftMargin=L_MARGIN, rightMargin=R_MARGIN,
        topMargin=2.0*cm, bottomMargin=1.8*cm,
    )
    story = []

    # ── COVER ──────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 5*cm))
    story.append(Paragraph("DATA DICTIONARY", S_TITLE))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Amazon India: A Decade of Sales Analytics", S_SUBTITLE))
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("11 Years of E-Commerce Data  ·  1.1 Million Transactions  ·  PostgreSQL Star Schema", S_COVER_META))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"Version 1.0  ·  {datetime.now().strftime('%B %Y')}  ·  Amazon India Sales Analytics", S_COVER_META))
    story.append(PageBreak())

    # ── TABLE OF CONTENTS ──────────────────────────────────────────────────────
    story.append(Paragraph("Table of Contents", S_H1))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceAfter=10))
    toc = [
        ("1. Project Overview",                   "3"),
        ("2. Dataset 1 — Sales Transactions",     "3"),
        ("   2.1  Column Reference (36 columns)", "3"),
        ("   2.2  Data Quality Issues (10 Fixes)","5"),
        ("3. Dataset 2 — Product Catalog",        "6"),
        ("4. PostgreSQL Star Schema",             "7"),
        ("5. Key Business Metrics Reference",     "8"),
    ]
    for label, pg in toc:
        row = Table([[Paragraph(label, S_TOC), Paragraph(pg, S_TOC_PG)]],
                    colWidths=[USABLE_W * 0.88, USABLE_W * 0.12])
        row.setStyle(TableStyle([("VALIGN", (0,0),(-1,-1), "MIDDLE"),
                                  ("BOTTOMPADDING",(0,0),(-1,-1),2)]))
        story.append(row)
    story.append(PageBreak())

    # ── SECTION 1 — PROJECT OVERVIEW ──────────────────────────────────────────
    story.append(Paragraph("1. Project Overview", S_H1))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceAfter=8))
    overview_rows = [
        ["Attribute", "Detail"],
        ["Project Name",    "Amazon India: A Decade of Sales Analytics"],
        ["Data Period",     "2015 – 2025 (11 years)"],
        ["Raw Transactions","1,127,609 rows across 11 annual CSV files"],
        ["After Cleaning",  "1,122,000 rows (99.5% retention rate)"],
        ["Products",        "2,004 unique SKUs across 6 subcategories"],
        ["Database",        "PostgreSQL 15+ — Star schema with 4 tables"],
        ["ETL Pipeline",    "Python 3.11 · SQLAlchemy · Pandera validation"],
        ["Subcategories",   "Smartphones, Laptops, Tablets, Smart Watch, Audio, TV & Entertainment"],
        ["Geographic Scope","30+ Indian cities — Metro / Tier1 / Tier2 / Rural"],
    ]
    widths_2col = [USABLE_W * 0.30, USABLE_W * 0.70]
    t_ov = Table(_wrap(overview_rows), colWidths=widths_2col, repeatRows=1)
    t_ov.setStyle(_tbl_style(len(overview_rows)))
    story.append(t_ov)
    story.append(Spacer(1, 0.5*cm))

    # ── SECTION 2 — SALES TRANSACTIONS ────────────────────────────────────────
    story.append(Paragraph("2. Dataset 1 — Sales Transactions", S_H1))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceAfter=4))
    story.append(Paragraph(
        "Source: <b>data/raw/amazon_india_{year}.csv</b> (11 files, 2015–2025)  "
        "Cleaned output: <b>data/processed/cleaned_df_sales.csv</b>",
        S_BODY))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("2.1  Column Reference", S_H2))

    # 4 cols: column name | type | description | example
    widths_tx = [USABLE_W * f for f in (0.24, 0.13, 0.44, 0.19)]
    tx_rows = [
        ["Column", "Data Type", "Description", "Example"],
        ["transaction_id",        "VARCHAR",   "Unique transaction identifier",                              "TXN_2019_000001"],
        ["order_date",            "DATE",      "Order date — standardised to YYYY-MM-DD",                  "2019-10-15"],
        ["order_month",           "SMALLINT",  "Month number extracted from order_date",                   "10"],
        ["order_year",            "SMALLINT",  "Year extracted from order_date",                           "2019"],
        ["order_quarter",         "SMALLINT",  "Quarter number (1–4)",                                     "4"],
        ["customer_id",           "VARCHAR",   "Unique customer identifier",                               "CUST_000123"],
        ["customer_city",         "VARCHAR",   "Standardised city name (Bangalore normalised to Bengaluru)","Bengaluru"],
        ["customer_state",        "VARCHAR",   "Indian state",                                             "Karnataka"],
        ["customer_tier",         "VARCHAR",   "City tier classification: Metro / Tier1 / Tier2 / Rural",  "Metro"],
        ["customer_spending_tier","VARCHAR",   "Spending behaviour segment: High / Medium / Low",          "High"],
        ["customer_age_group",    "VARCHAR",   "Age bracket: 18-25 / 26-35 / 36-45 / 46-55 / 55+",       "26-35"],
        ["is_prime_member",       "BOOLEAN",   "Prime membership flag. Cleaned from Yes/No/1/0/Y/N",       "True"],
        ["product_id",            "VARCHAR",   "Unique product identifier",                                "PROD_000456"],
        ["product_name",          "VARCHAR",   "Product display name",                                     "Samsung Galaxy S21"],
        ["category",              "VARCHAR",   "Top-level category — 'Electronics' for all rows in this dataset","Electronics"],
        ["subcategory",           "VARCHAR",   "Product subcategory — 6 distinct values; use this for analysis","Smartphones"],
        ["brand",                 "VARCHAR",   "Brand name (100+ brands)",                                 "Samsung"],
        ["is_prime_eligible",     "BOOLEAN",   "Whether product qualifies for Prime free delivery",        "True"],
        ["product_weight_kg",     "NUMERIC",   "Product weight in kilograms",                              "0.18"],
        ["product_rating",        "NUMERIC",   "Product catalogue rating 1.0–5.0",                        "4.2"],
        ["original_price_inr",    "NUMERIC",   "List price before discount. Cleaned: removed Rs symbols, commas, text","75000.00"],
        ["discount_percent",      "NUMERIC",   "Discount applied as percentage (0–100)",                  "15.0"],
        ["discounted_price_inr",  "NUMERIC",   "Price after discount applied",                             "63750.00"],
        ["quantity",              "SMALLINT",  "Units ordered (typically 1–3 units)",                     "1"],
        ["subtotal_inr",          "NUMERIC",   "discounted_price_inr multiplied by quantity",              "63750.00"],
        ["delivery_charges",      "NUMERIC",   "Delivery fee in INR. Zero for Prime-eligible orders",     "200.00"],
        ["final_amount_inr",      "NUMERIC",   "Total amount paid: subtotal plus delivery charges",        "63950.00"],
        ["payment_method",        "VARCHAR",   "Standardised payment method. Cleaned: UPI/PhonePe/GooglePay all mapped to UPI","UPI"],
        ["payment_category",      "VARCHAR",   "Grouped payment type: Digital / Card / Cash",              "Digital"],
        ["delivery_days",         "NUMERIC",   "Days from order placement to delivery. Cleaned from text values like 'Same Day'","3.0"],
        ["delivery_type",         "VARCHAR",   "Delivery speed tier: Same Day / Express / Standard",       "Standard"],
        ["is_festival_sale",      "BOOLEAN",   "Festival period order flag. Cleaned from inconsistent True/False/1/0","False"],
        ["festival_name",         "VARCHAR",   "Festival name when is_festival_sale is True; NULL otherwise","Diwali"],
        ["sale_type",             "VARCHAR",   "Sale classification: Festival / Regular",                  "Regular"],
        ["return_status",         "VARCHAR",   "Order outcome: Delivered / Returned / Cancelled",          "Delivered"],
        ["customer_rating",       "NUMERIC",   "Post-purchase customer rating 1.0–5.0. Imputed if missing","4.0"],
        ["rating_missing",        "SMALLINT",  "Flag: 1 if customer_rating was imputed; 0 if original",  "0"],
    ]
    t_tx = Table(_wrap(tx_rows), colWidths=widths_tx, repeatRows=1)
    t_tx.setStyle(_tbl_style(len(tx_rows)))
    story.append(t_tx)
    story.append(Spacer(1, 0.5*cm))

    # ── Data Quality Table ──────────────────────────────────────────────────────
    story.append(Paragraph("2.2  Data Quality Issues Fixed (10 Cleaning Challenges)", S_H2))
    widths_dq = [USABLE_W * f for f in (0.04, 0.33, 0.40, 0.23)]
    dq_rows = [
        ["#", "Issue", "Method Applied", "Rows Affected"],
        ["1",  "Mixed date formats: DD/MM/YYYY, DD-MM-YY, YYYY-MM-DD, and invalid entries",
               "Multi-format parse with fallback; invalid dates → sentinel 1900-01-01",
               "~4,757 rows"],
        ["2",  "Price column: Rs symbols, Indian comma separators, text like 'Price on Request'",
               "Regex strip all non-numeric; text entries → NaN → median imputation per subcategory",
               "~2,898 rows"],
        ["3",  "Ratings in multiple formats: '4 stars', '3/5', '2.5/5.0', missing values",
               "Regex numeric extraction; fraction normalisation to 1.0–5.0 scale",
               "~15,336 rows"],
        ["4",  "Inconsistent city names: Bangalore / Bengaluru, spelling errors, case variations",
               "Canonical mapping dictionary with 30+ aliases; lowercase normalisation",
               "~222 rows"],
        ["5",  "Boolean columns with mixed: True/False, Yes/No, 1/0, Y/N, and nulls",
               "Explicit mapping table for all variants → Python bool",
               "All boolean columns"],
        ["6",  "Category name variations: Electronics / ELECTRONICS / Electronic & Accessories",
               "Lowercase + canonical string mapping to 'Electronics'",
               "~0 after standardise"],
        ["7",  "Delivery days: negative values, text like 'Same Day' or '1-2 days', values > 30",
               "Text parsing → numeric; negatives → 0; IQR-based cap at 95th percentile per tier",
               "~110 rows"],
        ["8",  "Duplicate transactions: same customer, product, date, and amount",
               "Keep first occurrence; bulk orders identified separately via quantity > 1",
               "~244 removed"],
        ["9",  "Price outliers: prices 100x expected value due to decimal point entry errors",
               "IQR method per subcategory; decimal shift correction applied",
               "~1,457 fixed"],
        ["10", "Payment method naming: UPI / PhonePe / GooglePay, Credit Card / CC / CREDIT_CARD",
               "Canonical mapping to 7 standard categories: UPI, COD, Credit Card, Debit Card, Net Banking, Wallet, BNPL",
               "All payment rows"],
    ]
    t_dq = Table(_wrap(dq_rows), colWidths=widths_dq, repeatRows=1)
    t_dq.setStyle(_tbl_style(len(dq_rows)))
    story.append(t_dq)
    story.append(PageBreak())

    # ── SECTION 3 — PRODUCT CATALOG ───────────────────────────────────────────
    story.append(Paragraph("3. Dataset 2 — Product Catalog", S_H1))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceAfter=4))
    story.append(Paragraph(
        "Source: <b>data/raw/amazon_india_products_catalog.csv</b>  "
        "Output: <b>data/processed/cleaned_products.csv</b>  "
        "2,004 rows x 11 columns. Joined to fact_transactions on product_id.",
        S_BODY))
    story.append(Spacer(1, 0.3*cm))

    widths_prod = [USABLE_W * f for f in (0.26, 0.15, 0.41, 0.18)]
    prod_rows = [
        ["Column",           "Data Type",  "Description",                                           "Example"],
        ["product_id",       "VARCHAR PK", "Unique product identifier — foreign key in fact table", "PROD_000456"],
        ["product_name",     "VARCHAR",    "Full product display name",                             "Samsung Galaxy S21"],
        ["category",         "VARCHAR",    "Top-level category ('Electronics' for all rows)",        "Electronics"],
        ["subcategory",      "VARCHAR",    "Subcategory — 6 distinct values; use for all analysis", "Smartphones"],
        ["brand",            "VARCHAR",    "Brand name (100+ distinct brands in dataset)",           "Samsung"],
        ["base_price_2015",  "NUMERIC",    "Original launch price in 2015 (INR) — inflation baseline","45000.00"],
        ["weight_kg",        "NUMERIC",    "Product physical weight in kilograms",                   "0.18"],
        ["rating",           "NUMERIC",    "Catalogue aggregate rating averaged across all reviews", "4.3"],
        ["is_prime_eligible","BOOLEAN",    "Whether product qualifies for Amazon Prime free delivery","True"],
        ["launch_year",      "SMALLINT",   "Year product was first listed on Amazon India",          "2021"],
        ["model",            "VARCHAR",    "Model or variant identifier string",                     "SM-G991B"],
    ]
    t_prod = Table(_wrap(prod_rows), colWidths=widths_prod, repeatRows=1)
    t_prod.setStyle(_tbl_style(len(prod_rows)))
    story.append(t_prod)
    story.append(PageBreak())

    # ── SECTION 4 — STAR SCHEMA ───────────────────────────────────────────────
    story.append(Paragraph("4. PostgreSQL Star Schema", S_H1))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceAfter=6))
    story.append(Paragraph(
        "After ETL, all data is loaded into a PostgreSQL star schema. "
        "Schema defined in <b>scripts/init_db.sql</b> and created automatically on first run. "
        "Upsert strategy (INSERT ... ON CONFLICT DO UPDATE) makes ETL re-runnable safely.",
        S_BODY))
    story.append(Spacer(1, 0.3*cm))

    tables_summary = [
        ["Table",             "Type",      "Full Dataset Rows", "Purpose"],
        ["fact_transactions", "Fact",      "~1,122,000",        "One row per transaction — all numeric measures"],
        ["dim_customers",     "Dimension", "~354,000",          "Customer master: city, tier, age group, Prime flag"],
        ["dim_products",      "Dimension", "2,004",             "Product master: category, brand, launch year"],
        ["dim_time",          "Dimension", "~4,000",            "Date dimension: year, month, quarter, festival flag"],
    ]
    widths_4col = [USABLE_W * f for f in (0.27, 0.13, 0.20, 0.40)]
    t_sum = Table(_wrap(tables_summary), colWidths=widths_4col, repeatRows=1)
    t_sum.setStyle(_tbl_style(len(tables_summary)))
    story.append(t_sum)
    story.append(Spacer(1, 0.4*cm))

    _schema_block(story,
        "4.1  fact_transactions — Central Fact Table",
        "One row per transaction. Foreign keys to all 3 dimension tables. Upserted on transaction_id.",
        [
            ["Column",               "Type",     "Constraint / Notes"],
            ["transaction_id",       "VARCHAR",  "PRIMARY KEY"],
            ["order_date",           "DATE",     "Indexed — primary date filter"],
            ["order_year",           "SMALLINT", "Indexed — used in all annual aggregations"],
            ["order_month",          "SMALLINT", ""],
            ["order_quarter",        "SMALLINT", ""],
            ["customer_id",          "VARCHAR",  "FK — dim_customers.customer_id"],
            ["product_id",           "VARCHAR",  "FK — dim_products.product_id"],
            ["original_price_inr",   "NUMERIC",  "NULL allowed for products without list price"],
            ["discount_percent",     "NUMERIC",  "0 to 100"],
            ["discounted_price_inr", "NUMERIC",  ""],
            ["final_amount_inr",     "NUMERIC",  "Indexed — used in all revenue aggregations"],
            ["quantity",             "SMALLINT", "Typically 1–3 units"],
            ["payment_method",       "VARCHAR",  "7 standard values after cleaning"],
            ["delivery_days",        "NUMERIC",  "NULL if order not yet delivered"],
            ["delivery_type",        "VARCHAR",  "Same Day / Express / Standard"],
            ["is_festival_sale",     "BOOLEAN",  ""],
            ["festival_name",        "VARCHAR",  "NULL when is_festival_sale = False"],
            ["product_rating",       "NUMERIC",  "1.0 to 5.0"],
            ["customer_rating",      "NUMERIC",  "1.0 to 5.0; NULL if customer has not rated"],
            ["return_status",        "VARCHAR",  "Delivered / Returned / Cancelled"],
        ]
    )

    _schema_block(story,
        "4.2  dim_customers — Customer Dimension",
        "One row per unique customer. Populated from distinct customer_ids in fact_transactions.",
        [
            ["Column",              "Type",    "Constraint / Notes"],
            ["customer_id",         "VARCHAR", "PRIMARY KEY"],
            ["customer_city",       "VARCHAR", "Standardised city name"],
            ["customer_state",      "VARCHAR", "Indian state"],
            ["customer_tier",       "VARCHAR", "Metro / Tier1 / Tier2 / Rural"],
            ["customer_spending_tier","VARCHAR","High / Medium / Low"],
            ["customer_age_group",  "VARCHAR", "18-25 / 26-35 / 36-45 / 46-55 / 55+"],
            ["is_prime_member",     "BOOLEAN", ""],
        ]
    )

    _schema_block(story,
        "4.3  dim_products — Product Dimension",
        "One row per unique product. Sourced from the product catalogue CSV.",
        [
            ["Column",         "Type",     "Constraint / Notes"],
            ["product_id",     "VARCHAR",  "PRIMARY KEY"],
            ["product_name",   "VARCHAR",  ""],
            ["category",       "VARCHAR",  "'Electronics' for all rows in this dataset"],
            ["subcategory",    "VARCHAR",  "6 distinct values — always use subcategory for analysis"],
            ["brand",          "VARCHAR",  "100+ brands"],
            ["launch_year",    "SMALLINT", "Used for product lifecycle analysis"],
            ["is_prime_eligible","BOOLEAN",""],
        ]
    )

    _schema_block(story,
        "4.4  dim_time — Date Dimension",
        "Date dimension for time-based joins. One row per calendar date in the dataset range.",
        [
            ["Column",             "Type",     "Constraint / Notes"],
            ["date",               "DATE",     "PRIMARY KEY"],
            ["year",               "SMALLINT", ""],
            ["month",              "SMALLINT", "1 to 12"],
            ["quarter",            "SMALLINT", "1 to 4"],
            ["is_festival_period", "BOOLEAN",  "True for known festival windows: Diwali, Prime Day, Great Indian Festival, etc."],
        ]
    )

    # ── SECTION 5 — KEY METRICS ───────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Key Business Metrics Reference", S_H1))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceAfter=8))
    story.append(Paragraph(
        "Key figures from the full 1.1M row dataset for quick reference during analysis and presentation.",
        S_BODY))
    story.append(Spacer(1, 0.3*cm))

    kpi_rows = [
        ["Metric",                           "Value",                    "Context"],
        ["Total raw transactions",           "1,127,609",               "Across 11 annual CSV files"],
        ["After ETL cleaning",               "1,122,000",               "99.5% retention rate"],
        ["Unique customers",                 "~354,000",                 "Long-term customer base"],
        ["Unique products (SKUs)",           "2,004",                    "Electronics catalog"],
        ["Dataset period",                   "2015 to 2025",            "11 complete years"],
        ["Subcategories (6 total)",          "Smartphones, Laptops, Tablets, Smart Watch, Audio, TV","Use subcategory not category"],
        ["States covered",                   "15+ Indian states",       ""],
        ["Cities covered",                   "30+ cities",              "Metro to Rural"],
        ["Smartphone revenue share",         "73.2%",                   "India is mobile-first"],
        ["Peak revenue year",                "2020",                    "COVID e-commerce surge"],
        ["10-Year Revenue CAGR",             "6.2%",                    "Healthy mature-market growth"],
        ["Prime vs Non-Prime AOV",           "Rs 78,127 vs Rs 62,296", "+25% for Prime members"],
        ["Average Customer CLV (mean)",      "Rs 74,100",               "Right-skewed distribution"],
        ["Average Customer CLV (median)",    "Rs 47,900",               "More representative typical value"],
        ["Festival AOV vs Regular AOV",      "Rs 47,400 vs Rs 77,700", "Festival is LOWER — discount effect"],
        ["Highest return rate category",     "Audio at 8.1%",           "Subjective sound quality"],
        ["Lowest return rate category",      "TV & Entertainment 5.1%", "Screen size meets expectations"],
        ["Optimal discount range",           "20 to 30%",               "Highest revenue per INR discounted"],
        ["Average delivery time",            "3.3 to 3.5 days all tiers","Rural only 0.2 days slower than Metro"],
        ["Top 36% products",                 "Generate 80% revenue",    "Pareto concentration — 720 SKUs drive the business"],
        ["UPI payment share 2025",           "Largest and growing",     "Displaced COD since 2016 demonetisation"],
        ["BNPL return rate",                 "7.8% vs 6.7% UPI",       "Deferred payment reduces commitment"],
    ]
    widths_kpi = [USABLE_W * f for f in (0.38, 0.28, 0.34)]
    t_kpi = Table(_wrap(kpi_rows), colWidths=widths_kpi, repeatRows=1)
    t_kpi.setStyle(_tbl_style(len(kpi_rows)))
    story.append(t_kpi)

    doc.build(story, onFirstPage=_on_cover, onLaterPages=_on_page)
    print(f"[OK] Data Dictionary PDF saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
