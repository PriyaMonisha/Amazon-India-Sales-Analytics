# Data Dictionary — Amazon India Sales Analytics

Two source datasets, cleaned and loaded into a PostgreSQL star schema.

---

## Dataset 1: Sales Transactions (`cleaned_df_sales.csv`)

**Rows:** 1,122,000 · **Period:** 2015–2025 · **Source:** `data/raw/amazon_india_{year}.csv` (11 files)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `transaction_id` | string | Unique transaction identifier | `TXN_2015_000001` |
| `order_date` | date | Order date in YYYY-MM-DD format (cleaned from mixed formats) | `2019-10-15` |
| `order_month` | int | Month number extracted from order_date | `10` |
| `order_year` | int | Year extracted from order_date | `2019` |
| `order_quarter` | int | Quarter (1–4) extracted from order_date | `4` |
| `customer_id` | string | Unique customer identifier | `CUST_000123` |
| `customer_city` | string | Standardised city name (cleaned: Bangalore/Bengaluru → Bengaluru) | `Bengaluru` |
| `customer_state` | string | Indian state | `Karnataka` |
| `customer_tier` | string | City tier classification: Metro / Tier1 / Tier2 / Rural | `Metro` |
| `customer_spending_tier` | string | Spending behaviour segment: High / Medium / Low | `High` |
| `customer_age_group` | string | Age bracket: 18-25 / 26-35 / 36-45 / 46-55 / 55+ | `26-35` |
| `is_prime_member` | bool | Whether customer has Prime membership (True/False, cleaned from Yes/No/1/0) | `True` |
| `product_id` | string | Unique product identifier | `PROD_000456` |
| `product_name` | string | Product display name | `Samsung Galaxy S21` |
| `category` | string | Top-level product category (8 categories) | `Electronics` |
| `subcategory` | string | Product subcategory (25+ subcategories) | `Smartphones` |
| `brand` | string | Product brand (100+ brands) | `Samsung` |
| `original_price_inr` | float | List price in INR before discount (cleaned: ₹ symbols, commas removed; outliers corrected) | `75000.00` |
| `discount_percent` | float | Discount applied as percentage (0–100) | `15.0` |
| `discounted_price_inr` | float | Price after discount | `63750.00` |
| `quantity` | int | Units ordered | `1` |
| `subtotal_inr` | float | `discounted_price_inr × quantity` | `63750.00` |
| `final_amount_inr` | float | Final amount paid including delivery charges | `63950.00` |
| `delivery_charges` | float | Delivery fee in INR (0 for Prime-eligible) | `200.00` |
| `payment_method` | string | Standardised payment method (cleaned: UPI/PhonePe/GooglePay → UPI) | `UPI` |
| `payment_category` | string | Grouped payment type: Digital / Card / Cash | `Digital` |
| `delivery_days` | float | Days from order to delivery (cleaned: negatives fixed, "Same Day" → 0) | `3.0` |
| `delivery_type` | string | Delivery speed tier: Same Day / Express / Standard | `Standard` |
| `is_prime_eligible` | bool | Whether the product qualifies for Prime delivery | `True` |
| `is_festival_sale` | bool | Whether order was placed during a festival sale period | `False` |
| `festival_name` | string | Festival name if is_festival_sale=True, else null | `Diwali` |
| `sale_type` | string | Sale classification: Festival / Regular | `Regular` |
| `product_rating` | float | Product rating 1.0–5.0 (cleaned: "4 stars", "3/5" → numeric) | `4.2` |
| `customer_rating` | float | Customer-given rating after purchase (1.0–5.0) | `4.0` |
| `rating_missing` | int | Flag: 1 if customer_rating was imputed, 0 if original | `0` |
| `return_status` | string | Order outcome: Delivered / Returned / Cancelled | `Delivered` |
| `product_weight_kg` | float | Product weight in kilograms | `0.18` |

---

## Dataset 2: Product Catalog (`cleaned_products.csv`)

**Rows:** ~2,000 · **Source:** `data/raw/amazon_india_products_catalog.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `product_id` | string | Unique product identifier (joins to sales dataset) | `PROD_000456` |
| `product_name` | string | Product display name | `Samsung Galaxy S21` |
| `category` | string | Top-level category | `Electronics` |
| `subcategory` | string | Subcategory | `Smartphones` |
| `brand` | string | Brand name | `Samsung` |
| `base_price_2015` | float | Original launch price in 2015 (INR) | `45000.00` |
| `weight_kg` | float | Product weight | `0.18` |
| `rating` | float | Catalogue rating (avg across all reviews) | `4.3` |
| `is_prime_eligible` | bool | Prime-eligible product flag | `True` |
| `launch_year` | int | Year product was first listed on Amazon India | `2021` |
| `model` | string | Model/variant identifier | `SM-G991B` |

---

## PostgreSQL Star Schema

After ETL, data is loaded into 4 tables:

### `fact_transactions`
The main fact table — one row per transaction. Foreign keys to all dimension tables.

| Column | Type | Notes |
|--------|------|-------|
| `transaction_id` | VARCHAR PK | Primary key |
| `order_date` | DATE | Indexed |
| `order_year` | SMALLINT | Indexed |
| `order_month` | SMALLINT | |
| `order_quarter` | SMALLINT | |
| `customer_id` | VARCHAR | FK → dim_customers |
| `product_id` | VARCHAR | FK → dim_products |
| `original_price_inr` | NUMERIC | |
| `discount_percent` | NUMERIC | |
| `discounted_price_inr` | NUMERIC | |
| `final_amount_inr` | NUMERIC | Indexed |
| `quantity` | SMALLINT | |
| `payment_method` | VARCHAR | |
| `delivery_days` | NUMERIC | |
| `delivery_type` | VARCHAR | |
| `is_festival_sale` | BOOLEAN | |
| `festival_name` | VARCHAR | |
| `product_rating` | NUMERIC | |
| `customer_rating` | NUMERIC | |
| `return_status` | VARCHAR | |

### `dim_customers`
One row per unique customer.

| Column | Type | Notes |
|--------|------|-------|
| `customer_id` | VARCHAR PK | |
| `customer_city` | VARCHAR | |
| `customer_state` | VARCHAR | |
| `customer_tier` | VARCHAR | Metro / Tier1 / Tier2 / Rural |
| `customer_spending_tier` | VARCHAR | |
| `customer_age_group` | VARCHAR | |
| `is_prime_member` | BOOLEAN | |

### `dim_products`
One row per unique product.

| Column | Type | Notes |
|--------|------|-------|
| `product_id` | VARCHAR PK | |
| `product_name` | VARCHAR | |
| `category` | VARCHAR | |
| `subcategory` | VARCHAR | |
| `brand` | VARCHAR | |
| `launch_year` | SMALLINT | |
| `is_prime_eligible` | BOOLEAN | |

### `dim_time`
Date dimension for time-based analysis.

| Column | Type | Notes |
|--------|------|-------|
| `date` | DATE PK | |
| `year` | SMALLINT | |
| `month` | SMALLINT | |
| `quarter` | SMALLINT | |
| `is_festival_period` | BOOLEAN | |

---

## Data Quality Issues Fixed (10 Cleaning Challenges)

| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | Mixed date formats (DD/MM/YYYY, DD-MM-YY, YYYY-MM-DD, invalid) | Parsed with multiple format attempts; invalid → NaT → dropped |
| 2 | Price with ₹ symbols, commas, "Price on Request" text | Regex strip → numeric; non-numeric → NaN |
| 3 | Ratings in multiple formats (4 stars, 3/5, 2.5/5.0) | Extracted numeric, scaled to 1.0–5.0 |
| 4 | Inconsistent city names (Bangalore/Bengaluru, spelling errors) | Mapping dictionary + case normalisation |
| 5 | Boolean columns with True/False/Yes/No/1/0/Y/N | Mapped all variants to Python bool |
| 6 | Category name variations (Electronics/ELECTRONICS/Electronic) | Lowercased + canonical mapping |
| 7 | Delivery days: negatives, "Same Day", "1-2 days", outliers | Parsed text → numeric; negatives → 0; outliers capped |
| 8 | Duplicate transactions (same customer + product + date + amount) | Kept first occurrence; bulk orders identified by quantity > 1 |
| 9 | Price outliers (100× higher due to decimal point errors) | IQR method per subcategory; corrected decimal shift |
| 10 | Payment method naming (UPI/PhonePe/GooglePay, CC/Credit Card) | Canonical mapping to 7 standard categories |
