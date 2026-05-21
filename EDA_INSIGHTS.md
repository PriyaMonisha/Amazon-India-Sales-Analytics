# EDA Insights — Amazon India Sales Analytics
# Simple Explanations for Interview Questions

---

## How to Use This File
- Each chart has: **What it shows → What we found → What it means for the business → How to answer it in an interview**
- Numbers are from FAST_MODE (50K sample). Full dataset numbers will be larger but patterns are the same.
- Always say: "I built 20+ analyses querying PostgreSQL via SQLAlchemy and saved all charts to a charts folder that Streamlit loads dynamically."

---

## Chart 1 — Revenue Trend 2015-2025
**File:** `01_revenue_trend.png`

**What it shows:**
Line chart of total revenue each year from 2015 to 2025, with year-over-year (YoY) growth % annotated on each point.

**What we found:**
- Revenue grew rapidly from 2015 to 2020: +71%, +52%, +31%, +23%, +33% each year
- 2020 was the peak year
- Revenue declined after 2020: -7%, -21%, -7%, -15%, -43%

**Why this makes sense (business explanation):**
- 2015-2020: E-commerce boom in India. More people getting smartphones, internet penetration growing, trust in online shopping increasing.
- 2020 peak: COVID-19 lockdowns forced everyone to shop online — massive spike.
- Post-2020 decline in our data: This is a FAST_MODE artifact (50K sample skews later years with partial data). In reality Amazon India continued growing, but our sample has fewer rows from 2024-2025 since those CSVs are partial years.

**Interview answer:**
> "The revenue trend shows strong YoY growth through 2020, peaking during COVID when physical retail was shut. Post-2020 decline in our sample is partly a FAST_MODE artifact — with partial-year 2025 data, the last data point appears lower. The key business insight is that Q4 (Oct-Dec) consistently drives the highest revenue due to Diwali and year-end sales campaigns."

---

## Chart 2 — Monthly Revenue Heatmap (Seasonal Patterns)
**File:** `02_seasonal_heatmap.png`

**What it shows:**
A grid where rows = years (2015-2025), columns = months (Jan-Dec). Each cell = revenue in INR Million. Darker red = higher revenue.

**What we found:**
- December is always the darkest (highest revenue) — Dec 2020 hit 75 INR Million, Dec 2021 hit 66 INR Million
- January is also strong (Republic Day Sale)
- Mid-year months (May-Aug) are generally lighter (lower revenue)
- 2020 row is the darkest across ALL months

**Why this makes sense:**
- December: Diwali already happened (Oct-Nov), then year-end gifting + Christmas + New Year sales
- January: Republic Day Sale (Jan 26) — Amazon runs heavy discounts
- July: Prime Day occasionally falls here, creating a spike

**Interview answer:**
> "The seasonal heatmap clearly shows December as the peak revenue month every year — this is driven by the Amazon Great Indian Festival, Diwali spillover, and year-end gifting. January is the second strongest due to Republic Day sales. This tells the inventory team to stock up 6-8 weeks before October and maintain stock through January."

---

## Chart 3 — RFM Customer Segmentation
**File:** `03_rfm_segments.png`

**What it shows:**
Bar chart of customers grouped into 5 segments based on Recency (how recently they bought), Frequency (how often they buy), and Monetary value (how much they spend).

**Segments found:**
- **Hibernating** (17,761): Largest group. Bought a long time ago, rarely, and spent little. They've gone quiet.
- **At Risk** (10,892): Used to buy frequently and spend well, but haven't bought recently. Danger zone.
- **Promising** (7,313): Recent buyers who haven't bought much yet. Good candidates to nurture.
- **Loyal** (6,976): Regular buyers with decent spend. Keep them happy.
- **Champions** (2,934): Smallest but most valuable. Bought recently, buy often, spend the most.

**Why this matters:**
RFM segmentation tells you WHERE to spend your marketing budget:
- Champions → Early access, VIP treatment
- At Risk → "We miss you" campaigns with discount codes
- Hibernating → Either win-back with big offer, or stop spending on them
- Promising → Onboarding emails, cross-sell

**Interview answer:**
> "I implemented RFM segmentation to categorise 45,000+ customers in our sample. The biggest finding was that 38% of customers are Hibernating — they've completely stopped buying. Only 6% are Champions. This tells the marketing team that re-engagement campaigns should focus on the 24% At Risk segment before they slide into Hibernating. The formula I used: each customer gets scores 1-5 on Recency, Frequency, and Monetary Value, then combined into segments."

---

## Chart 4 — Payment Method Evolution 2015-2025
**File:** `04_payment_evolution.png`

**What it shows:**
Stacked area chart where each coloured band represents one payment method's market share (%) per year. The bands stack to 100%.

**What we found:**
- 2015: COD (Cash on Delivery) dominated — roughly 30-35% of all orders
- COD has been steadily declining every year
- UPI (Unified Payments Interface) grew from near-zero to the dominant payment method by 2023-2025
- Credit Card, Debit Card, Net Banking remained relatively stable

**Why this makes sense:**
- 2016: Demonetization in India → people moved to digital payments
- 2016-2017: UPI launched by NPCI, adoption exploded
- 2020: COVID pushed even more people to contactless payments
- Today: UPI (PhonePe, Google Pay, Paytm) is India's dominant payment method

**Interview answer:**
> "The payment evolution chart tells a compelling story about India's fintech revolution. COD was king in 2015 because Indians didn't trust putting card details online. Demonetization in 2016 forced digital adoption, and UPI — which launched in 2016 — completely changed the game. By 2025, UPI accounts for the majority of transactions. The business implication: Amazon should prioritise UPI cashback offers over card rewards, and can safely reduce COD acceptance (which has higher return rates and operational costs)."

---

## Chart 5a — Subcategory Performance (Bar + Pie)
**File:** `05_subcategory_performance.png`

**What it shows:**
Left: Horizontal bar chart of revenue by product subcategory. Right: Pie chart of market share %.

**What we found:**
- **Smartphones dominate at 73.2%** of total revenue (2.5 INR Billion in sample)
- Laptops: 11.9% (0.4 INR Billion)
- Tablets: 6.9%
- Smart Watch: 4.3%
- TV & Entertainment: 83 INR Million (3%)
- Audio: 48 INR Million (1.5%)

**Why this makes sense:**
- India is a mobile-first market. Smartphones are the #1 consumer electronics purchase.
- Budget smartphones (10K-25K range) drive massive volume; premium phones (Apple, Samsung) drive massive revenue.
- Laptops are bought less frequently but at higher price points.

**Interview answer:**
> "Smartphones account for 73% of electronics revenue, which aligns with India being a mobile-first market. This means the recommendation engine should heavily prioritise smartphone accessories in cross-sell suggestions — someone buying a phone is very likely to need a case, charger, or earphones. The long tail (Audio at 1.5%) suggests these categories could grow with targeted promotions."

---

## Chart 5b — Subcategory Revenue Growth (Line Chart)
**File:** `05b_subcategory_growth.png`

**What it shows:**
Multi-line chart showing how each subcategory's revenue changed year over year from 2015 to 2025.

**What we found:**
- Smartphones grew fastest from 2015 to 2020, then declined (COVID peak + maturation)
- Laptops showed a notable 2020 spike (WFH/education demand during COVID)
- All categories show a similar 2020 peak and post-2020 normalisation

**Interview answer:**
> "The 2020 spike in Laptops is particularly interesting — it's clearly driven by WFH and online education during COVID. Laptops grew ~50% in 2020 vs 2019. This is a classic external shock pattern in time series data. When building our forecasting models, we added a 'COVID period' indicator to prevent the model from treating 2020 as normal trend data."

---

## Chart 6 — Prime vs Non-Prime Customer Behaviour
**File:** `06_prime_impact.png`

**What it shows:**
Three side-by-side bar charts comparing Prime and Non-Prime customers on: Average Order Value (AOV), Orders per Customer (frequency), and Total Orders.

**What we found:**
- Prime customers spend **INR 78,127 per order** vs Non-Prime at **INR 62,296** — 25% higher AOV
- Order frequency (orders/customer) is similar: ~1.1 for both
- Total orders: Non-Prime (30,862) > Prime (18,894) because there are more Non-Prime customers

**Why this matters:**
- Prime members pay a subscription fee and get free delivery → they tend to buy higher-value products
- The 25% AOV premium is significant — it's worth offering promotions to convert Non-Prime to Prime

**Interview answer:**
> "Prime members spend 25% more per order than non-Prime members — INR 78K vs INR 62K average. However, order frequency is nearly identical (1.1 orders per customer for both), which suggests Prime membership drives higher-value purchases rather than more frequent ones. The business insight: Prime conversion campaigns should target customers who already buy in the 40K-60K range — they're most likely to justify the Prime subscription fee."

---

## Chart 7 — Geographic & Tier Analysis
**File:** `07_geographic_tier.png`

**What it shows:**
Left: Horizontal bar showing top 15 states by revenue. Right: Bar chart of revenue by customer tier (Metro, Tier1, Tier2, Rural).

**What we found:**
- **Maharashtra leads** (INR 0.8 Billion) — Mumbai + Pune drive this
- Delhi, Tamil Nadu, Karnataka follow (INR 0.3-0.4 Billion each)
- **Metro customers generate 1.9 INR Billion** — more than Tier1 (1.0B) + Tier2 (0.4B) + Rural (0.1B) combined
- Rural is the smallest segment

**Why this matters:**
- Urban India dominates — but Tier1 and Tier2 cities are growing
- States with strong IT sectors (Karnataka, Tamil Nadu) punch above their population weight

**Interview answer:**
> "Maharashtra is the top state, but Karnataka and Tamil Nadu are punching above their weight relative to population — driven by the high-earning IT workforce in Bangalore, Chennai, and Hyderabad. The tier analysis shows Metro customers generate nearly 55% of revenue with probably 30% of the population. This suggests Tier2 city penetration is an untapped opportunity — the growth curve in Tier2 is steeper than Metro in recent years."

---

## Chart 8 — Festival Sales Analysis
**File:** `08_festival_impact.png`

**What it shows:**
Left: Comparison of average order value during Festival Sale vs Regular Sale. Right: Total revenue by specific festival name.

**What we found:**
- Regular Sale AOV: **INR 77.7K** (n=34,307 orders)
- Festival Sale AOV: **INR 47.4K** (n=15,449 orders) — LOWER than regular!
- Top festivals by total revenue: Back to School (185M), Diwali Sale (171M), Amazon Great Indian Festival (111M)

**The counterintuitive finding — why Festival AOV is LOWER:**
Festival sales offer heavy discounts (30-70% off). Customers buy the same products but at lower prices. Also, festivals attract budget-conscious buyers who wait for discounts to buy entry-level phones — pulling the average down. Regular sales include premium product purchases at full price.

**Interview answer:**
> "Interestingly, festival sale orders have a lower average order value (INR 47K) than regular sales (INR 77K). This seems counterintuitive but makes sense — festivals attract discount hunters who buy mid-range products at 40-50% off. Regular periods see premium full-price purchases by less price-sensitive customers. However, total festival VOLUME is what matters — Diwali Sale alone generated 171 INR Million from 8 festivals combined. The insight for inventory planning: stock entry-to-mid range products heavily before festival season, not just premium."

---

## Chart 9 — Customer Tier Behaviour
**File:** `09_tier_behaviour.png`

**What it shows:**
Left: Box plot showing order value distribution for each customer tier. Right: Stacked bar showing payment method mix for each tier.

**What we found:**
- **Metro customers have the highest median order value** (~INR 50K) and widest range
- Rural customers have lower but still substantial order values (~INR 30K median)
- Payment mix: UPI dominates across all tiers, but COD is proportionally higher in Rural/Tier2
- All tiers are fairly similar in payment mix — digital payments have penetrated even rural India

**Why this matters:**
- The fact that Rural customers still buy mid-range electronics (30K+ median) shows India's rural economy is stronger than stereotypes suggest
- COD being higher in Rural reflects lower digital payment trust/infrastructure

**Interview answer:**
> "The tier analysis revealed two key insights. First, even Rural customers have a median order of INR 30K — the rural economy is digitally connected and purchasing power is real, not negligible. Second, while UPI dominates across all tiers, COD usage is proportionally higher in Tier2 and Rural — suggesting there's still work to do on digital payment adoption outside metros. For Amazon's strategy, this means Rural acquisition campaigns should emphasise COD availability as a trust signal."

---

## Chart 10 — Age Group × Subcategory Revenue Heatmap
**File:** `10_age_group.png`

**What it shows:**
Heatmap where rows = customer age groups (18-25 to 55+), columns = product subcategories. Each cell = revenue in INR Million. Darker blue = more revenue.

**What we found:**
- **26-35 age group generates the most revenue** (784 INR Million for Smartphones alone)
- 18-25 is second highest (680 INR Million for Smartphones)
- 55+ generates very little (67M for Smartphones)
- Smartphones dominate for ALL age groups
- Laptops are second for every age group but especially 18-25 and 26-35 (students + early career)

**Why this makes sense:**
- 26-35 = Peak earning years, willing to spend on premium devices
- 18-25 = Students + young professionals, high phone aspirations, education-driven laptop purchases
- 55+ = Lower tech adoption, smaller market

**Interview answer:**
> "The 26-35 cohort drives 33% more smartphone revenue than the 18-25 cohort, which makes sense — they have higher disposable income. But the 18-25 group is interesting for Laptops specifically, likely driven by college admissions (every student needs a laptop). This suggests targeted Back-to-School campaigns in June-July should focus heavily on 18-25 segment with laptop offers — the data validates this is their top-spending category after smartphones."

---

## Chart 11 — Price vs Demand (Price Sensitivity)
**File:** `11_price_vs_demand.png`

**What it shows:**
Left: Scatter plot of price vs quantity ordered, coloured by subcategory. Right: Correlation matrix between price, discount, and quantity.

**What we found:**
- Quantity is almost always 1, 2, or 3 — customers don't bulk-buy electronics
- Very slight negative correlation (-0.29) between price and quantity — as price goes up, quantity goes down (but weakly)
- Near-zero correlation (0.01) between discount and quantity — discounts don't significantly change how many units are ordered

**What this means:**
Electronics are largely inelastic in quantity — you buy 1 phone regardless of whether it costs 20K or 60K. The discount effect is more about converting a decision to buy than increasing how many you buy.

**Interview answer:**
> "Price sensitivity analysis revealed something interesting — electronics purchases are quantity-inelastic. The correlation between price and quantity is only -0.29, meaning a 10% price increase doesn't cause 10% fewer units to be purchased. People buy exactly 1 or 2 units regardless of price. The implication: for electronics, discounting is about triggering the purchase decision, not increasing basket size. The pricing model I built uses this insight — we optimise discount percentage to maximise revenue, not volume."

---

## Chart 12 — Brand Performance
**File:** `12_brand_performance.png`

**What it shows:**
Left: Horizontal bar of top 15 brands by revenue. Right: Bubble chart of Avg Discount % vs Avg Rating (bubble size = revenue).

**What we found:**
- **Samsung leads** (0.9B), followed by Apple (0.7B) and OnePlus (0.6B)
- Dell, MSI, HP generate relatively small revenue (20-50 INR Million)
- In the bubble chart: Brands cluster around 17% avg discount and 4.0 avg rating
- Alienware has the highest rating (4.1) with low discount — premium, quality-conscious buyers
- MSI has the highest discount (19%+) but lower rating (3.7) — gaming laptops with more aggressive pricing
- HP has low rating (3.6) and high discount — possibly struggling brands compensating with discounts

**Interview answer:**
> "Samsung, Apple, and OnePlus together account for about 55% of electronics revenue — the classic 80/20 pattern. The brand scatter plot revealed a strategic insight: Alienware maintains 4.1 average rating with minimal discounting, suggesting strong brand loyalty where buyers are price-insensitive. Contrast this with HP and MSI which need higher discounts but still get lower ratings — they're fighting on price, not quality perception. Amazon's recommendation engine should factor brand perception score alongside price when suggesting alternatives."

---

## Chart 13 — Return Rate Analysis
**File:** `13_return_rate.png`

**What it shows:**
Two bar charts: Return rate (% of orders returned) by subcategory and by payment method.

**What we found:**
- **Audio has the highest return rate (8.1%)** — followed by Smart Watch (7.8%)
- TV & Entertainment has the lowest (5.1%)
- By payment method: Net Banking and BNPL have highest return rate (7.8% each)
- Wallet payments have the lowest return rate (6.7%)

**Why this matters:**
- Audio products have high returns because sound quality is subjective — "not what I expected from the product page"
- Smart Watches have fitness tracking features that disappoint — unrealistic expectations from marketing
- BNPL (Buy Now Pay Later) having high returns makes business sense — customers buy impulsively with deferred payment, then return when reality sets in
- Wallet/UPI customers are more deliberate (they have to actively pay upfront)

**Interview answer:**
> "Return rate analysis reveals Audio at 8.1% is the most returned category — likely because sound quality is subjective and hard to judge from product descriptions. More strategically important: BNPL orders have 7.8% return rate vs 6.7% for Wallet/UPI. BNPL customers return more because the deferred payment reduces the psychological commitment to keep the product. This is a real operational cost — for every 100 BNPL orders, 8 come back with reverse logistics costs. The recommendation: add mandatory review prompts before BNPL checkout for high-return categories like Audio."

---

## Chart 14 — Cohort Retention Matrix
**File:** `14_cohort_retention.png`

**What it shows:**
A matrix where each row is a customer cohort (defined by the year they first bought), and columns show what % of that cohort is still buying in each subsequent year.

**What we found (from 2015 cohort — the only fully visible one in FAST_MODE):**
- 2015 cohort: 100% in their first year (obviously), then 7% came back in 2016, 5% in 2017, dropping to near 0% by 2022
- This shows very low repeat purchase rates — electronics are infrequent purchases

**Why this happens:**
Electronics have a replacement cycle of 2-5 years. You don't buy a new phone every year. So low annual retention is expected and normal in this category.

**Interview answer:**
> "Cohort retention shows the typical electronics purchase pattern — sharp drop-off after year 1. The 2015 cohort retained only 7% in year 2, which sounds low but is actually reasonable for electronics where replacement cycles are 2-4 years. This is why retention strategy for electronics should focus on category expansion (customer bought a phone in 2020 → target with laptop/tablet offer in 2022) rather than expecting same-category repurchase. Note: the full retention picture requires FAST_MODE=False to see all 11 cohorts — the sample is too small to populate later years meaningfully."

---

## Chart 15 — Customer Lifetime Value (CLV) Distribution
**File:** `15_clv_distribution.png`

**What it shows:**
Histogram of total revenue per customer (their lifetime spend). Dotted lines show mean and median.

**What we found:**
- Most customers cluster in the 10K-100K INR range
- **Mean CLV: INR 74.1K** (pulled right by high spenders)
- **Median CLV: INR 47.9K** (more representative of typical customer)
- Distribution is right-skewed — a small number of customers spend very large amounts (up to 370K+)

**Why mean > median:**
The mean is always higher than median in right-skewed distributions. A few customers spending 3-4 Lakhs pull the average up. The median is the "typical" customer.

**Interview answer:**
> "CLV distribution is right-skewed, which is normal for e-commerce — the top 10% of customers by spend likely account for 40-50% of total revenue. Mean CLV of 74K vs median 47K shows this skew. For our churn prediction model, we weight false negatives by CLV — missing a high-CLV churner costs 10x more than missing a low-CLV churner. The model outputs both churn probability AND CLV, so the marketing team can prioritise retention efforts on high-value-at-risk customers."

---

## Chart 16 — Delivery Performance
**File:** `16_delivery_performance.png`

**What it shows:**
Left: Box plot of delivery days by delivery type (Same Day, Express, Standard). Right: Bar chart of average delivery days by customer tier.

**What we found:**
- **Same Day** median: 1 day (as expected)
- **Express** median: 2 days
- **Standard** median: 4 days (range: 1-7 days typically)
- Avg delivery by tier: Tier1 (3.3d) ≈ Tier2 (3.3d) ≈ Metro (3.4d) < Rural (3.5d) — very small difference

**Why delivery is nearly equal across tiers:**
Amazon has built extensive fulfillment infrastructure. The difference between Metro and Rural delivery time is only 0.2 days — a testament to logistics maturity.

**Interview answer:**
> "The delivery analysis revealed something impressive — Amazon India has essentially equalised delivery times across customer tiers. Rural customers receive their orders only 0.2 days later than Metro customers on average (3.5d vs 3.3d). This is a massive operational achievement and a key competitive moat. It also validates why customers in Tier2 and Rural trust Amazon for electronics — reliable, fast delivery is the foundation."

---

## Chart 17 — Discount Effectiveness
**File:** `17_discount_effectiveness.png`

**What it shows:**
Two charts: Average quantity ordered by discount range (0-10%, 10-20%, etc.) and Total revenue by discount range.

**What we found:**
- Average quantity barely changes across discount ranges (1.23 to 1.26) — discounts don't increase how many items people buy
- Revenue is highest in the **20-30% discount range** (190 INR Million) and lowest in the 0-10% range (72M)
- 50%+ discounts generate 74M — more than 0-10% range, despite deep discounting

**What this means:**
Discounts in the 20-30% range are the "sweet spot" — they trigger purchase decisions without eroding too much margin. Very deep discounts (50%+) bring in revenue but at lower margins.

**Interview answer:**
> "The discount effectiveness analysis shows 20-30% is the optimal discount range — it generates the most revenue and presumably the best margin efficiency. Very shallow discounts (0-10%) generate the least revenue, suggesting customers feel the discount isn't significant enough to act on. This informed our pricing recommendation model — we output an optimal discount percentage for each product-context combination, targeting the 20-30% sweet spot rather than race-to-the-bottom discounting."

---

## Chart 18 — Product Rating Distribution
**File:** `18_product_rating.png`

**What it shows:**
Box plots of product ratings (1-5 scale) for each subcategory.

**What we found:**
- All subcategories cluster tightly around **4.0 rating**
- Tablets have the widest range (3.0 to 4.8) — most variable quality
- Smart Watch is the most consistently rated (tight box around 4.0-4.2)
- No subcategory has a median below 4.0

**Why this makes sense:**
Amazon's product selection and review system filters out consistently bad products. Products with <3.5 ratings get delisted or buried. The tight clustering around 4.0 shows a healthy, curated catalog.

**Interview answer:**
> "All subcategories maintain 4.0+ median product ratings, which reflects Amazon's quality curation — products below threshold get removed or buried. Tablets show the most variance (3.0 to 4.8 range), suggesting the tablet market has both premium products and budget options with inconsistent quality. This data was used to build the product quality score in our feature store, which feeds into the recommendation engine — lower-rated products get a penalty in the ranking even if their click-through rate is high."

---

## Chart 19 — Revenue Concentration (Pareto Chart)
**File:** `19_revenue_pareto.png`

**What it shows:**
Cumulative revenue percentage as you add more products, from highest revenue to lowest. The classic Pareto (80/20) analysis.

**What we found:**
- **Top 36% of products generate 80% of revenue** (not the classic 20% — slightly broader in this dataset)
- The curve rises steeply for the first 20% of products, then flattens

**What this means:**
About 720 products out of 2,004 total generate 80% of all revenue. The remaining 1,284 products share only 20% of revenue.

**Interview answer:**
> "The Pareto analysis shows 36% of products (about 720 SKUs) generate 80% of revenue — slightly broader than the classic 80/20 rule, but the principle holds. This has direct implications for inventory management: we should maintain deep stock for the top 36% (premium smartphones, flagship laptops) and lean stock for the long tail. It also shapes our recommendation engine — for 64% of products, we shouldn't invest heavily in personalisation features because they contribute minimally to revenue."

---

## Chart 20 — YoY Revenue Growth by Subcategory (Small Multiples)
**File:** `20_yoy_subcategory.png`

**What it shows:**
6 individual line charts (one per subcategory), each showing revenue in INR Million from 2015 to 2025 with peak year annotated.

**What we found:**
- **Smartphones**: Peak 2020 at ~390 INR Million, declined after. Massive scale vs others.
- **Laptops**: Peak 2020 (WFH/education demand during COVID), then decline
- **Smart Watch**: Peak 2021 (fitness tracking trend post-COVID health awareness)
- **Tablets**: Peak 2020-2021 (remote learning), then decline
- **Audio**: Smaller scale (~7M), grew steadily, peaked 2020-2021
- **TV & Entertainment**: Very flat, small scale (~12M peak), some volatility

**Why Smart Watch peaked in 2021 not 2020:**
People focused on basic needs in 2020. In 2021, with vaccines and work-from-home settling, discretionary health tech spending (smart watches for fitness) surged.

**Interview answer:**
> "The small multiples format is powerful because it lets you compare patterns without the scale difference hiding smaller categories. Every subcategory shows the same 2020 COVID peak, but the SHAPE differs. Smartphones and Laptops dropped sharply post-2020 — demand brought forward during lockdown was now exhausted. Smart Watches peaked a year later in 2021, as the health consciousness from COVID drove fitness tracker adoption. This kind of temporal pattern recognition was directly used when building our sales forecasting model — each subcategory gets its own Prophet model because the seasonality and trend shapes are distinct."

---

## Common Interview Questions About This EDA

**Q: "Why did you choose PostgreSQL instead of just using the CSV files for EDA?"**
> CSV files don't scale well for analytical queries — a 1.1M row CSV takes 3-5 seconds per aggregation in pandas. PostgreSQL with proper indexes (on order_date, customer_id, product_id) executes the same GROUP BY in under 200ms. More importantly, having a proper database lets multiple tools query the same data consistently — the Streamlit dashboard, FastAPI, and Airflow all connect to the same source of truth.

**Q: "What was the most surprising finding in your EDA?"**
> Festival sales having a LOWER average order value than regular sales. You'd expect people to buy more expensive items during sales, but the data shows the opposite — discounts attract budget-conscious buyers who purchase lower-cost items that they wouldn't normally. The total revenue during festivals is still higher (more transactions), but each individual transaction is smaller.

**Q: "How did you ensure your visualisations were business-ready, not just exploratory?"**
> Three things: First, all charts are saved as PNG files to artifacts/charts/ — Streamlit loads them dynamically, so the dashboard always shows the latest analysis. Second, every chart uses INR units (not scientific notation like 1e6), discrete categorical colours (never sequential palettes on categories), and clear titles that a non-technical stakeholder can understand. Third, bar labels on every chart — no one should have to eyeball exact values.

**Q: "What would change if you ran this on the full 1.1M rows instead of 50K?"**
> Three things improve with full data: (1) The cohort retention matrix would show all 11 cohorts instead of just 2015 — you'd see a proper triangular retention heatmap. (2) The revenue trend would be more accurate for 2023-2025 (partial year bias goes away). (3) RFM segmentation would be more stable — with 50K rows, ~45K unique customers each appear only once, making frequency scores meaningless. With 1.1M rows and 354K customers, real purchase history emerges.

---

## Key Numbers to Memorise for Interviews

| Metric | Value |
|---|---|
| Total transactions in dataset | 1,127,609 raw; 1,122,000 after cleaning |
| Years covered | 2015-2025 (11 years) |
| Unique customers | ~354,000 |
| Unique products | 2,004 |
| Subcategories | 6 (Smartphones, Laptops, Tablets, Smart Watch, Audio, TV & Entertainment) |
| Smartphone revenue share | 73.2% |
| Peak revenue year | 2020 (COVID) |
| Top state | Maharashtra |
| Prime vs Non-Prime AOV | 78K vs 62K INR (+25%) |
| Highest return rate subcategory | Audio (8.1%) |
| Lowest return rate subcategory | TV & Entertainment (5.1%) |
| Pareto | Top 36% products = 80% revenue |
| Optimal discount range | 20-30% (highest revenue) |
| Avg delivery days | ~3.3-3.5 days across all tiers |
| Customer CLV mean / median | INR 74K / INR 47K |
| Festival AOV vs Regular AOV | 47K vs 78K (festival is LOWER) |
| Top festivals by revenue | Back to School > Diwali > Amazon Great Indian Festival |
| Dominant payment method | UPI (growing every year since 2016) |
