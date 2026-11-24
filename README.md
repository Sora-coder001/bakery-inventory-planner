# Café Inventory Planner

Weekly and daily purchase planning for a café, built on top of the team's demand forecast.

The client is anonymous. No business name, location, staff names or contact details appear here.
The raw materials, recipes, suppliers and stock levels in `planner_data/` are illustrative values
written for the project, not the client's own figures.

**The forecast output is not published.** `live_forecast.csv` and everything derived from it (the
dashboard and the two CSV outputs) hold the client's demand figures, so they are git-ignored. Put your
own `planner_data/live_forecast.csv` in place, or point the script at the database, and the outputs are
regenerated locally.

The planner answers one question: **given what we expect to sell next week and what is on the shelf
today, what should be bought, when, and in what pack sizes?** It writes CSV files, database tables and a
self-contained HTML dashboard for the website admin page.

This repository holds the inventory side only. The demand forecast is a separate model
(`demand_forecast.ipynb`, maintained by another member of the team); the planner reads its output and
never changes it.

---

## 1. How it works

```
forecast (per product, per week)        current stock (per raw material)
        |                                        |
        |  recipes: qty of each ingredient       |
        |  per one unit of a product             |
        v                                        v
   ingredient demand per day  ────────────────►  inventory position
                                                 (on hand + on order − what expires first)
                                                         |
                                     replenishment policy per material (A / B / C)
                                                         |
                                                         v
                                   packs to buy, order date, arrival date, supplier
```

1. Read the latest forecast run: units per product per week, with a low and a high.
2. Round product units to whole numbers (see 4.2) and spread each week over its trading days.
3. Multiply through the recipes to get a daily quantity of every raw material, converted to that
   material's stock unit.
4. Subtract what is on hand, allowing for expiry and for orders already placed.
5. Apply the material's replenishment policy to decide whether to order, and how much.
6. Repeat the whole thing three times: on the low, base and high forecast.

---

## 2. Running it

```bash
pip install pandas numpy scipy sqlalchemy "psycopg[binary]" psycopg2-binary

# CSV mode — reads planner_data/, writes planner_output/
python inventory_planner.py

# Database mode — reads and writes Supabase
DATABASE_URL="postgresql://user:password@host:5432/postgres" python inventory_planner.py
```

Useful environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | unset | Supabase connection string. Unset means CSV mode. |
| `FORECAST_FILE` | `planner_data/live_forecast.csv` | Forecast file in CSV mode. |
| `RUN_DATE` | today | Pretend the run is on another date, for testing. |

Scheduled daily by `.github/workflows/planner.yml`, at 11:00 UTC (06:00 in Indianapolis), after the
forecast and the stock counts are up to date. It needs a `DATABASE_URL` repository secret: the Supabase
pooler connection string from Settings - Database - Connection string. With that set, each run writes
`planner_product_need_next_week`, `planner_ingredient_need_next_week` and `planner_dashboard` (the page
itself) back to Supabase, replacing that day's rows and keeping earlier runs as history. The CSV files
are only produced when no database is configured.

### Settings at the top of the script

| Setting | Value | Meaning |
|---|---|---|
| `REVIEW_DAYS` | 7 | The plan covers the next 7 days. |
| `HORIZON_DAYS` | 28 | How far ahead demand is built, so lead times fit inside it. |
| `CAFE_DAYS` | Mon–Fri | Trading days. |
| `AUDIT_DAYS` | Mon, Thu | Class A stock counts. |
| `FORECAST_ERROR_FRAC` | 0.156 | Store-level weekly error (WAPE) from the forecast evaluation. |
| `DAILY_ERROR_FRAC` | 0.35 | Day-level item error, used only for same-day production. |
| `SERVICE_LEVEL` | A 95%, B 90%, C 90% | Chance of not running out within the window. |
| `WHOLE_UNITS` | `True` | Products are sold whole; round units up. |
| `STOCK_NOTE` | set | Footer note on the dashboard. Empty it once stock counts are real. |

---

## 3. Inputs

| What | Database | CSV | Required columns |
|---|---|---|---|
| Forecast | `forecast_weekly` + `forecast_runs` (latest run) | `live_forecast.csv` | `week_start, item, forecast_qty, low, high, rating[, holiday, expected_trading_days]` |
| Raw materials | `"Inventory"` | `Inventory.csv` | `unique_id, ingredient_name, unit, pack_qty, pack_size, shelf_life_days, lead_time_days, replenishment_model, supplier_channel, cost_per_unit_usd, current_stock` |
| Recipes | `recipes` | `recipes.csv` | `item, ingredient_id, qty_per_item, usage_unit` |
| Scanned stock | `ingredient_stock` view | `stock.csv` | `ingredient_id, qty, last_usable_date` |

Optional, skipped if absent: `open_orders`, `red_line`, `cake_orders`, `cake_recipes`, `same_day_items`.

## 4. Outputs

Written to `planner_output/` as CSV, or to the database as `planner_<name>` tables stamped with
`run_date` (today's rows are replaced, older runs are kept as history).

These files are git-ignored, because they contain the client's demand figures.

| Output | Contents |
|---|---|
| `dashboard.html` | The page for the admin site: by product, then by ingredient. |
| `product_need_next_week.csv` | Per product: how much of each ingredient it needs next week, low / base / high. |
| `ingredient_need_next_week.csv` | Per material: need next week, stock on hand, class, and packs to buy. |

---

## 5. Assumptions

Everything below is a choice, not a fact. Each one can be changed; most are one line in the script.

### 5.1 Demand and the forecast

- The forecast model is rerun weekly, and the planner always reads the **latest run** (in database mode,
  the most recent row in `forecast_runs`). Every plan is therefore built on fresh numbers.
- Those numbers are taken as given: the planner does not re-forecast, smooth, or override them. Its job
  starts where the forecast ends.
- Only the **top 18 café products** are forecast, so only their ingredients are planned. Anything else on
  the menu is invisible to the planner.
- Custom cake orders are forecast separately in dollars and are **not** converted to ingredients here.
  The `cake_orders` and `cake_recipes` tables exist for that but are empty, so cakes currently add nothing
  to the plan.
- A week's forecast is **spread evenly across that week's trading days**. There is no day-of-week pattern,
  so Friday and Tuesday are treated the same.
- `expected_trading_days` from the forecast decides how many days a week is open. Holiday weeks come
  through as 1 day (Thanksgiving, Christmas) or 0 days (New Year).
- The `rating` column is carried through for the reader (green = plan from the number, amber = plan from
  the range, red = use a par level) but **does not change the maths**. A red item is planned from its
  number like any other.
- Forecast error is assumed to be **15.6%** at store level for a week, taken from the forecast's own
  evaluation. It is treated as the same percentage for every product and every week.
- Safety stock covers **café demand only**. Cake orders are known in advance, so they get none.
- The planner stops with an error if the forecast does not cover the weeks it needs, rather than guessing.

### 5.2 Products are whole units

- A week's forecast is rounded **up** to a whole number, because half a cappuccino cannot be sold.
  The high is rounded up too and the low is rounded down, so the range does not narrow artificially.
- Ingredient quantities are **not** rounded. 0.5 stalk of celery stays 0.5.
- Rounding happens once, as the forecast is read, so the chart, the product view, the ingredient needs and
  the purchase plan all agree. On a quiet week this lifts demand: every product forecast at 0.1 becomes 1.

### 5.3 Inventory

- **Stock on hand** comes from barcode scans (`ingredient_stock`) where they exist, and from
  `Inventory.current_stock` for everything else. Scans win when both exist.
- **The current stock numbers in the database are made-up examples**, generated to exercise the model.
  They must be replaced with a real count before anyone orders from this.
- Stock with no expiry date is assumed to expire `shelf_life_days − 1` from the run date. In other words,
  everything unlabelled is treated as if it arrived today, which is optimistic.
- Shelf life in the raw material sheet is sometimes a range ("14–21 days"). The **lower** number is used;
  the original text is kept in `shelf_life_text`.
- Stock is consumed **first-expiry-first-out**. Real kitchens are not this disciplined.
- Stock that will expire inside the review window **does not count as cover**. This is why short-life
  items are still ordered when the shelf is not empty.
- There is no allowance for waste, spillage, theft or staff meals. Usage equals what the recipes say.
- Opened packs are treated as whole quantities. The shorter shelf life of an opened carton is ignored.
- Lead time is a fixed number of days per material and deliveries always arrive on time and in full.
- Orders are rounded **up to whole packs**, minimum one pack. No supplier minimum-order value is modelled.
- `cost_per_unit_usd` is empty in the current data, so the planner reports no spend.

### 5.4 Replenishment policies

Taken from the `Recommended Inventory Policy` column of the raw material master.

| Class | Policy | Materials | Rule |
|---|---|---|---|
| A | Periodic review (P-model) | 30 | Counted Mon and Thu, topped up to `T = (P + L) × d + S`, order `Q = max(0, T − I)` |
| B | Reorder point | 82 | Order when the inventory position falls to `ROP = d × L + SS` |
| C | Two-bin (17) or red line (7) | 24 | Visual check only |
| — | Make-to-order | 1 | Excluded from the weekly plan (edible image sheets) |

- `d` is average daily demand over the relevant window, not a long-run average, so it follows the forecast.
- **Class C materials are never ordered automatically.** They are only ordered when a row in the
  `red_line` table says the bin is exposed. That table does not exist yet, so class C currently never
  appears in the plan. Cups and lids are class B, so they are planned normally.
- Service levels are assumed, not measured: 95% for class A, 90% for B and C.
- The safety-stock formula is `z × 1.25 × 0.156 × (café demand over the window)`. The 1.25 is a rough
  allowance for the extra uncertainty of splitting a store-level forecast across items.

### 5.5 Recipes

- Recipe quantities are **estimates**, not measured yields, and most are marked `is_estimate = true`.
- Recipes come from the `Recipe_BOM` sheet of the raw material framework: 29 menu items, 97 lines.
  `Recipes_top18` holds all 29; the `recipes` table holds 27 items and 95 lines, keyed to the website's
  menu names.
- **Hot and iced are separate recipes** in `Recipes_top18`, but the forecast has one "Americano" and one
  "Latte". The `recipes` table therefore uses the **hot** version for both, because the POS records iced
  as an add-on. Iced drinks use a cold cup rather than a hot cup, and slightly less milk, so cup mix and
  milk are both slightly off.
- **Guava Paste Dulce de Leche** is one product in the forecast but two recipes in the BOM (guava pastry
  and dulce de leche pastry). Both are counted against it, so one sale consumes 2 sheets of puff pastry
  plus both fillings. If a sale is really one pastry, halve it.
- **Alfajor** has no recipe in the BOM sheet. Its six ingredients were carried over from the earlier
  recipe table, matched to the new material IDs by name, and converted from grams to ounces.
- Recipe names are matched to forecast names by an explicit mapping, including the misspelling
  "Capuccino" and "Empanada Cheesecake, onion & Cilantro". A product with no recipe is reported in
  the run log rather than silently dropped.
- No shared preparation or batching. If two products use the same filling, each is calculated separately,
  and a batch that must be made in a minimum size is not modelled.
- No yield loss: a 50 lb bag of flour gives 50 lb of usable flour.

### 5.6 Units

- Recipe units are converted to the material's stock unit using fixed factors: oz↔lb, g→oz/lb,
  fl oz↔gal, ml→fl oz.
- Volume and weight are **not** interchangeable. A recipe in fl oz against a material stocked in lb
  cannot be converted without a density, and is reported in the run log instead of guessed.
- **Known gap:** the Ham & Cheese Sandwich uses lettuce in oz, but lettuce is stocked by the head. It
  needs the weight of one head before it can be planned.

### 5.7 Identifiers

- Raw materials use the `RM-` IDs from the raw material master. `Inventory`, `ingredients`, `recipes` and
  `Recipes_top18` all use them. The earlier `ING-` IDs are gone.
- Backups of the pre-migration tables exist as `Inventory_backup_20261122`, `Recipes_top18_backup_20261122`,
  `ingredients_backup_20261122` and `recipes_backup_20261122`. Delete them once the new data is trusted.

---

## 6. Known limitations

- **Short shelf life versus the audit cycle.** A class A item counted on Thursday is topped up to cover
  until the next Monday count. For ciabatta and croissants, with a 3-day life, part of that order expires
  before it can be used, and the plan flags the waste rather than avoiding it. Fixing this means capping
  the order at what can be used before expiry.
- **Database mode is untested.** The SQL path is written but has only been run against CSV copies of the
  same tables.
- **No feedback loop.** Actual usage, waste and stock-outs are not fed back to adjust the model.
- **One location, one supplier per material.** No alternatives, substitutions or stock transfers.
- **The dashboard needs inline scripts.** It is one self-contained file, so a site with a strict
  Content-Security-Policy will block it. If that happens, split the CSS and JS into separate files, or
  allow them with a hash or nonce.

---

## 7. Files

```
inventory_planner.py                     the planner: settings, inputs, policies, outputs, dashboard
planner_data/                            inputs in CSV mode
  Inventory.csv                          137 raw materials, from the framework sheet
  recipes.csv                            95 recipe lines across 27 menu items
  live_forecast.csv                      not in the repo: supply your own forecast output
planner_output/                          not in the repo: written when the planner runs
cafe_inventory_framework.xlsx            source sheet: raw material master and recipe BOM
README.md                                this file
```

The demand forecast model (`demand_forecast.ipynb`) is maintained separately and is not in this
repository. The planner only reads its output.

The script is written in `# %%` cells, so it can be run top to bottom as a script or opened as a notebook.
