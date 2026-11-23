# %% [markdown]
# Inventory planner for a cafe
# Reads the forecast model's output (forecast_qty, low, high per menu item and week), the raw material master
# (Inventory), the recipes and the current stock, then runs the weekly purchase planner and the daily planner
# for the low, base and high forecast. Built to run once a day (see the scheduling notes at the end).
# Weekly policy follows the Inventory Models framework:
#   P-model items  -> Class A periodic review, audited Monday and Thursday: T = (P + L) x d + S, Q = max(0, T - I)
#   ROP items      -> Class B reorder point: ROP = d x L + SS
#   Two-bin / red line items -> Class C visual check, ordered only when flagged in red_line
#   Make-to-order items are left out of the weekly plan.
#
# Storage: if the environment variable DATABASE_URL is set (Supabase Postgres connection string), inputs are read
# from and outputs written to that database. Otherwise CSV files are used (folder planner_data in, planner_output out).
#
# Inputs (database table / CSV file):
#   forecast        forecast_weekly (latest run)  /  live_forecast.csv
#                   week_start, item, forecast_qty, low, high, rating[, holiday, expected_trading_days]
#   raw materials   "Inventory"  /  Inventory.csv
#                   unique_id, ingredient_name, unit, pack_qty, pack_size, shelf_life_days, lead_time_days,
#                   replenishment_model, supplier_channel, cost_per_unit_usd, current_stock
#   recipes         recipes  /  recipes.csv             item, ingredient_id, qty_per_item, usage_unit
#   stock           ingredient_stock view (packs from barcode scans), otherwise "Inventory".current_stock
#                   CSV: stock.csv (ingredient_id, qty, last_usable_date), otherwise Inventory.csv current_stock
#   optional        open_orders (ingredient_id, qty, arrival_date), red_line (ingredient_id, red_line_exposed),
#                   cake_orders (order_id, due_date, flavor, quantity), cake_recipes (flavor, ingredient_id, qty_per_cake),
#                   same_day_items (item, price, unit_cost, salvage)

# %% Cell 1: settings
import math
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_DIR = "planner_data"
OUT_DIR = "planner_output"
FORECAST_FILE = os.environ.get("FORECAST_FILE", os.path.join(DATA_DIR, "live_forecast.csv"))
DATABASE_URL = os.environ.get("DATABASE_URL")
RUN_DATE = pd.Timestamp(os.environ.get("RUN_DATE") or pd.Timestamp.now(tz="America/Indiana/Indianapolis").date())
TODAY = RUN_DATE                            # day the daily planner is run
PLAN_DATE = RUN_DATE                        # the weekly plan is a rolling 7 days starting today
REVIEW_DAYS = 7
HORIZON_DAYS = 28
FORECAST_ERROR_FRAC = 0.156                 # store-level weekly WAPE from the forecast evaluation
DAILY_ERROR_FRAC = 0.35                     # daily item-level error used for same-day production quantities
CAFE_DAYS = [0, 1, 2, 3, 4]                 # café trades Monday to Friday
AUDIT_DAYS = [0, 3]                         # Class A periodic review on Monday and Thursday
BAKE_DAYS_BEFORE_DUE = 1                    # cake ingredients are used this many days before the due date
SCENARIOS = ["low", "prediction", "high"]
WHOLE_UNITS = True                                # products are sold whole, so round each week's units up
STOCK_NOTE = "Stock levels are example values."   # shown on the dashboard; set to "" once real counts are in
POLICY_CLASS = {"P": "A", "ROP": "B", "TWOBIN": "C", "REDLINE": "C", "MTO": "MTO"}
SERVICE_LEVEL = {"A": 0.95, "B": 0.90, "C": 0.90}
UNIT_FACTOR = {                             # multiply a recipe quantity in unit [0] to get unit [1]
    ("oz", "lb"): 1 / 16, ("lb", "oz"): 16.0, ("g", "oz"): 1 / 28.3495, ("g", "lb"): 1 / 453.592,
    ("fl oz", "gal"): 1 / 128, ("gal", "fl oz"): 128.0, ("ml", "fl oz"): 1 / 29.5735,
}
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
print("Settings loaded. Run date:", RUN_DATE.date(), "| storage:", "database" if DATABASE_URL else "CSV files")

# %% Cell 2: inputs
if DATABASE_URL:
    from sqlalchemy import create_engine, inspect, text
    engine = create_engine(DATABASE_URL)


def read_sql(query):
    return pd.read_sql(text(query), engine)


def read_optional(name, columns):
    """Optional table or CSV; returns an empty frame with the given columns when it does not exist."""
    if DATABASE_URL:
        if inspect(engine).has_table(name):
            return read_sql('select ' + ", ".join(columns) + ' from "' + name + '"')
    else:
        path = os.path.join(DATA_DIR, name + ".csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def save(df, name):
    """Store one output table. Rows for today's run date are replaced, earlier days are kept as history."""
    df = df.assign(run_date=RUN_DATE.date())
    if DATABASE_URL:
        table = "planner_" + name
        with engine.begin() as conn:
            exists = inspect(conn).has_table(table)
            if exists:
                conn.execute(text('delete from "' + table + '" where run_date = :d'), {"d": str(RUN_DATE.date())})
        df.to_sql(table, engine, if_exists="append", index=False)
        if not exists:
            with engine.begin() as conn:
                conn.execute(text('alter table "' + table + '" enable row level security'))
    else:
        df.to_csv(os.path.join(OUT_DIR, name + ".csv"), index=False)


inv_cols = ["unique_id", "ingredient_name", "unit", "pack_qty", "pack_size", "shelf_life_days", "lead_time_days",
            "replenishment_model", "supplier_channel", "cost_per_unit_usd", "current_stock"]
if DATABASE_URL:
    forecast = read_sql("""select w.week_start, w.item, w.forecast_qty, w.low, w.high, w.rating
                           from forecast_weekly w
                           where w.run_id = (select r.id from forecast_runs r order by r.run_at desc limit 1)""")
    inventory = read_sql('select ' + ", ".join(inv_cols) + ' from "Inventory"')
    recipes = read_sql("select item, ingredient_id, qty_per_item, usage_unit from recipes")
    scanned = read_sql("select ingredient_id, packs_on_hand, next_expiry from ingredient_stock")
else:
    forecast = pd.read_csv(FORECAST_FILE)
    inventory = pd.read_csv(os.path.join(DATA_DIR, "Inventory.csv"))[inv_cols]
    recipes = pd.read_csv(os.path.join(DATA_DIR, "recipes.csv"))
    scanned = read_optional("stock", ["ingredient_id", "qty", "last_usable_date"])

forecast = forecast.rename(columns={"item": "menu_item", "forecast_qty": "prediction"})
forecast["week_start"] = pd.to_datetime(forecast["week_start"])
if "expected_trading_days" not in forecast:
    forecast["expected_trading_days"] = len(CAFE_DAYS)
if "holiday" not in forecast:
    forecast["holiday"] = ""
forecast["expected_trading_days"] = forecast["expected_trading_days"].fillna(len(CAFE_DAYS)).astype(int)
forecast["holiday"] = forecast["holiday"].fillna("")
if WHOLE_UNITS:
    forecast["prediction"] = np.maximum(np.ceil(forecast["prediction"].astype(float) - 1e-9), 0.0)
    forecast["high"] = np.maximum(np.ceil(forecast["high"].astype(float) - 1e-9), 0.0)
    forecast["low"] = np.maximum(np.floor(forecast["low"].astype(float) + 1e-9), 0.0)

open_orders = read_optional("open_orders", ["ingredient_id", "qty", "arrival_date"])
open_orders["arrival_date"] = pd.to_datetime(open_orders["arrival_date"])
red_line_df = read_optional("red_line", ["ingredient_id", "red_line_exposed"])
red_line = dict(zip(red_line_df["ingredient_id"], red_line_df["red_line_exposed"].astype(bool)))
cake_orders = read_optional("cake_orders", ["order_id", "due_date", "flavor", "quantity"])
cake_orders["due_date"] = pd.to_datetime(cake_orders["due_date"])
cake_recipes = read_optional("cake_recipes", ["flavor", "ingredient_id", "qty_per_cake"])
same_day_items = read_optional("same_day_items", ["item", "price", "unit_cost", "salvage"])

# stop the run if the forecast does not cover the weeks needed for the review window plus the longest lead time
window_end = RUN_DATE + pd.Timedelta(days=REVIEW_DAYS + int(inventory["lead_time_days"].max()) - 1)
needed_weeks = sorted({(d - pd.Timedelta(days=d.weekday())) for d in pd.date_range(RUN_DATE, window_end)
                       if d.weekday() in CAFE_DAYS})
missing = [w.date() for w in needed_weeks if w not in set(forecast["week_start"])]
if missing:
    raise SystemExit("Forecast output has no rows for week(s) " + ", ".join(map(str, missing)) +
                     ". Rerun the forecast model first.")
print("Loaded:", forecast["menu_item"].nunique(), "forecast items over", forecast["week_start"].nunique(), "weeks |",
      len(inventory), "raw materials |", len(recipes), "recipe lines |", len(cake_orders), "cake orders")

# %% Cell 3: preprocessing - raw material table, current stock, recipe units, daily requirements per scenario
ingredients = pd.DataFrame({
    "ingredient": inventory["unique_id"],
    "name": inventory["ingredient_name"],
    "inventory_class": inventory["replenishment_model"].map(POLICY_CLASS),
    "unit": inventory["unit"],
    "pack_size": inventory["pack_qty"].astype(float),
    "pack_label": inventory["pack_size"],
    "shelf_life_days": inventory["shelf_life_days"].astype(int),
    "lead_time_days": inventory["lead_time_days"].astype(int),
    "supplier": inventory["supplier_channel"],
    "cost_per_pack": pd.to_numeric(inventory["cost_per_unit_usd"], errors="coerce").fillna(0.0),
    "min_packs": 1,
})
ingredients["service_level"] = ingredients["inventory_class"].map(SERVICE_LEVEL)
mto = ingredients[ingredients["inventory_class"] == "MTO"]
ingredients = ingredients[ingredients["inventory_class"] != "MTO"].reset_index(drop=True)
ing_info = ingredients.set_index("ingredient")

# current stock: scanned packs where an ingredient has scans, otherwise current_stock from the raw material table
if DATABASE_URL:
    scanned = scanned.merge(ingredients[["ingredient", "pack_size", "shelf_life_days"]],
                            left_on="ingredient_id", right_on="ingredient")
    scanned["qty"] = scanned["packs_on_hand"].astype(float) * scanned["pack_size"]
    scanned["last_usable_date"] = pd.to_datetime(scanned["next_expiry"])
scanned = scanned[["ingredient_id", "qty", "last_usable_date"]]
manual = inventory.loc[~inventory["unique_id"].isin(scanned["ingredient_id"]), ["unique_id", "current_stock"]]
manual = manual.rename(columns={"unique_id": "ingredient_id", "current_stock": "qty"})
manual["last_usable_date"] = pd.NaT
stock_lots = pd.concat([scanned, manual], ignore_index=True)
stock_lots["qty"] = pd.to_numeric(stock_lots["qty"], errors="coerce").fillna(0.0)
stock_lots = stock_lots[(stock_lots["qty"] > 0) & stock_lots["ingredient_id"].isin(ing_info.index)].copy()
no_date = stock_lots["last_usable_date"].isna()
stock_lots.loc[no_date, "last_usable_date"] = [RUN_DATE + pd.Timedelta(days=int(ing_info.loc[i, "shelf_life_days"]) - 1)
                                               for i in stock_lots.loc[no_date, "ingredient_id"]]
stock_lots["last_usable_date"] = pd.to_datetime(stock_lots["last_usable_date"])
stock_lots = stock_lots.rename(columns={"ingredient_id": "ingredient"})

# recipe quantities converted to the raw material's stock unit
bom = recipes.merge(ingredients[["ingredient", "unit"]], left_on="ingredient_id", right_on="ingredient", how="left")
bom["factor"] = [1.0 if u == s else UNIT_FACTOR.get((u, s), np.nan) for u, s in zip(bom["usage_unit"], bom["unit"])]
unconverted = bom[bom["factor"].isna() & bom["unit"].notna()]
no_material = bom[bom["unit"].isna()]
bom = bom.dropna(subset=["factor", "unit"])
bom["qty_per_unit"] = bom["qty_per_item"].astype(float) * bom["factor"]
cafe_bom = bom.rename(columns={"item": "menu_item"})[["menu_item", "ingredient", "qty_per_unit"]]
no_recipe = sorted(set(forecast["menu_item"]) - set(cafe_bom["menu_item"]))
print("Stock on hand for", stock_lots["ingredient"].nunique(), "of", len(ingredients), "raw materials")
print("Recipe lines with a unit that cannot be converted:", len(unconverted),
      "| forecast items without a recipe:", no_recipe if no_recipe else "none")

days = pd.date_range(PLAN_DATE, periods=HORIZON_DAYS, freq="D")


def build_requirements(forecast_col):
    """Daily raw material requirements using one forecast column (low, prediction or high).
    Each week's forecast is spread evenly over that week's expected trading days."""
    cafe_rows = []
    for f in forecast.itertuples():
        open_days = CAFE_DAYS[:max(0, min(int(f.expected_trading_days), len(CAFE_DAYS)))]
        for d in open_days:
            cafe_rows.append((f.week_start + pd.Timedelta(days=d), f.menu_item, getattr(f, forecast_col) / len(open_days)))
    cafe_items = pd.DataFrame(cafe_rows, columns=["date", "menu_item", "units"])
    cafe_daily = (cafe_items.merge(cafe_bom, on="menu_item")
                  .assign(qty=lambda x: x.units * x.qty_per_unit)
                  .groupby(["date", "ingredient"], as_index=False)["qty"].sum().rename(columns={"qty": "cafe_qty"}))
    cake_daily = (cake_orders.merge(cake_recipes.rename(columns={"ingredient_id": "ingredient"}), on="flavor")
                  .assign(qty=lambda x: x.quantity * x.qty_per_cake,
                          date=lambda x: x.due_date - pd.Timedelta(days=BAKE_DAYS_BEFORE_DUE))
                  .groupby(["date", "ingredient"], as_index=False)["qty"].sum().rename(columns={"qty": "cake_qty"}))
    grid = pd.MultiIndex.from_product([days, ingredients["ingredient"]], names=["date", "ingredient"]).to_frame(index=False)
    req = (grid.merge(cafe_daily, on=["date", "ingredient"], how="left")
               .merge(cake_daily, on=["date", "ingredient"], how="left"))
    req[["cafe_qty", "cake_qty"]] = req[["cafe_qty", "cake_qty"]].astype(float).fillna(0.0)
    req["total_qty"] = req["cafe_qty"] + req["cake_qty"]
    return req


requirements = {s: build_requirements(s) for s in SCENARIOS}
week_need = pd.DataFrame({s: requirements[s][requirements[s]["date"] < PLAN_DATE + pd.Timedelta(days=REVIEW_DAYS)]
                         .groupby("ingredient")["total_qty"].sum() for s in SCENARIOS})
week_need = week_need[week_need["prediction"] > 0].join(ing_info[["name", "unit"]]).round(2)
print("\nRaw material need for the next", REVIEW_DAYS, "days, by scenario:")
print(week_need[["name", "unit"] + SCENARIOS].to_string())

def product_week_need(col):
    """Ingredient quantity per forecast week, kept against the product that drives it."""
    units = forecast[["week_start", "menu_item", col]].rename(columns={"menu_item": "product"})
    lines = units.merge(cafe_bom, left_on="product", right_on="menu_item")
    lines[col] = lines[col] * lines["qty_per_unit"]
    return lines[["week_start", "product", "ingredient", col]]


product_need = None
for s_name in SCENARIOS:
    l = product_week_need(s_name)
    product_need = l if product_need is None else product_need.merge(l, on=["week_start", "product", "ingredient"], how="outer")
product_need = (product_need.fillna(0.0)
                .merge(ing_info[["name", "unit"]], left_on="ingredient", right_index=True, how="left")
                .rename(columns={"ingredient": "ingredient_id"}))
product_need["week_start"] = product_need["week_start"].dt.date
product_need = product_need[["week_start", "product", "ingredient_id", "name", "unit"] + SCENARIOS].round(2)
print("\nIngredient need per product, week of", plan_week_label := str(product_need["week_start"].min()), "(base forecast, top rows):")
print(product_need[product_need["week_start"] == product_need["week_start"].min()]
      .sort_values("prediction", ascending=False).head(10).to_string(index=False))

# %% Cell 4: policy functions (Class A periodic review, Class B reorder point, Class C red line)
def simulate(lots, arrivals, demand, start, n_days):
    """First-expiry-first-out simulation. lots are [last_usable_date, qty]; arrivals are [last_usable_date, qty, arrival_date].
    Returns per-day shortage, expired quantity and ending stock."""
    lots = sorted([list(l) for l in lots], key=lambda l: l[0])
    arrivals = sorted([list(a) for a in arrivals], key=lambda a: a[0])
    shortage, expired, ending = [], [], []
    for i in range(n_days):
        day = start + pd.Timedelta(days=i)
        for a in arrivals:
            if a[2] == day:
                lots.append([a[0], a[1]])
        lots = sorted(lots, key=lambda l: l[0])
        exp = sum(l[1] for l in lots if l[0] < day)
        lots = [l for l in lots if l[0] >= day]
        need = demand.get(day, 0.0)
        for l in lots:
            take = min(l[1], need)
            l[1] -= take
            need -= take
        lots = [l for l in lots if l[1] > 1e-9]
        shortage.append(need)
        expired.append(exp)
        ending.append(sum(l[1] for l in lots))
    return np.array(shortage), np.array(expired), np.array(ending)


def arrival_entry(ing, arrival_date, qty):
    """Arrival as [last_usable_date, qty, arrival_date]."""
    return [arrival_date + pd.Timedelta(days=int(ing.shelf_life_days) - 1), qty, arrival_date]


def open_arrivals(ing, orders_df):
    o = orders_df[orders_df["ingredient_id"] == ing.ingredient]
    return [arrival_entry(ing, r.arrival_date, float(r.qty)) for r in o.itertuples()]


def position_on(day, lots, arrivals, demand, start):
    """Inventory position the morning of `day`: projected on-hand plus orders still in transit."""
    i = (day - start).days
    on_hand = sum(q for d, q in lots if d >= day) if i == 0 else simulate(lots, arrivals, demand, start, i)[2][-1]
    in_transit = sum(a[1] for a in arrivals if a[2] >= day)
    return on_hand + in_transit


def safety_stock(ing, cafe, start, days=REVIEW_DAYS):
    """Safety stock on the café share of demand only; cake orders are known."""
    cafe_week = sum(cafe.get(start + pd.Timedelta(days=i), 0.0) for i in range(days))
    return norm.ppf(ing.service_level) * 1.25 * FORECAST_ERROR_FRAC * cafe_week


def avg_daily_demand(demand, start, n):
    return sum(demand.get(start + pd.Timedelta(days=i), 0.0) for i in range(n)) / n


def p_model_order(ing, day, lots, arrivals, demand, cafe, start):
    """Class A: audited on `day`; order up to T = (P + L) x d + S, with Q = max(0, T - I). I excludes stock that expires inside the window."""
    next_audit = [day + pd.Timedelta(days=k) for k in range(1, 8) if (day + pd.Timedelta(days=k)).weekday() in AUDIT_DAYS][0]
    p, lead = (next_audit - day).days, int(ing.lead_time_days)
    d = avg_daily_demand(demand, day, p + lead)
    s = safety_stock(ing, cafe, day)
    target = (p + lead) * d + s
    position = position_on(day, lots, arrivals, demand, start)
    i = (day - start).days
    expiring = simulate(lots, arrivals, demand, start, i + p + lead)[1][i:].sum()
    q = max(0.0, target - (position - expiring))
    packs = max(int(ing.min_packs), math.ceil(q / ing.pack_size - 1e-9)) if q > 1e-9 else 0
    return packs, {"P": p, "d": round(float(d), 2), "S": round(float(s), 1), "T": round(float(target), 1),
                   "I": round(float(position), 1), "expiring_in_window": round(float(expiring), 1)}


def rop_order(ing, day, lots, arrivals, demand, cafe, start):
    """Class B: ROP = d x L + SS. Order one pack at a time while the inventory position is at or below ROP."""
    lead = int(ing.lead_time_days)
    d = avg_daily_demand(demand, day, REVIEW_DAYS)
    rop = d * lead + safety_stock(ing, cafe, day)
    position = position_on(day, lots, arrivals, demand, start)
    packs, p0 = 0, position
    if d <= 1e-9:
        return 0, {"d": 0.0, "ROP": 0.0, "I": round(float(p0), 1)}
    while position <= rop + 1e-9:
        packs += int(ing.min_packs)
        position += int(ing.min_packs) * ing.pack_size
    return packs, {"d": round(float(d), 2), "ROP": round(float(rop), 1), "I": round(float(p0), 1)}
print("Policy functions defined")

# %% Cell 5: weekly planner for the low, base and high forecast
def plan_week(ing, lots_df, orders_df, req_df, red_line_flags, scenario):
    lots = [[r.last_usable_date, r.qty] for r in lots_df[lots_df["ingredient"] == ing.ingredient].itertuples()]
    r = req_df[req_df["ingredient"] == ing.ingredient].set_index("date")
    demand, cafe = r["total_qty"].to_dict(), r["cafe_qty"].to_dict()
    lead = int(ing.lead_time_days)
    arrivals = open_arrivals(ing, orders_df)
    rows = []
    for k in range(REVIEW_DAYS):
        day = PLAN_DATE + pd.Timedelta(days=k)
        packs, detail, rule = 0, {}, ""
        if ing.inventory_class == "A" and day.weekday() in AUDIT_DAYS:
            packs, detail = p_model_order(ing, day, lots, arrivals, demand, cafe, PLAN_DATE)
            rule = "P-model audit"
        elif ing.inventory_class == "B":
            packs, detail = rop_order(ing, day, lots, arrivals, demand, cafe, PLAN_DATE)
            rule = "ROP trigger"
        elif ing.inventory_class == "C" and k == 0 and red_line_flags.get(ing.ingredient, False):
            packs, rule = int(ing.min_packs), "Red line exposed"
        if packs > 0:
            arrival = day + pd.Timedelta(days=lead)
            arrivals.append(arrival_entry(ing, arrival, packs * ing.pack_size))
            rows.append({"scenario": scenario, "ingredient": ing.ingredient, "name": ing.name,
                         "class": ing.inventory_class, "supplier": ing.supplier, "order_date": day.date(),
                         "arrival_date": arrival.date(), "packs": packs, "pack_label": ing.pack_label,
                         "order_qty": round(packs * ing.pack_size, 1), "unit": ing.unit,
                         "cost": round(packs * ing.cost_per_pack, 2), "rule": rule, "inputs": detail})
    s_chk, e_chk, _ = simulate(lots, arrivals, demand, PLAN_DATE, REVIEW_DAYS + lead)
    flags = []
    if s_chk.sum() > 1e-6:
        flags.append("SHORTAGE " + str(round(s_chk.sum(), 1)))
    if e_chk.sum() > 1e-6:
        flags.append("EXPIRY " + str(round(e_chk.sum(), 1)))
    if ing.shelf_life_days < REVIEW_DAYS:
        flags.append("SHELF LIFE UNDER 7 DAYS")
    if not rows:
        rows.append({"scenario": scenario, "ingredient": ing.ingredient, "name": ing.name,
                     "class": ing.inventory_class, "supplier": ing.supplier, "order_date": None,
                     "arrival_date": None, "packs": 0, "pack_label": ing.pack_label, "order_qty": 0,
                     "unit": ing.unit, "cost": 0.0,
                     "rule": "visual check (two-bin / red line)" if ing.inventory_class == "C" else "no order needed",
                     "inputs": {}})
    for row in rows:
        row["flags"] = ", ".join(flags)
    return rows


plans = pd.DataFrame([row for s in SCENARIOS for ing in ingredients.itertuples()
                      for row in plan_week(ing, stock_lots, open_orders, requirements[s], red_line, s)])
print("Weekly plans built for scenarios:", ", ".join(SCENARIOS), "| rows:", len(plans))

# %% Cell 6: output tables - forecast for the plan weeks and purchase plan with low, base, high
plan_weeks = sorted({d - pd.Timedelta(days=d.weekday()) for d in pd.date_range(PLAN_DATE, periods=REVIEW_DAYS)})
item_forecast = forecast[forecast["week_start"].isin(plan_weeks)][
    ["week_start", "menu_item", "low", "prediction", "high", "rating", "holiday", "expected_trading_days"]].copy()
item_forecast["week_start"] = item_forecast["week_start"].dt.date
print("\nForecast used for the plan window (units per week):")
print(item_forecast.to_string(index=False))

by_ing = (plans.pivot_table(index=["class", "ingredient", "name", "unit", "pack_label"], columns="scenario",
                            values=["packs", "order_qty", "cost"], aggfunc="sum", fill_value=0))
by_ing = by_ing.reindex(columns=pd.MultiIndex.from_product([["packs", "order_qty", "cost"], SCENARIOS]), fill_value=0)
by_ing.columns = [m + "_" + s for m, s in by_ing.columns]
by_ing = by_ing.reset_index()
to_buy = by_ing[by_ing[["packs_low", "packs_prediction", "packs_high"]].sum(axis=1) > 0]
print("\nPurchase plan for the", REVIEW_DAYS, "days from", PLAN_DATE.date(), "(low / base / high forecast):")
print(to_buy[["class", "ingredient", "name", "pack_label", "packs_low", "packs_prediction", "packs_high",
              "order_qty_low", "order_qty_prediction", "order_qty_high", "unit"]].to_string(index=False))

if ingredients["cost_per_pack"].sum() > 0:
    by_class = by_ing.groupby("class")[["cost_low", "cost_prediction", "cost_high"]].sum().round(2)
    print("\nPlanned spend by inventory class ($):")
    print(by_class.to_string())
else:
    print("\nPlanned spend: no cost data in the raw material table yet")

base = plans[(plans["scenario"] == "prediction") & (plans["packs"] > 0)]
print("\nPurchase runs for the base forecast:")
print(base.groupby(["supplier", "order_date", "arrival_date"])
      .agg(items=("name", lambda s: ", ".join(s)), packs=("packs", "sum")).reset_index().to_string(index=False))
flags = plans[(plans["scenario"] == "prediction") & (plans["flags"] != "")][["ingredient", "name", "flags"]].drop_duplicates()
print("\nPlan flags (base forecast):")
print(flags.to_string(index=False) if len(flags) else "none")

data_gaps = pd.DataFrame(
    [{"type": "unit cannot be converted", "detail": r.item + " uses " + str(r.qty_per_item) + " " + r.usage_unit +
      " of " + ing_info.loc[r.ingredient_id, "name"] + " (stocked in " + r.unit + ")"} for r in unconverted.itertuples()] +
    [{"type": "recipe ingredient not in raw materials", "detail": r.item + ": " + r.ingredient_id} for r in no_material.itertuples()] +
    [{"type": "forecast item without a recipe", "detail": i} for i in no_recipe] +
    [{"type": "make-to-order, not planned", "detail": r.name} for r in mto.itertuples()])
print("\nData gaps:")
print(data_gaps.to_string(index=False) if len(data_gaps) else "none")

print("\nWeekly plan built for", RUN_DATE.date())

# %% Cell 7: daily planner using today's stock and the base forecast plus this week's planned orders
weekly_orders = base[["ingredient", "order_qty", "arrival_date"]].rename(columns={"ingredient": "ingredient_id",
                                                                                   "order_qty": "qty"})
weekly_orders["arrival_date"] = pd.to_datetime(weekly_orders["arrival_date"])
orders_all = pd.concat([open_orders, weekly_orders], ignore_index=True)
req_base = requirements["prediction"]


def plan_daily(ing, lots_df, orders_df, req_df, today, red_line_flags):
    lots = [[r.last_usable_date, r.qty] for r in lots_df[lots_df["ingredient"] == ing.ingredient].itertuples()]
    r = req_df[req_df["ingredient"] == ing.ingredient].set_index("date")
    demand, cafe = r["total_qty"].to_dict(), r["cafe_qty"].to_dict()
    lead = int(ing.lead_time_days)
    arrivals = open_arrivals(ing, orders_df)
    short, expired, _ = simulate(lots, arrivals, demand, today, lead + REVIEW_DAYS)
    packs, rule = 0, "no policy trigger today"
    if ing.inventory_class == "A":
        if today.weekday() in AUDIT_DAYS:
            packs, _ = p_model_order(ing, today, lots, arrivals, demand, cafe, today)
            rule = "P-model audit day"
        else:
            rule = "not an audit day"
    elif ing.inventory_class == "B":
        packs, _ = rop_order(ing, today, lots, arrivals, demand, cafe, today)
        rule = "ROP check"
    elif ing.inventory_class == "C":
        packs = int(ing.min_packs) if red_line_flags.get(ing.ingredient, False) else 0
        rule = "red line exposed" if packs else "visual check"
    actions = []
    if packs > 0:
        actions.append("ORDER " + str(packs) + " x " + str(ing.pack_label) + ", arrives " +
                       str((today + pd.Timedelta(days=lead)).date()))
    short_days = np.where(short > 1e-6)[0]
    if len(short_days) > 0 and int(short_days[0]) < lead:
        actions.append("SHORT BEFORE AN ORDER CAN ARRIVE")
    elif len(short_days) > 0 and packs == 0:
        actions.append("PROJECTED SHORTAGE: consider a top-up order")
    if expired.sum() > 1e-6:
        actions.append("WASTE RISK " + str(round(expired.sum(), 1)) + " " + ing.unit)
    use_first = sum(q for d, q in lots if today <= d <= today + pd.Timedelta(days=3))
    if use_first > 1e-6 and ing.inventory_class == "A":
        actions.append("USE FIRST " + str(round(use_first, 1)))
    return {"ingredient": ing.ingredient, "name": ing.name, "class": ing.inventory_class, "rule": rule,
            "usable_now": round(sum(q for d, q in lots if d >= today), 1), "unit": ing.unit,
            "need_today": round(demand.get(today, 0.0), 1), "action": "; ".join(actions) if actions else "OK"}


daily = pd.DataFrame([plan_daily(ing, stock_lots, orders_all, req_base, TODAY, red_line) for ing in ingredients.itertuples()])
daily_used = daily[(daily["action"] != "OK") | (daily["need_today"] > 0)]
print("\nDaily action list for", TODAY.date(), "(raw materials with a need or an action)")
print(daily_used.to_string(index=False) if len(daily_used) else "nothing to do today")

# %% Cell 8: same-day production quantities (newsvendor) for the low, base and high forecast
def production_quantity(row, units_week, trading_days):
    mu = units_week / trading_days
    cu, co = row.price - row.unit_cost, row.unit_cost - row.salvage
    ratio = cu / (cu + co)
    return max(0, round(mu + norm.ppf(ratio) * DAILY_ERROR_FRAC * mu))


this_week = forecast[forecast["week_start"] == TODAY - pd.Timedelta(days=TODAY.weekday())]
if len(same_day_items) == 0:
    print("\nSame-day production: no same_day_items table (item, price, unit_cost, salvage) yet")
elif TODAY.weekday() not in CAFE_DAYS or len(this_week) == 0 or int(this_week["expected_trading_days"].iloc[0]) == 0:
    print("\nCafe is closed on", TODAY.date(), "- no production quantities")
else:
    rows = []
    for r in same_day_items.itertuples():
        f = this_week[this_week["menu_item"] == r.item]
        if len(f) == 0:
            continue
        f, n = f.iloc[0], int(this_week["expected_trading_days"].iloc[0])
        rows.append({"item": r.item, "date": TODAY.date(), "make_low": production_quantity(r, f.low, n),
                     "make_base": production_quantity(r, f.prediction, n), "make_high": production_quantity(r, f.high, n)})
    production = pd.DataFrame(rows)
    print("\nProduction quantities for", TODAY.date())
    print(production.to_string(index=False))

# %% Cell 9: dashboard - one self-contained HTML page (no outside scripts or fonts) for the website admin page
import json

# page template; __DATA__ is replaced with this run's data

DASHBOARD_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inventory Plan</title>
<style>
.mcd{
  color-scheme: light;
  --brand:#bd3739;
  --page:#f8f1ef; --surface:#fffdfc; --ink:#1b1413; --ink-2:#5b4f4d; --muted:#8c7f7c;
  --grid:#ece0dd; --axis:#d8cac6; --ring:rgba(27,20,19,0.12); --wash:rgba(189,55,57,0.06);
  --series:#bd3739; --series-hover:#a32d2f; --range:#5b4f4d;
  --good:#0ca30c; --warning:#fab219; --critical:#7a1416; --on-warning:#1b1413;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--page);
  font-size: 14px; line-height: 1.45; padding: 24px 16px 40px; box-sizing: border-box; min-height: 100%;
}
/* Dark mode is opt-in: set data-theme="dark" on <html>. The OS setting is not followed. */
:root[data-theme="dark"] .mcd{
  color-scheme: dark;
  --page:#120e0e; --surface:#1f1817; --ink:#f5efed; --ink-2:#cbbdba; --muted:#9a8d8a;
  --grid:#332827; --axis:#423534; --ring:rgba(255,255,255,0.12); --wash:rgba(255,255,255,0.06);
  --series:#d9565a; --series-hover:#e4696d; --range:#cbbdba; --critical:#ffa3a5;
}
html, body{margin:0; background:#f8f1ef;}
html[data-theme="dark"], html[data-theme="dark"] body{background:#120e0e;}
.mcd *{box-sizing:border-box;}
.mcd-wrap{max-width:1100px; margin:0 auto;}
.mcd h1{font-size:22px; font-weight:600; margin:0; letter-spacing:-0.01em;}
.mcd h2{font-size:15px; font-weight:600; margin:0;}
.mcd-sub{color:var(--ink-2); margin-top:4px;}
.mcd-head{margin-bottom:16px; background:var(--brand); color:#ffffff; border-radius:12px; padding:16px 18px;}
.mcd-head .mcd-sub{color:rgba(255,255,255,0.88);}
.mcd-badge{display:inline-flex; align-items:center; font-size:12px; font-weight:600; padding:3px 9px; border-radius:999px;
  background:rgba(255,255,255,0.18); color:#ffffff; border:1px solid rgba(255,255,255,0.4); margin-left:8px; vertical-align:3px;}
.mcd-filters{display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; margin:0 0 16px;}
.mcd-filters .lab{color:var(--ink-2); font-size:13px;}
.mcd-seg{display:inline-flex; background:var(--surface); border:1px solid var(--ring); border-radius:9px; padding:2px;}
.mcd-seg button{appearance:none; border:0; background:transparent; color:var(--ink-2); font:inherit; font-size:13px; padding:5px 12px;
  border-radius:7px; cursor:pointer;}
.mcd-seg button:hover{background:var(--wash); color:var(--ink);}
.mcd-seg button[aria-pressed="true"]{background:var(--series); color:#ffffff; font-weight:600;}
.mcd-seg button:focus-visible, .mcd-btn:focus-visible, .mcd-chips button:focus-visible{outline:2px solid var(--series); outline-offset:2px;}
.mcd-card{background:var(--surface); border:1px solid var(--ring); border-radius:12px; padding:16px; margin-bottom:16px; min-width:0;}
.mcd-card-h{display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-start; gap:8px 12px; margin-bottom:10px;}
.mcd-card-h p{margin:3px 0 0; color:var(--ink-2); font-size:13px;}
.mcd-btn{appearance:none; font:inherit; font-size:12px; color:var(--ink-2); background:transparent; border:1px solid var(--ring);
  border-radius:7px; padding:4px 10px; cursor:pointer;}
.mcd-btn:hover{background:var(--wash); color:var(--ink);}
.mcd-chips{display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; align-items:center;}
.mcd-chips .lab{color:var(--ink-2); font-size:12px; margin-right:2px;}
.mcd-chips > span[id$="-wrap"]{display:inline-flex; flex-wrap:wrap; gap:6px;}
.mcd-chips button{appearance:none; font:inherit; font-size:12px; border:1px solid var(--ring); background:transparent; color:var(--ink-2);
  border-radius:999px; padding:3px 10px; cursor:pointer;}
.mcd-chips button:hover{background:var(--wash);}
.mcd-chips button[aria-pressed="true"]{border-color:var(--series); color:var(--series); background:var(--wash); font-weight:600;}
.mcd-key{display:flex; gap:16px; flex-wrap:wrap; color:var(--ink-2); font-size:12px; margin:2px 0 8px;}
.mcd-key span{display:inline-flex; align-items:center; gap:6px;}
.mcd-chart{position:relative; width:100%;}
.mcd-chart svg{display:block; width:100%; height:auto; overflow:visible;}
.mcd-chart .bar{fill:var(--series);}
.mcd-chart .hit{fill:transparent;}
.mcd-chart g.row:hover .bar, .mcd-chart g.row:focus .bar{fill:var(--series-hover);}
.mcd-chart g.row:focus{outline:none;}
.mcd-chart g.row:focus .hit{stroke:var(--series); stroke-width:1;}
.mcd-tip{position:absolute; pointer-events:none; background:var(--surface); color:var(--ink); border:1px solid var(--ring);
  box-shadow:0 4px 16px rgba(0,0,0,0.12); border-radius:8px; padding:8px 10px; font-size:12px; min-width:180px; z-index:5; display:none;}
.mcd-tip .t{color:var(--ink-2); margin-bottom:4px;}
.mcd-tip .row{display:flex; justify-content:space-between; gap:14px;}
.mcd-tip .row b{font-weight:600; font-variant-numeric:tabular-nums;}
.mcd-tip .row span{color:var(--ink-2);}
.mcd-prods{display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:10px; margin-top:14px;}
.mcd-prod{border:1px solid var(--ring); border-radius:10px; padding:10px 12px;}
.mcd-prod .ph{display:flex; justify-content:space-between; align-items:baseline; gap:8px; margin-bottom:6px;}
.mcd-prod .pn{font-weight:600; font-size:13px; min-width:0; overflow-wrap:anywhere;}
.mcd-prod .pu{font-size:13px; font-weight:600; font-variant-numeric:tabular-nums; white-space:nowrap;}
.mcd-prod .pr{color:var(--muted); font-size:11px; font-weight:400;}
.mcd-prod dl{margin:0; display:grid; grid-template-columns:minmax(0,1fr) auto; gap:3px 10px; font-size:12px;}
.mcd-prod dt{color:var(--ink-2); min-width:0; overflow-wrap:anywhere;}
.mcd-prod dd{margin:0; text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap;}
.mcd-prod dt.short, .mcd-prod dd.short{color:var(--critical); font-weight:600;}
.mcd-table-wrap{overflow-x:auto; max-width:100%;}
.mcd table{border-collapse:collapse; width:100%; font-size:13px;}
.mcd th{text-align:left; font-weight:600; color:var(--ink-2); font-size:12px; padding:6px 8px; border-bottom:1px solid var(--axis); white-space:nowrap;}
.mcd td{padding:7px 8px; border-bottom:1px solid var(--grid); vertical-align:top;}
.mcd td.n, .mcd th.n{text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap;}
.mcd td.nw{white-space:nowrap;}
.mcd tbody tr:hover{background:var(--wash);}
.mcd .muted{color:var(--muted);}
.mcd .sub{color:var(--muted); font-size:11px;}
.mcd .cls{display:inline-block; min-width:20px; text-align:center; font-size:11px; font-weight:600; border:1px solid var(--axis);
  border-radius:5px; padding:0 5px; color:var(--ink-2);}
.mcd-status{display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:600; white-space:nowrap;}
.mcd-status i{width:16px; height:16px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
  font-style:normal; font-size:11px; font-weight:700; color:#ffffff; flex:none;}
.mcd-status.critical i{background:var(--critical);}
.mcd-status.warning i{background:var(--warning); color:var(--on-warning);}
.mcd-status.good i{background:var(--good);}
.mcd-empty{color:var(--muted); font-size:13px; padding:8px 0;}
.mcd-foot{color:var(--muted); font-size:12px; margin-top:8px;}
.mcd-note{border:1px solid var(--critical); border-radius:10px; padding:12px 14px; margin-bottom:16px; font-size:13px; color:var(--ink);}
.mcd-note b{display:block; margin-bottom:4px;}
.mcd-note code{font-family:ui-monospace, Menlo, Consolas, monospace; font-size:12px; overflow-wrap:anywhere;}
@media (max-width: 520px){
  .mcd{padding:16px 16px 32px;}
}
</style>
</head>
<body>
<div class="mcd" id="mcd">
 <div class="mcd-wrap">
  <div class="mcd-head">
   <h1>Inventory plan<span id="mcd-holiday"></span></h1>
   <div class="mcd-sub" id="mcd-sub"></div>
  </div>

  <div class="mcd-filters" role="group" aria-label="Forecast scenario">
   <span class="lab" id="mcd-scn-lab">Forecast scenario</span>
   <div class="mcd-seg" id="mcd-scn" aria-labelledby="mcd-scn-lab">
    <button type="button" data-s="low" aria-pressed="false">Low</button>
    <button type="button" data-s="prediction" aria-pressed="true">Base</button>
    <button type="button" data-s="high" aria-pressed="false">High</button>
   </div>
  </div>

  <section class="mcd-card" aria-labelledby="mcd-prod-h">
   <div class="mcd-card-h">
    <div><h2 id="mcd-prod-h">By product</h2><p id="mcd-prod-p"></p></div>
    <button type="button" class="mcd-btn" id="mcd-prod-toggle" aria-pressed="false">Table</button>
   </div>
   <div class="mcd-chips" id="mcd-weeks" role="group" aria-label="Forecast week"></div>
   <div id="mcd-prod-visual">
    <div class="mcd-key" id="mcd-fc-key"></div>
    <div class="mcd-chart" id="mcd-fc"></div>
    <div class="mcd-prods" id="mcd-prod"></div>
   </div>
   <div class="mcd-table-wrap" id="mcd-prod-table" hidden></div>
  </section>

  <section class="mcd-card" aria-labelledby="mcd-po-h">
   <div class="mcd-card-h">
    <div><h2 id="mcd-po-h">By ingredient</h2><p id="mcd-po-p"></p></div>
   </div>
   <div class="mcd-chips">
    <span class="lab" id="mcd-cls-lab">Class</span><span id="mcd-cls-wrap"></span>
    <span class="lab" style="margin-left:10px">Show</span><span id="mcd-view-wrap"></span>
   </div>
   <div class="mcd-table-wrap" id="mcd-po"></div>
  </section>

  <div class="mcd-note" id="mcd-note" hidden></div>
  <div class="mcd-foot" id="mcd-foot"></div>
 </div>
 <div class="mcd-tip" id="mcd-tip" role="status" aria-live="polite"></div>
</div>

<script>
(function(){
  function fail(what, detail){
    var n = document.getElementById("mcd-note");
    if (!n) return;
    n.hidden = false;
    n.textContent = "";
    var b = document.createElement("b"); b.textContent = "The dashboard could not finish drawing."; n.appendChild(b);
    var p = document.createElement("div"); p.textContent = what; n.appendChild(p);
    if (detail){ var c = document.createElement("code"); c.textContent = String(detail); n.appendChild(c); }
  }
  window.addEventListener("error", function(e){
    fail("A script error stopped it. If the message below is empty or says only \u0022Script error\u0022, the page is in a cross-origin frame that hides the detail - open dashboard.html directly to see it.",
      (e && e.message ? e.message : "") + (e && e.filename ? " (" + e.filename + ":" + e.lineno + ")" : ""));
  });
  try {
  var D = __DATA__;
  var state = {scn: "prediction", week: D.default_week, cls: "All", view: "buy", table: false};
  var SCN = {low: "Low", prediction: "Base", high: "High"};
  var root = document.getElementById("mcd");
  var tip = document.getElementById("mcd-tip");

  function el(tag, attrs, text){
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs){ if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]); }
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }
  function svg(tag, attrs){
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function fmt(v, d){
    if (v === null || v === undefined || isNaN(v)) return "-";
    var dd = (d === undefined) ? (Math.abs(v) < 10 && v % 1 !== 0 ? 1 : 0) : d;
    return Number(v).toLocaleString("en-US", {minimumFractionDigits: dd, maximumFractionDigits: dd});
  }
  function qty(v){
    if (v > 0 && v < 0.1) return "<0.1";
    if (v < 0 && v > -0.1) return ">-0.1";
    return fmt(v, (v % 1 === 0 || Math.abs(v) >= 100) ? 0 : 1);
  }
  function dlabel(s){
    if (!s) return "-";
    var p = String(s).split("-"); var dt = new Date(Date.UTC(+p[0], +p[1]-1, +p[2]));
    return dt.toLocaleDateString("en-US", {weekday: "short", month: "short", day: "numeric", timeZone: "UTC"});
  }
  function wlabel(s){
    var p = String(s).split("-"); var dt = new Date(Date.UTC(+p[0], +p[1]-1, +p[2]));
    return dt.toLocaleDateString("en-US", {month: "short", day: "numeric", timeZone: "UTC"});
  }
  var RATING = {green: "plan from the number", amber: "plan from the range", red: "use a par level"};

  document.getElementById("mcd-sub").textContent =
    "Orders for " + dlabel(D.plan_start) + " - " + dlabel(D.plan_end) + " · run " + dlabel(D.run_date) + " · forecast: " + D.forecast_source;
  if (D.holidays.length) document.getElementById("mcd-holiday").appendChild(el("span", {"class": "mcd-badge"}, D.holidays.join(", ") + " week"));
  document.getElementById("mcd-foot").textContent =
    "inventory_planner.py · class A = P-model (audits Mon & Thu), B = reorder point, C = two-bin / red line (visual check). " + (D.stock_note || "");

  var seg = document.getElementById("mcd-scn");
  seg.addEventListener("click", function(e){
    var b = e.target.closest("button"); if (!b) return;
    state.scn = b.getAttribute("data-s");
    Array.prototype.forEach.call(seg.querySelectorAll("button"), function(x){ x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
    render();
  });

  var wk = document.getElementById("mcd-weeks");
  D.weeks.forEach(function(w){
    wk.appendChild(el("button", {type: "button", "data-w": w.week, "aria-pressed": w.week === state.week ? "true" : "false"},
      "Week of " + wlabel(w.week) + (w.holiday ? " · " + w.holiday : "")));
  });
  wk.addEventListener("click", function(e){
    var b = e.target.closest("button"); if (!b) return;
    state.week = b.getAttribute("data-w");
    Array.prototype.forEach.call(wk.querySelectorAll("button"), function(x){ x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
    renderProducts();
  });

  document.getElementById("mcd-prod-toggle").addEventListener("click", function(){
    state.table = !state.table;
    this.setAttribute("aria-pressed", state.table ? "true" : "false");
    this.textContent = state.table ? "Chart" : "Table";
    document.getElementById("mcd-prod-visual").hidden = state.table;
    document.getElementById("mcd-prod-table").hidden = !state.table;
  });

  function chipGroup(hostId, values, current, onPick){
    var host = document.getElementById(hostId);
    values.forEach(function(v){
      host.appendChild(el("button", {type: "button", "data-v": v[0], "aria-pressed": v[0] === current ? "true" : "false"}, v[1]));
    });
    host.addEventListener("click", function(e){
      var b = e.target.closest("button"); if (!b) return;
      Array.prototype.forEach.call(host.querySelectorAll("button"), function(x){ x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
      onPick(b.getAttribute("data-v"));
      renderPlan();
    });
  }
  chipGroup("mcd-cls-wrap", [["All", "All"], ["A", "A"], ["B", "B"], ["C", "C"]], state.cls, function(v){ state.cls = v; });
  chipGroup("mcd-view-wrap", [["buy", "To buy"], ["all", "Everything used"]], state.view, function(v){ state.view = v; });

  /* ---- by product ---- */
  function showTip(evt, f, anchor){
    tip.textContent = "";
    tip.appendChild(el("div", {"class": "t"}, f.item + " · week of " + wlabel(f.week)));
    [["Low", f.low], ["Base", f.prediction], ["High", f.high]].forEach(function(r){
      var row = el("div", {"class": "row"});
      row.appendChild(el("b", null, fmt(r[1], 0)));
      row.appendChild(el("span", null, r[0] + (r[0] === SCN[state.scn] ? " (shown)" : "")));
      tip.appendChild(row);
    });
    tip.appendChild(el("div", {"class": "t", style: "margin:6px 0 0"},
      (RATING[f.rating] || f.rating) + (f.days < 5 ? " · " + f.days + " trading day" + (f.days === 1 ? "" : "s") : "")));
    tip.style.display = "block";
    var rr = root.getBoundingClientRect(), ar = anchor.getBoundingClientRect();
    var x = (evt && evt.clientX ? evt.clientX : ar.left + ar.width / 2) - rr.left + 14;
    var w = tip.offsetWidth;
    if (x + w > rr.width - 8) x = Math.max(8, x - w - 28);
    tip.style.left = x + "px"; tip.style.top = (ar.top - rr.top - 8) + "px";
  }
  function hideTip(){ tip.style.display = "none"; }
  function niceStep(raw){
    var p = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9)))), n = raw / p;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
  }

  function renderProducts(){
    var rows = D.forecast.filter(function(f){ return f.week === state.week; })
      .sort(function(a, b){ return b[state.scn] - a[state.scn] || b.prediction - a.prediction; });
    var w = D.weeks.filter(function(x){ return x.week === state.week; })[0] || {};
    var short = {}; D.need.forEach(function(n){ short[n.id] = (n.stock - n[state.scn]) < 0; });
    var need = {};
    D.product_need.forEach(function(r){ if (r.week === state.week && r[state.scn] > 0) (need[r.item] = need[r.item] || []).push(r); });
    var nShort = Object.keys(need).filter(function(it){
      return need[it].some(function(r){ return short[r.id]; }); }).length;
    document.getElementById("mcd-prod-p").textContent =
      "Units forecast for the week and the ingredients each product needs · " + SCN[state.scn].toLowerCase() + " scenario" +
      (w.days < 5 ? " · " + w.days + " trading day" + (w.days === 1 ? "" : "s") : "") +
      (nShort ? " · " + nShort + " product" + (nShort === 1 ? "" : "s") + " short on an ingredient this week" : "");

    /* chart */
    var host = document.getElementById("mcd-fc"); host.textContent = "";
    var key = document.getElementById("mcd-fc-key"); key.textContent = "";
    var k1 = el("span"); var s1 = svg("svg", {width: 14, height: 10});
    s1.appendChild(svg("rect", {x: 0, y: 1, width: 14, height: 8, rx: 2, fill: "var(--series)"}));
    k1.appendChild(s1); k1.appendChild(document.createTextNode(SCN[state.scn] + " forecast"));
    var k2 = el("span"); var s2 = svg("svg", {width: 16, height: 10});
    s2.appendChild(svg("line", {x1: 1, y1: 5, x2: 15, y2: 5, stroke: "var(--range)", "stroke-width": 1.5}));
    s2.appendChild(svg("line", {x1: 1, y1: 1, x2: 1, y2: 9, stroke: "var(--range)", "stroke-width": 1.5}));
    s2.appendChild(svg("line", {x1: 15, y1: 1, x2: 15, y2: 9, stroke: "var(--range)", "stroke-width": 1.5}));
    k2.appendChild(s2); k2.appendChild(document.createTextNode("Low-high range"));
    key.appendChild(k1); key.appendChild(k2);

    if (rows.length){
      var W = Math.max(300, host.clientWidth || 640), narrow = W < 520;
      var labW = narrow ? 118 : 210, valW = 44, rowH = 26, barH = 14, top = 6, axisH = 22;
      var maxV = Math.max.apply(null, rows.map(function(r){ return r.high; }).concat([1]));
      var step = niceStep(maxV / 4), xMax = Math.ceil(maxV / step) * step, plotW = W - labW - valW;
      var H = top + rows.length * rowH + axisH;
      var s = svg("svg", {viewBox: "0 0 " + W + " " + H, role: "list",
        "aria-label": "Forecast by product, week of " + wlabel(state.week)});
      var X = function(v){ return labW + (v / xMax) * plotW; };
      for (var t = 0; t <= xMax + 1e-9; t += step){
        s.appendChild(svg("line", {x1: X(t), x2: X(t), y1: top - 2, y2: top + rows.length * rowH,
          stroke: t === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1}));
        var tl = svg("text", {x: X(t), y: H - 6, "text-anchor": "middle", "font-size": 11, fill: "var(--muted)"});
        tl.textContent = fmt(t, step < 1 ? 1 : 0); s.appendChild(tl);
      }
      rows.forEach(function(f, i){
        var y = top + i * rowH, cy = y + rowH / 2, v = f[state.scn];
        var g = svg("g", {"class": "row", tabindex: 0, role: "listitem",
          "aria-label": f.item + ": " + SCN[state.scn] + " " + fmt(v, 0) + ", range " + fmt(f.low, 0) + " to " + fmt(f.high, 0)});
        var hit = svg("rect", {x: 0, y: y, width: W, height: rowH, "class": "hit"});
        g.appendChild(hit);
        var name = f.item, maxChars = narrow ? 16 : 30;
        if (name.length > maxChars) name = name.slice(0, maxChars - 1) + "...";
        var lt = svg("text", {x: labW - 8, y: cy + 4, "text-anchor": "end", "font-size": 12, fill: "var(--ink)"});
        lt.textContent = name; g.appendChild(lt);
        var bw = Math.max(0, X(v) - X(0));
        if (bw > 0){
          var r = Math.min(4, bw), x0 = X(0), y0 = cy - barH / 2, x1 = X(v);
          g.appendChild(svg("path", {"class": "bar", d: "M" + x0 + "," + y0 + " H" + (x1 - r) + " Q" + x1 + "," + y0 + " " + x1 + "," + (y0 + r) +
            " V" + (y0 + barH - r) + " Q" + x1 + "," + (y0 + barH) + " " + (x1 - r) + "," + (y0 + barH) + " H" + x0 + " Z"}));
        }
        g.appendChild(svg("line", {x1: X(f.low), x2: X(f.high), y1: cy, y2: cy, stroke: "var(--range)", "stroke-width": 1.5}));
        g.appendChild(svg("line", {x1: X(f.low), x2: X(f.low), y1: cy - 5, y2: cy + 5, stroke: "var(--range)", "stroke-width": 1.5}));
        g.appendChild(svg("line", {x1: X(f.high), x2: X(f.high), y1: cy - 5, y2: cy + 5, stroke: "var(--range)", "stroke-width": 1.5}));
        var vt = svg("text", {x: Math.max(X(v), X(f.high)) + 6, y: cy + 4, "font-size": 11, fill: "var(--ink-2)"});
        vt.textContent = fmt(v, 0); g.appendChild(vt);
        g.addEventListener("pointermove", function(e){ showTip(e, f, hit); });
        g.addEventListener("pointerleave", hideTip);
        g.addEventListener("focus", function(){ showTip(null, f, hit); });
        g.addEventListener("blur", hideTip);
        s.appendChild(g);
      });
      host.appendChild(s);
    } else {
      host.appendChild(el("div", {"class": "mcd-empty"}, "No forecast rows for this week."));
    }

    /* product cards */
    var cards = document.getElementById("mcd-prod"); cards.textContent = "";
    rows.forEach(function(f){
      var lines = (need[f.item] || []).slice().sort(function(a, b){ return b[state.scn] - a[state.scn]; });
      if (!lines.length) return;
      var d = el("div", {"class": "mcd-prod"});
      var h = el("div", {"class": "ph"});
      h.appendChild(el("div", {"class": "pn"}, f.item));
      var pu = el("div", {"class": "pu"}, fmt(f[state.scn], 0));
      pu.appendChild(el("span", {"class": "pr"}, " units"));
      h.appendChild(pu); d.appendChild(h);
      var dl = el("dl");
      lines.forEach(function(r){
        var sh = short[r.id];
        dl.appendChild(el("dt", {"class": sh ? "short" : null}, r.name));
        dl.appendChild(el("dd", {"class": sh ? "short" : null}, qty(r[state.scn]) + " " + r.unit));
      });
      d.appendChild(dl);
      cards.appendChild(d);
    });

    /* product table */
    var tw = document.getElementById("mcd-prod-table"); tw.textContent = "";
    var tb = el("table"), th = el("thead"), tr = el("tr");
    [["Product", ""], ["Units", "n"], ["Ingredient", ""], ["Need", "n"], ["Unit", ""]].forEach(function(x){ tr.appendChild(el("th", {"class": x[1]}, x[0])); });
    th.appendChild(tr); tb.appendChild(th);
    var bd = el("tbody");
    rows.forEach(function(f){
      var lines = (need[f.item] || []).slice().sort(function(a, b){ return b[state.scn] - a[state.scn]; });
      if (!lines.length){
        var row = el("tr");
        row.appendChild(el("td", null, f.item));
        row.appendChild(el("td", {"class": "n"}, fmt(f[state.scn], 0)));
        row.appendChild(el("td", {"class": "muted"}, "no recipe"));
        row.appendChild(el("td", {"class": "n"}, "-")); row.appendChild(el("td", null, ""));
        bd.appendChild(row); return;
      }
      lines.forEach(function(r, i){
        var row = el("tr");
        row.appendChild(el("td", null, i === 0 ? f.item : ""));
        row.appendChild(el("td", {"class": "n"}, i === 0 ? fmt(f[state.scn], 0) : ""));
        row.appendChild(el("td", null, r.name));
        row.appendChild(el("td", {"class": "n"}, qty(r[state.scn])));
        row.appendChild(el("td", {"class": "muted"}, r.unit));
        bd.appendChild(row);
      });
    });
    tb.appendChild(bd); tw.appendChild(tb);
  }

  /* ---- by ingredient ---- */
  function renderPlan(){
    var host = document.getElementById("mcd-po"); host.textContent = "";
    var byId = {};
    D.need.forEach(function(n){ byId[n.id] = {id: n.id, name: n.name, unit: n.unit, cls: n.cls, need: n[state.scn], stock: n.stock, packs: 0, dates: [], arrive: null, pack_label: "", supplier: "", rule: ""}; });
    D.orders.filter(function(o){ return o.scenario === state.scn; }).forEach(function(o){
      var r = byId[o.id] = byId[o.id] || {id: o.id, name: o.name, unit: o.unit, cls: o.cls, need: 0, stock: 0, packs: 0, dates: [], arrive: null, pack_label: "", supplier: "", rule: ""};
      r.packs += o.packs; r.pack_label = o.pack_label; r.supplier = o.supplier; r.rule = o.rule;
      if (r.dates.indexOf(o.order_date) < 0) r.dates.push(o.order_date);
      if (!r.arrive || o.arrival_date > r.arrive) r.arrive = o.arrival_date;
    });
    var rows = Object.keys(byId).map(function(k){ return byId[k]; })
      .filter(function(r){ return (r.need > 0 || r.packs > 0) && (state.cls === "All" || r.cls === state.cls)
        && (state.view === "all" || r.packs > 0 || (r.stock - r.need) < 0); })
      .sort(function(a, b){ return (b.packs > 0) - (a.packs > 0) || (a.stock - a.need) - (b.stock - b.need); });
    var ordering = rows.filter(function(r){ return r.packs > 0; });
    var packs = ordering.reduce(function(s, r){ return s + r.packs; }, 0);
    document.getElementById("mcd-po-p").textContent =
      "Next " + D.review_days + " days · " + SCN[state.scn].toLowerCase() + " scenario · " + ordering.length +
      " material" + (ordering.length === 1 ? "" : "s") + " to buy, " + packs + " pack" + (packs === 1 ? "" : "s") +
      " · quantities rounded up to whole packs" + (state.view === "buy" ? " · showing only what needs buying" : "");
    if (!rows.length){ host.appendChild(el("div", {"class": "mcd-empty"}, "Nothing to buy for this selection - stock covers the week.")); return; }
    var tb = el("table"), th = el("thead"), tr = el("tr");
    [["Material", ""], ["Class", ""], ["Need", "n"], ["On hand", "n"], ["Buy", "n"], ["Pack", ""], ["Order on", ""], ["Arrives", ""], ["Supplier", ""], ["Status", ""]]
      .forEach(function(x){ tr.appendChild(el("th", {"class": x[1]}, x[0])); });
    th.appendChild(tr); tb.appendChild(th);
    var bd = el("tbody");
    rows.forEach(function(r){
      var row = el("tr");
      var m = el("td"); m.appendChild(el("div", null, r.name)); m.appendChild(el("div", {"class": "sub"}, r.id)); row.appendChild(m);
      var c = el("td"); c.appendChild(el("span", {"class": "cls"}, r.cls || "-")); row.appendChild(c);
      row.appendChild(el("td", {"class": "n"}, qty(r.need) + " " + r.unit));
      row.appendChild(el("td", {"class": "n"}, qty(r.stock) + " " + r.unit));
      row.appendChild(el("td", {"class": "n"}, r.packs ? fmt(r.packs, 0) : "-"));
      row.appendChild(el("td", {"class": "muted"}, r.packs ? r.pack_label : ""));
      row.appendChild(el("td", {"class": "nw"}, r.dates.length ? r.dates.sort().map(dlabel).join(", ") : ""));
      row.appendChild(el("td", {"class": "nw"}, r.arrive ? dlabel(r.arrive) : ""));
      row.appendChild(el("td", {"class": "muted"}, r.supplier));
      var left = r.stock - r.need;
      var s = r.packs > 0 ? ["warning", "~", "Ordering"] : (left < 0 ? ["critical", "!", "Short"] : ["good", "check", "Covered"]);
      var st = el("span", {"class": "mcd-status " + s[0]});
      st.appendChild(el("i", {"aria-hidden": "true"}, s[1] === "check" ? "✓" : s[1]));
      st.appendChild(document.createTextNode(s[2]));
      var sc = el("td"); sc.appendChild(st); row.appendChild(sc);
      bd.appendChild(row);
    });
    tb.appendChild(bd); host.appendChild(tb);
  }

  function render(){ renderProducts(); renderPlan(); }
  render();
  var lastW = 0;
  if (window.ResizeObserver){
    new ResizeObserver(function(){
      var w = document.getElementById("mcd-fc").clientWidth;
      if (Math.abs(w - lastW) > 4){ lastW = w; renderProducts(); }
    }).observe(document.getElementById("mcd-fc"));
  }
  } catch (err) {
    fail("Error while building the page.", (err && err.message ? err.message : err) + (err && err.stack ? " | " + String(err.stack).split("\u000a")[1] : ""));
  }
})();
</script>
</body>
</html>
'''


def records(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))


week_meta = (forecast.groupby("week_start", as_index=False)
             .agg(holiday=("holiday", "first"), days=("expected_trading_days", "first")).sort_values("week_start"))
plan_week_strs = [w.strftime("%Y-%m-%d") for w in plan_weeks if w in set(forecast["week_start"])]
dash_fc = forecast.assign(week=forecast["week_start"].dt.strftime("%Y-%m-%d"), item=forecast["menu_item"],
                          days=forecast["expected_trading_days"])[["week", "item", "low", "prediction", "high", "rating", "days"]]
dash_orders = plans[plans["packs"] > 0].assign(
    id=lambda x: x["ingredient"], cls=lambda x: x["class"], qty=lambda x: x["order_qty"],
    order_date=lambda x: x["order_date"].astype(str), arrival_date=lambda x: x["arrival_date"].astype(str))[
    ["scenario", "id", "name", "cls", "supplier", "order_date", "arrival_date", "packs", "pack_label", "qty", "unit", "rule", "flags"]]
on_hand = stock_lots.groupby("ingredient")["qty"].sum()
need_all = pd.DataFrame({s: requirements[s][requirements[s]["date"] < PLAN_DATE + pd.Timedelta(days=REVIEW_DAYS)]
                        .groupby("ingredient")["total_qty"].sum() for s in SCENARIOS})
need_all = need_all[need_all["high"] > 0].join(ing_info[["name", "unit"]])
need_all["stock"] = on_hand.reindex(need_all.index).fillna(0.0)
need_all["cls"] = ing_info["inventory_class"].reindex(need_all.index)
dash_need = need_all.round(2).reset_index().rename(columns={"ingredient": "id"})
dash = {
    "run_date": RUN_DATE.strftime("%Y-%m-%d"),
    "plan_start": PLAN_DATE.strftime("%Y-%m-%d"),
    "plan_end": (PLAN_DATE + pd.Timedelta(days=REVIEW_DAYS - 1)).strftime("%Y-%m-%d"),
    "review_days": REVIEW_DAYS,
    "forecast_source": "forecast_weekly (latest run)" if DATABASE_URL else os.path.basename(FORECAST_FILE),
    "weeks": [{"week": r.week_start.strftime("%Y-%m-%d"), "holiday": r.holiday, "days": int(r.days)} for r in week_meta.itertuples()],
    "plan_weeks": plan_week_strs,
    "default_week": plan_week_strs[0] if plan_week_strs else week_meta["week_start"].iloc[0].strftime("%Y-%m-%d"),
    "holidays": sorted({h for h in forecast[forecast["week_start"].isin(plan_weeks)]["holiday"] if h}),
    "forecast": records(dash_fc),
    "orders": records(dash_orders),
    "need": records(dash_need),
    "product_need": records(product_need.rename(columns={"week_start": "week", "product": "item", "ingredient_id": "id"})
                            .assign(week=lambda x: x["week"].astype(str))),
    "stock_note": STOCK_NOTE,
}
next_week = dash["default_week"]
product_csv = (product_need[product_need["week_start"].astype(str) == next_week]
               .rename(columns={"week_start": "week", "product": "item", "ingredient_id": "ingredient"})
               .sort_values(["item", "name"]))
ingredient_csv = (dash_need.rename(columns={"id": "ingredient"})
                  .merge(plans[plans["scenario"] == "prediction"].groupby("ingredient", as_index=False)
                         .agg(packs_to_buy=("packs", "sum")), on="ingredient", how="left")
                  .assign(week=next_week, packs_to_buy=lambda x: x["packs_to_buy"].fillna(0).astype(int))
                  .sort_values("prediction", ascending=False))
ingredient_csv = ingredient_csv[["week", "ingredient", "name", "unit", "cls", "low", "prediction", "high",
                                 "stock", "packs_to_buy"]]
save(product_csv, "product_need_next_week")
save(ingredient_csv, "ingredient_need_next_week")

html = DASHBOARD_TEMPLATE.replace("__DATA__", json.dumps(dash).replace("</", "<\\/"))
with open(os.path.join(OUT_DIR, "dashboard.html"), "w", encoding="utf-8") as fh:
    fh.write(html)
if DATABASE_URL:
    save(pd.DataFrame({"html": [html]}), "dashboard")
print("\nOutputs for", RUN_DATE.date(), "- week of", next_week)
print("  dashboard.html                 the page for the admin site")
print("  product_need_next_week.csv    ", len(product_csv), "rows:", product_csv["item"].nunique(), "products")
print("  ingredient_need_next_week.csv ", len(ingredient_csv), "rows:",
      int((ingredient_csv["packs_to_buy"] > 0).sum()), "to buy")

print("\nRun finished for", RUN_DATE.date())

# %% [markdown]
# Scheduling (run once a day, early morning, after the forecast model and the latest stock counts are in the database)
#   Linux server or website host (cron):
#     0 6 * * *  cd /path/to/planner && DATABASE_URL="postgresql://user:password@host/dbname" python3 inventory_planner.py >> planner.log 2>&1
#   GitHub Actions (file .github/workflows/planner.yml):
#     on: {schedule: [{cron: "0 11 * * *"}], workflow_dispatch: {}}
#     jobs: {run: {runs-on: ubuntu-latest, steps: [{uses: actions/checkout@v4},
#       {run: "pip install pandas numpy scipy sqlalchemy psycopg2-binary"},
#       {run: "python inventory_planner.py", env: {DATABASE_URL: "${{ secrets.DATABASE_URL }}"}}]}}
#   The GitHub cron time is in UTC (11:00 UTC is 7:00 in Indianapolis during standard time).
#   Supabase connection string: Project Settings > Database > Connection string (use the pooler URI).
#   CSV run: put live_forecast.csv, Inventory.csv and recipes.csv in planner_data, then run the script.
