from dataclasses import dataclass
from typing import Optional

@dataclass
class Feature:
    name: str
    dtype: str                        # 'float', 'int', 'categorical', 'binary', 'datetime', 'id'
    description: str
    possible_values: Optional[list]   = None   # for categoricals / binaries
    value_range: Optional[tuple]      = None   # (min, max) for numerics
    notes: str                        = ""
    used_in_model: bool               = True   # False = identifier / raw temporal / target
 
 
FEATURES: list[Feature] = [
 
    # ── Identifiers ───────────────────────────────────────────────────────────
    Feature("Transaction_ID",  "id", "Unique transaction identifier",       used_in_model=False),
    Feature("Customer_ID",     "id", "Unique customer identifier",          used_in_model=False),
    Feature("Merchant_ID",     "id", "Unique merchant identifier",          used_in_model=False),
    Feature("Device_ID",       "id", "Device used for transaction",         used_in_model=False),
    Feature("IP_Address",      "id", "IP address of transaction origin",    used_in_model=False),
 
    # ── Raw temporal (parsed into derived columns in preprocess.py) ───────────
    Feature("Transaction_Time", "datetime", "Time of transaction (HH:MM)",
            notes="Parsed into Transaction_Hour", used_in_model=False),
    Feature("Transaction_Date", "datetime", "Date of transaction (YYYY-MM-DD)",
            notes="Parsed into Transaction_DayOfWeek, Transaction_Month",
            used_in_model=False),
 
    # ── Numeric ───────────────────────────────────────────────────────────────
    Feature("Transaction_Amount (in Million)", "float",
            "Value of the transaction in millions",
            value_range=(1, 9)),
 
    Feature("Distance_From_Home", "float",
            "Distance (km) between transaction location and customer home",
            value_range=(1, 599),
            notes="Higher distance is a known fraud signal"),
 
    Feature("Account_Balance (in Million)", "float",
            "Customer account balance at time of transaction",
            value_range=(3, 39)),
 
    Feature("Daily_Transaction_Count", "int",
            "Number of transactions by this customer today",
            value_range=(1, 7)),
 
    Feature("Weekly_Transaction_Count", "int",
            "Number of transactions by this customer this week",
            value_range=(1, 24)),
 
    Feature("Avg_Transaction_Amount (in Million)", "float",
            "Customer historical average transaction amount",
            value_range=(1, 5)),
 
    Feature("Max_Transaction_Last_24h (in Million)", "float",
            "Largest transaction by this customer in the last 24 hours",
            value_range=(1, 9)),
 
    Feature("Failed_Transaction_Count", "int",
            "Number of failed transactions by this customer recently",
            value_range=(0, 2),
            possible_values=[0, 1, 2]),
 
    Feature("Previous_Fraud_Count", "int",
            "Number of prior fraudulent transactions on this account",
            value_range=(0, 1),
            possible_values=[0, 1],
            notes="Binary in practice; strong prior indicator"),
 
    # ── Derived temporal (created by preprocess.py, not in raw CSV) ───────────
    Feature("Transaction_Hour", "int",
            "Hour of day (0-23) extracted from Transaction_Time",
            value_range=(0, 23),
            notes="Derived — not present in raw CSV"),
 
    Feature("Transaction_DayOfWeek", "int",
            "Day of week (0=Mon, 6=Sun) extracted from Transaction_Date",
            value_range=(0, 6),
            notes="Derived — not present in raw CSV"),
 
    Feature("Transaction_Month", "int",
            "Month (1-12) extracted from Transaction_Date",
            value_range=(1, 12),
            notes="Derived — not present in raw CSV"),
 
    # ── Categorical ───────────────────────────────────────────────────────────
    Feature("Transaction_Type", "categorical",
            "Channel through which the transaction was made",
            possible_values=["Online", "ATM", "POS"]),
 
    Feature("Merchant_Category", "categorical",
            "Category of the merchant",
            possible_values=["ATM", "Electronics", "Grocery",
                             "Fuel", "Restaurant", "Clothing"]),
 
    Feature("Transaction_Location", "categorical",
            "City where the transaction occurred",
            possible_values=["Singapore", "Faisalabad", "London", "Karachi",
                             "Lahore", "Bangkok", "Multan", "Islamabad",
                             "Kuala Lumpur", "Dubai"]),
 
    Feature("Customer_Home_Location", "categorical",
            "Customer registered home city",
            possible_values=["Lahore", "Faisalabad", "Karachi", "Islamabad", "Multan"]),
 
    Feature("Card_Type", "categorical",
            "Payment card type",
            possible_values=["Debit", "Credit"]),
 
    # ── Binary Yes/No (encoded as 0/1 in preprocess.py) ──────────────────────
    Feature("Is_International_Transaction", "binary",
            "Whether the transaction crossed a national border",
            possible_values=["Yes", "No"],
            notes="Encoded Yes=1, No=0"),
 
    Feature("Is_New_Merchant", "binary",
            "Whether this is the customer's first transaction with this merchant",
            possible_values=["Yes", "No"],
            notes="Encoded Yes=1, No=0"),
 
    Feature("Unusual_Time_Transaction", "binary",
            "Flagged by the bank as occurring at an unusual hour",
            possible_values=["Yes", "No"],
            notes="Encoded Yes=1, No=0"),
 
    # ── Target label (y, not part of X) ──────────────────────────────────────
    Feature("Fraud_Label", "categorical",
            "Ground truth fraud classification",
            possible_values=["Normal", "Fraud"],
            notes="Encoded Normal=0, Fraud=1. ~4.8% fraud rate — imbalanced dataset.",
            used_in_model=False),
]
 
 
# ── Derived lookups (everything else imports these) ───────────────────────────
 
# Fast name → Feature lookup
FEATURE_MAP: dict[str, Feature] = {f.name: f for f in FEATURES}
 
# All columns that form X
MODEL_FEATURES   = [f for f in FEATURES if f.used_in_model]
 
# Grouped by dtype — consumed by preprocess.py
NUMERIC_FEATURES = [f for f in MODEL_FEATURES if f.dtype in ("float", "int")]
CAT_FEATURES     = [f for f in MODEL_FEATURES if f.dtype == "categorical"]
BINARY_FEATURES  = [f for f in MODEL_FEATURES if f.dtype == "binary"]
 
# Columns to strip before sklearn sees the data
DROP_COLS        = [f.name for f in FEATURES if f.dtype in ("id", "datetime")]
 
# The target column — looked up by name so it never needs to be typed twice
TARGET_COL       = "Fraud_Label"
 
 
def validate_dataframe(df) -> None:
    """
    Raise a descriptive ValueError if any expected raw column is absent.
    Derived features are excluded — they don't exist in the raw CSV yet.
    """
    derived  = {f.name for f in FEATURES if "Derived" in f.notes}
    expected = {f.name for f in FEATURES if f.name not in derived}
    missing  = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing {len(missing)} expected column(s):\n"
            + "\n".join(f"  - {c}" for c in sorted(missing))
        )
 
 
def print_summary() -> None:
    """Print a human-readable feature summary table."""
    print(f"{'Feature':<45} {'Type':<12} {'In Model':<10} {'Range / Values'}")
    print("-" * 100)
    for f in FEATURES:
        if f.value_range:
            val_info = f"{f.value_range[0]} – {f.value_range[1]}"
        elif f.possible_values:
            val_info = ", ".join(str(v) for v in f.possible_values)
        else:
            val_info = "—"
        flag = "✓" if f.used_in_model else "—"
        print(f"{f.name:<45} {f.dtype:<12} {flag:<10} {val_info}")
    print(
        f"\nTotal: {len(FEATURES)} | In model: {len(MODEL_FEATURES)} | "
        f"Numeric: {len(NUMERIC_FEATURES)} | Categorical: {len(CAT_FEATURES)} | "
        f"Binary: {len(BINARY_FEATURES)}"
    )
 
 
if __name__ == "__main__":
    print_summary()
 