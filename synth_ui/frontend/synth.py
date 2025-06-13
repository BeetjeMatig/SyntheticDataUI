"""Synthetic data generation utilities using KDEs, probabilistic sampling, and Mesa agents.

This module provides functions and classes to generate synthetic household data
based on real estate attributes, including address mappings, KDE sampling,
categorical distributions, hashing, and agent-based modeling.

Classes:
    Agent: Represents an individual agent in the simulation.
    Model: Represents the simulation model.

Functions:
    make_synthetic_housing: Generate synthetic housing data.
"""

# Import necessary libraries
import base64
import hashlib
import random
import re
from typing import Any

import pandas as pd
from faker import Faker
from mesa import Agent, Model
from mesa.agent import AgentSet
from scipy.stats import gaussian_kde

# Initialize Faker instance
fake = Faker("nl_NL")
fake.seed_instance(42)  # Set seed for reproducibility


CONFIG = {
    "kde_columns": ["OPPERVLAKTE", "INHOUD", "KAVEL", "WOZWAARDE"],
    "kde_sampling_rules": {
        "OPPERVLAKTE": {"min": 10.0},
        "INHOUD": {"min": 20.0},
        "KAVEL": {"min_field": "oppervlakte"},
        "WOZWAARDE": {"min": 0, "round_int": True},
        "TRANSACTIEPRIJS": {"min": 0, "round_int": True},
        "TRANSACTIECORRECTIE": {"min": 0, "round_int": True},
    },
    "categorical_columns": [
        "GEBRUIKSDOEL",
        "STATUS_VBO",
        "WONINGTYPECODE",
        "BOUWJAAR",
        "WOZPEIL",
        "WOZDATUM",
        "HUISNUMMER",
        "HUISLETTER",
        "TOEVOEGING",
    ],
    "synthetic_ids": {"BAG_NUMMER": "bag_start", "WOZ_NUMMER": "woz_start"},
    "hashed_columns": ["GEBRUIKERSCODE", "EIGENAARSCODE"],
    "one_to_one_mappings": {
        "WONINGTYPECODE": "WONINGTYPE",
        "BUURTCODE": "BUURTNAAM",
        "WIJKCODE": "WIJKNAAM",
    },
    "one_to_many_mappings": {"WONINGTYPE": "WONINGTYPE_", "BOUWJAAR": "BOUWJAAR_"},
    "threshold_binning": {
        "WOZWAARDE": "bin_wozwaarde",
        "OPPERVLAKTE": "bin_oppervlakte",
    },
    "output_columns": [
        "ADRES",
        "STRAATNAAM",
        "HUISNUMMER",
        "HUISLETTER",
        "TOEVOEGING",
        "POSTCODE",
        "WOONPLAATS",
        "BUURTCODE",
        "BUURTNAAM",
        "WIJKCODE",
        "WIJKNAAM",
        "BOUWJAAR_",
        "BOUWJAAR",
        "OPPERVLAKTE_",
        "OPPERVLAKTE",
        "INHOUD",
        "KAVEL",
        "WOZWAARDE_",
        "WOZWAARDE",
        "WOZDATUM",
        "GEBRUIKSDOEL",
        "STATUS_VBO",
        "WONINGTYPE_",
        "WONINGTYPE",
        "WONINGTYPECODE",
        "EIGENDOM",
        "GEBRUIKERSCODE",
        "GEBRUIKERSNAAM",
        "EIGENAARSCODE",
        "EIGENAARSNAAM",
        "BAG_NUMMER",
        "WOZ_NUMMER",
        "WOZPEIL",
    ],
}


def build_address_data(
    df: pd.DataFrame, city_col="WOONPLAATS"
) -> dict[str, list[dict[str, Any]]]:
    """Build a mapping from city to address list.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing address fields, e.g., 'POSTCODE', 'STRAATNAAM'.
    city_col : str, optional
        Column name for city grouping (default 'WOONPLAATS').

    Returns
    -------
    dict of str to list of dict[str, Any]
        Mapping from city to a list of address dictionaries with keys
        'postcode', 'street', 'buurtcode', 'wijkcode'.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for city, grp in df.groupby(city_col):
        # base subset: non-null key fields
        base = grp.dropna(subset=["POSTCODE", "STRAATNAAM", "BUURTCODE", "WIJKCODE"])
        # remove completely blank street names
        nonblank = base[base["STRAATNAAM"].astype(str).str.strip().ne("")]
        # first try: alpha-containing streets
        addrs: list[dict[str, Any]] = []
        for _, row in nonblank.iterrows():
            street = str(row["STRAATNAAM"])
            if re.search(r"[A-Za-z]", street):
                addrs.append(
                    {
                        "postcode": str(row["POSTCODE"]),
                        "street": street,
                        "buurtcode": row["BUURTCODE"],
                        "wijkcode": row["WIJKCODE"],
                    }
                )
        # fallback 1: include numeric-only but nonblank
        if not addrs and not nonblank.empty:
            for _, row in nonblank.iterrows():
                addrs.append(
                    {
                        "postcode": str(row["POSTCODE"]),
                        "street": str(row["STRAATNAAM"]),
                        "buurtcode": row["BUURTCODE"],
                        "wijkcode": row["WIJKCODE"],
                    }
                )
        # fallback 2: if still empty, skip this city
        if addrs:
            out[str(city)] = addrs
    return out


def build_kdes(df: pd.DataFrame, columns: list[str]) -> dict[str, gaussian_kde]:
    """Build Gaussian KDE estimators for specified numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame containing numeric data.
    columns : list of str
        List of column names to build KDEs for.

    Returns
    -------
    dict of str to gaussian_kde
        A KDE object for each specified column.
    """
    # Drop rows missing any KDE columns and convert them to numeric floats
    clean = df.dropna(subset=columns).copy()
    for col in columns:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    # Remove any rows where conversion failed
    clean = clean.dropna(subset=columns)
    # Build KDEs on numeric arrays
    return {col: gaussian_kde(clean[col].astype(float).values) for col in columns}


def build_weighted_categories(df: pd.DataFrame, columns: list[str]) -> dict[str, tuple]:
    """Compute weighted category distributions for given columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame with categorical columns.
    columns : list of str
        List of column names to compute distributions for.

    Returns
    -------
    dict of str to tuple
        Mapping from column name to a tuple of (values, probabilities).
    """
    df2 = df.copy()
    df2["HUISLETTER"] = df2["HUISLETTER"].fillna("__MISSING__").astype(str)
    df2["TOEVOEGING"] = df2["TOEVOEGING"].fillna("__MISSING__").astype(str)
    out: dict[str, tuple] = {}
    for col in columns:
        vals, probs = zip(*df2[col].dropna().value_counts(normalize=True).items())
        out[col] = (vals, probs)
    return out


def hash_id(val: Any, salt="leeuwarden", length=12) -> Any:
    """Hash an identifier using SHA-256 and return a URL-safe truncated string.

    Parameters
    ----------
    val : Any
        The value to hash (skips if null or placeholder '9999999999').
    salt : str, optional
        Salt string for hashing (default 'leeuwarden').
    length : int, optional
        Number of characters in the output hash (default 12).

    Returns
    -------
    Any
        Original value if null or placeholder; otherwise a hashed string.
    """
    if pd.isnull(val) or str(val) == "9999999999":
        return val
    d = hashlib.sha256((salt + str(val)).encode()).digest()
    return base64.urlsafe_b64encode(d).decode()[:length]


def build_fake_name_map(
    df: pd.DataFrame, koopwoning_logic: bool = True
) -> tuple[dict, dict]:
    """Generate fake name mappings for users and owners.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'GEBRUIKERSCODE', 'EIGENAARSCODE', and 'EIGENDOM_'.
    koopwoning_logic : bool, optional
        If True, owners of 'koopwoning' get user names; otherwise random.

    Returns
    -------
    tuple of dicts (owner_map, user_map)
        owner_map: mapping of owner code to name;
        user_map: mapping of user code to fake user name.
    """
    user_map: dict = {}
    owner_map: dict = {}
    for code in df["GEBRUIKERSCODE"].dropna().unique():
        user_map[code] = f"{fake.first_name()} {fake.last_name()}"
    for _, row in df.iterrows():
        eigendom = str(row["EIGENDOM_"]).strip().lower()
        ec = row["EIGENAARSCODE"]
        uc = row["GEBRUIKERSCODE"]
        if pd.isna(ec):
            continue
        if koopwoning_logic and "koopwoning" in eigendom:
            owner_map[ec] = user_map.get(uc)
        else:
            owner_map.setdefault(ec, fake.company())
    return owner_map, user_map


def build_mapping_dicts(df: pd.DataFrame, mappings: dict) -> dict:
    """Create mapping dictionaries from code to name columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame containing mapping columns.
    mappings : dict
        Dict of code-column to name-column mappings.

    Returns
    -------
    dict
        Mapping from each code-column to its name lookup dict.
    """
    out: dict = {}
    for code_col, name_col in mappings.items():
        mdf = df.dropna(subset=[code_col, name_col]).drop_duplicates(
            [code_col, name_col]
        )
        out[code_col] = mdf.set_index(code_col)[name_col].to_dict()
    return out


def build_class_mappings(df: pd.DataFrame, mapping: dict) -> dict:
    """Create class mappings for one-to-many relationships.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame containing source and target columns.
    mapping : dict
        Dict of name-column to class-column mappings.

    Returns
    -------
    dict
        Mapping from each class-column to a lookup dict by name.
    """
    out: dict = {}
    for name_col, class_col in mapping.items():
        mdf = (
            df.dropna(subset=[name_col, class_col])
            .drop_duplicates(name_col)
            .set_index(name_col)[class_col]
        )
        out[class_col] = mdf.to_dict()
    return out


def bin_wozwaarde(v: float) -> str:
    """Categorize 'wozwaarde' values into human-readable bins.

    Parameters
    ----------
    v : float
        The property value to categorize.

    Returns
    -------
    str
        Bin label.
    """
    if pd.isnull(v):
        return "Onbekend"
    if v < 50000:
        return "Tot € 50.000"
    if v < 80000:
        return "€ 50.000 tot € 80.000"
    if v < 110000:
        return "€ 80.000 tot € 110.000"
    if v < 170000:
        return "€ 110.000 tot € 170.000"
    if v < 230000:
        return "€ 170.000 tot € 230.000"
    return "€ 230.000 of meer"


def bin_oppervlakte(v: float) -> str:
    """Categorize 'oppervlakte' values into size bins in m².

    Parameters
    ----------
    v : float
        Surface area value.

    Returns
    -------
    str
        Size bin label.
    """
    if pd.isnull(v):
        return "Onbekend"
    if v < 50:
        return "Tot 50 m²"
    if v < 75:
        return "50 tot 75 m²"
    if v < 100:
        return "75 tot 100 m²"
    if v < 125:
        return "100 tot 125 m²"
    if v < 150:
        return "125 tot 150 m²"
    if v < 200:
        return "150 tot 200 m²"
    return "200 m² of meer"


class HouseholdAgent(Agent):
    """An agent representing a household with synthetic real estate attributes.

    Parameters
    ----------
    model : AddressModel
        The Mesa model instance providing data and configurations.
    """

    # tell Pylance these will exist
    huisnummer: int
    huisletter: str
    toevoeging: str
    bag_nummer: float
    woz_nummer: float

    def __init__(self, model: "AddressModel"):
        """Initialize the household agent with sampled attributes.

        Parameters
        ----------
        model : AddressModel
            The model to sample data from.
        """
        super().__init__(model)
        self.city = self.random.choice(list(model.address_data))
        # sample a coherent address tuple
        addr_list = model.address_data[self.city]
        addr = self.random.choice(addr_list)
        self.postcode = addr["postcode"]
        self.street = addr["street"]
        self.buurtcode = addr["buurtcode"]
        self.wijkcode = addr["wijkcode"]

        # --- KDE fields ---
        for f in CONFIG["kde_columns"]:
            val = model.kdes[f].resample(1).item()
            rules = CONFIG["kde_sampling_rules"].get(f, {})
            if rules.get("round_int"):
                val = round(val)
            if "min_field" in rules:
                val = max(getattr(self, rules["min_field"]), val)
            elif "min" in rules:
                val = max(rules["min"], val)
            setattr(self, f.lower(), int(val) if rules.get("round_int") else float(val))

        # --- categoricals ---
        for f in CONFIG["categorical_columns"]:
            vals, probs = model.categoricals[f]
            setattr(self, f.lower(), self.random.choices(vals, weights=probs)[0])

        # --- hashed ---
        for f in CONFIG["hashed_columns"]:
            sample = (
                model.df_lookup[f]
                .sample(n=1, random_state=self.random.randrange(1_000_000))
                .iloc[0]
            )
            setattr(self, f.lower(), sample)

        # --- fake names ---
        row = model.df_lookup.sample(
            n=1, random_state=self.random.randrange(1_000_000)
        ).iloc[0]
        self.eigenaarscode = row["EIGENAARSCODE"]
        self.gebruikerscode = row["GEBRUIKERSCODE"]
        self.eigendom_ = row["EIGENDOM_"]
        self.eigenaarsnaam = model.fake_owner_map.get(self.eigenaarscode)
        self.gebruikersnaam = model.fake_user_map.get(self.gebruikerscode)

        # --- synthetic IDs ---
        for attr, key in CONFIG["synthetic_ids"].items():
            setattr(self, attr.lower(), getattr(model, key) + self.unique_id)
        # --- one-to-one ---
        for code_field, name_field in CONFIG["one_to_one_mappings"].items():
            code_val = getattr(self, code_field.lower())
            setattr(
                self,
                name_field.lower(),
                model.code_to_name_mappings[code_field].get(code_val),
            )

        # --- one-to-many ---
        for inp, outp in CONFIG["one_to_many_mappings"].items():
            iv = getattr(self, inp.lower(), None)
            setattr(self, outp.lower(), model.value_to_class_mappings[outp].get(iv))

        # --- binning ---
        for f, fn in CONFIG["threshold_binning"].items():
            v = getattr(self, f.lower(), None)
            setattr(self, f"{f.lower()}_", globals()[fn](v))

        self.compose_address()

    def compose_address(self):
        """Compose a full address string from street, number, letter, and toevoeging."""
        # Compose huisnummer if present
        if self.huisnummer and self.huisnummer not in ("", "__MISSING__"):
            try:
                num = int(float(self.huisnummer))
                hnr = str(num)
            except (ValueError, TypeError):
                hnr = str(self.huisnummer)
            if self.huisletter and self.huisletter != "__MISSING__":
                hnr = f"{hnr}{self.huisletter}"
            parts = [self.street, hnr]
        else:
            parts = [self.street]
        # Only append toevoeging if present and not placeholder
        if self.toevoeging and self.toevoeging != "__MISSING__":
            try:
                parts.append(str(int(float(self.toevoeging))))
            except ValueError:
                # fallback to original toevoeging
                parts.append(self.toevoeging)
        self.adres = " ".join(parts)

    def step(self):
        """Perform a single model step (no operation for HouseholdAgent)."""
        pass

    def to_record(self) -> dict[str, Any]:
        """Convert agent attributes into a record dict for output.

        Returns
        -------
        dict of str to Any
            Dictionary containing output column values for this agent.
        """
        rec = {
            "ADRES": self.adres,
            "STRAATNAAM": self.street,
            "POSTCODE": self.postcode,
            "WOONPLAATS": self.city,
            "BUURTCODE": int(self.buurtcode),
            "WIJKCODE": int(self.wijkcode),
            "EIGENAARSNAAM": self.eigenaarsnaam,
            "GEBRUIKERSNAAM": self.gebruikersnaam,
            "EIGENDOM": self.eigendom_,
        }
        for f in CONFIG["kde_columns"]:
            v = getattr(self, f.lower())
            rec[f] = round(v, 1) if isinstance(v, float) else v
        for f in CONFIG["categorical_columns"]:
            v = getattr(self, f.lower())
            if f in {"HUISLETTER", "TOEVOEGING"}:
                # empty or placeholder yields empty string
                if not v or v == "__MISSING__":
                    rec[f] = ""
                elif f == "HUISLETTER":
                    rec[f] = v
                else:
                    # TOEVOEGING: try int convert, fallback to raw
                    try:
                        rec[f] = int(float(v))
                    except (ValueError, TypeError):
                        rec[f] = v
            elif f in {
                "HUISNUMMER",
                "BOUWJAAR",
                "WONINGTYPECODE",
                "WOZDATUM",
                "WOZPEIL",
            }:
                # Convert numeric strings with decimals to int
                try:
                    rec[f] = int(float(v))
                except (ValueError, TypeError):
                    rec[f] = v
            else:
                rec[f] = v
        # ensure synthetic IDs are integers to avoid float representation
        if hasattr(self, "bag_nummer") and self.bag_nummer is not None:
            rec["BAG_NUMMER"] = int(self.bag_nummer)
        if hasattr(self, "woz_nummer") and self.woz_nummer is not None:
            rec["WOZ_NUMMER"] = int(self.woz_nummer)
        for col in CONFIG["output_columns"]:
            if col not in rec:
                rec[col] = getattr(self, col.lower(), None)
        return rec


class AddressModel(Model):
    """Model generating household agents for synthetic address data.

    Parameters
    ----------
    num_agents : int
        Number of agents to generate per run.
    address_data : dict
        Mapping of city to address list from build_address_data.
    df_lookup : pd.DataFrame
        DataFrame containing lookup fields for agents.
    fake_owner_map : dict
        Mapping of owner codes to fake names.
    fake_user_map : dict
        Mapping of user codes to fake names.
    kdes : dict
        Precomputed Gaussian KDEs for numeric fields.
    categoricals : dict
        Precomputed weighted categories for categorical fields.
    code_to_name_mappings : dict
        Mapping of code columns to name columns for one-to-one mappings.
    value_to_class_mappings : dict
        Mapping of value columns to class columns for one-to-many mappings.
    bag_start : int
        Starting value for synthetic BAG_NUMMER.
    woz_start : int
        Starting value for synthetic WOZ_NUMMER.
    seed : int | None, optional
        Random seed for reproducibility (default None).
    """

    def __init__(
        self,
        num_agents: int,
        address_data: dict,
        df_lookup: pd.DataFrame,
        fake_owner_map: dict,
        fake_user_map: dict,
        kdes: dict,
        categoricals: dict,
        code_to_name_mappings: dict,
        value_to_class_mappings: dict,
        bag_start: int,
        woz_start: int,
        seed: int | None = None,
    ):
        """Set up the model with provided parameters and precomputed mappings."""
        super().__init__(seed=seed)
        self.num_agents = num_agents
        self.address_data = address_data
        self.df_lookup = df_lookup
        self.fake_owner_map = fake_owner_map
        self.fake_user_map = fake_user_map
        self.kdes = kdes
        self.categoricals = categoricals
        self.code_to_name_mappings = code_to_name_mappings
        self.value_to_class_mappings = value_to_class_mappings
        self.bag_start = bag_start
        self.woz_start = woz_start

        self.households = AgentSet([], random=self.random)
        for _ in range(self.num_agents):
            self.households.add(HouseholdAgent(self))

    def step(self):
        self.households.shuffle_do("step")

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            [a.to_record() for a in self.households if isinstance(a, HouseholdAgent)]
        )
        cols = CONFIG["output_columns"]
        return df[[c for c in cols if c in df.columns]]


def make_synthetic_housing(
    num_agents: int,
    df_lookup: pd.DataFrame,
    seed: int | None = None,
) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)
        fake.seed_instance(seed)

    # 1) hash IDs
    for col in CONFIG["hashed_columns"]:
        df_lookup[col] = df_lookup[col].apply(hash_id)

    # 2) fake names
    owner_map, user_map = build_fake_name_map(df_lookup)
    df_lookup["GEBRUIKERSNAAM"] = df_lookup["GEBRUIKERSCODE"].map(user_map)
    df_lookup["EIGENAARSNAAM"] = df_lookup["EIGENAARSCODE"].map(owner_map)

    # 3) build helpers
    addr = build_address_data(df_lookup)
    kdes = build_kdes(df_lookup, CONFIG["kde_columns"])
    cats = build_weighted_categories(df_lookup, CONFIG["categorical_columns"])
    one2one = build_mapping_dicts(df_lookup, CONFIG["one_to_one_mappings"])
    one2many = build_class_mappings(df_lookup, CONFIG["one_to_many_mappings"])
    # Ensure ID columns are numeric for computing starting values
    df_lookup["BAG_NUMMER"] = pd.to_numeric(df_lookup["BAG_NUMMER"], errors="coerce")
    df_lookup["WOZ_NUMMER"] = pd.to_numeric(df_lookup["WOZ_NUMMER"], errors="coerce")
    bag_start = int(df_lookup["BAG_NUMMER"].max() + 1)
    woz_start = int(df_lookup["WOZ_NUMMER"].dropna().max() + 1)

    # 4) run
    model = AddressModel(
        num_agents,
        addr,
        df_lookup,
        owner_map,
        user_map,
        kdes,
        cats,
        one2one,
        one2many,
        bag_start,
        woz_start,
        seed=seed,
    )
    model.step()
    return model.to_dataframe()
