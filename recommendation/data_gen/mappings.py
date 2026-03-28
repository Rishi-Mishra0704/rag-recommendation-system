from recommendation.data_gen.schemas import Industry

# ---------------------------------------------------------------------------
# Tag map: deterministic tags per industry
# ---------------------------------------------------------------------------
INDUSTRY_TAGS: dict[Industry, list[str]] = {
    Industry.AGRICULTURE: ["agriculture", "farming", "crop", "agri-trade", "harvest", "organic"],
    Industry.AUTOMOTIVE: ["automotive", "spare-parts", "vehicles", "oem", "aftermarket", "fleet"],
    Industry.CHEMICALS: ["chemicals", "industrial", "specialty-chemicals", "b2b", "raw-materials", "processing"],
    Industry.CONSTRUCTION: ["construction", "building", "infrastructure", "materials", "contractor", "civil"],
    Industry.ELECTRONICS: ["electronics", "telecom", "components", "hardware", "pcb", "semiconductors"],
    Industry.ENERGY: ["energy", "oil-gas", "renewables", "power", "solar", "utilities"],
    Industry.FOOD_BEVERAGE: ["food", "beverages", "fmcg", "packaged-goods", "supply-chain", "ingredients"],
    Industry.HEALTHCARE: ["healthcare", "medical", "devices", "diagnostics", "hospital", "wellness"],
    Industry.LOGISTICS: ["logistics", "freight", "supply-chain", "shipping", "warehousing", "trade"],
    Industry.MANUFACTURING: ["manufacturing", "factory", "production", "industrial", "oem", "b2b"],
    Industry.METALS_MINING: ["metals", "mining", "steel", "aluminum", "raw-materials", "commodities"],
    Industry.PHARMACEUTICALS: ["pharma", "drugs", "biologics", "api", "nutraceuticals", "biotech"],
    Industry.RETAIL: ["retail", "wholesale", "distribution", "fmcg", "consumer-goods", "e-commerce"],
    Industry.TEXTILE_APPAREL: ["textiles", "apparel", "fabric", "garments", "fashion", "yarn"],
    Industry.TECHNOLOGY: ["technology", "software", "saas", "it-services", "digital", "cloud"],
    Industry.WOOD_FURNITURE: ["wood", "furniture", "timber", "panels", "flooring", "interior"],
}

# ---------------------------------------------------------------------------
# Country → trade regions mapping
# ---------------------------------------------------------------------------
COUNTRY_REGION_MAP: dict[str, list[str]] = {
    "united states": ["North America"],
    "canada": ["North America"],
    "mexico": ["North America", "South America"],
    "brazil": ["South America"],
    "argentina": ["South America"],
    "colombia": ["South America"],
    "chile": ["South America"],
    "united kingdom": ["Western Europe"],
    "germany": ["Western Europe"],
    "france": ["Western Europe"],
    "italy": ["Western Europe"],
    "spain": ["Western Europe"],
    "netherlands": ["Western Europe"],
    "sweden": ["Western Europe"],
    "norway": ["Western Europe"],
    "denmark": ["Western Europe"],
    "switzerland": ["Western Europe"],
    "austria": ["Western Europe"],
    "belgium": ["Western Europe"],
    "portugal": ["Western Europe"],
    "poland": ["Eastern Europe"],
    "ukraine": ["Eastern Europe"],
    "czech republic": ["Eastern Europe"],
    "romania": ["Eastern Europe"],
    "hungary": ["Eastern Europe"],
    "russia": ["Eastern Europe", "Central Asia"],
    "turkey": ["Middle East", "Western Europe"],
    "saudi arabia": ["Middle East"],
    "united arab emirates": ["Middle East"],
    "israel": ["Middle East"],
    "iran": ["Middle East"],
    "iraq": ["Middle East"],
    "egypt": ["Middle East", "Africa"],
    "south africa": ["Africa"],
    "nigeria": ["Africa"],
    "kenya": ["Africa"],
    "ghana": ["Africa"],
    "ethiopia": ["Africa"],
    "india": ["South Asia", "India"],
    "pakistan": ["South Asia"],
    "bangladesh": ["South Asia"],
    "sri lanka": ["South Asia"],
    "indonesia": ["Southeast Asia"],
    "malaysia": ["Southeast Asia"],
    "thailand": ["Southeast Asia"],
    "vietnam": ["Southeast Asia"],
    "philippines": ["Southeast Asia"],
    "singapore": ["Southeast Asia"],
    "china": ["East Asia"],
    "japan": ["East Asia"],
    "south korea": ["East Asia"],
    "taiwan": ["East Asia"],
    "hong kong": ["East Asia"],
    "australia": ["Oceania"],
    "new zealand": ["Oceania"],
    "kazakhstan": ["Central Asia"],
    "uzbekistan": ["Central Asia"],
}


def country_to_regions(country: str) -> list[str]:
    key = country.strip().lower()
    return COUNTRY_REGION_MAP.get(key, ["Global"])
