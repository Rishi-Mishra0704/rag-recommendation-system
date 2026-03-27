# schemas.py - Data schemas and models for generated data
from __future__ import annotations

from enum import Enum
from uuid import UUID
from typing import Optional

from pydantic import BaseModel, Field


class Industry:
    AGRICULTURE = "Agriculture"
    AUTOMOTIVE = "Automotive"
    CHEMICALS = "Chemicals"
    CONSTRUCTION = "Construction"
    ELECTRONICS = "Electronics"
    ENERGY = "Energy"
    FOOD_BEVERAGE = "Food & Beverage"
    HEALTHCARE = "Healthcare"
    LOGISTICS = "Logistics"
    MANUFACTURING = "Manufacturing"
    METALS_MINING = "Metals & Mining"
    PHARMACEUTICALS = "Pharmaceuticals"
    RETAIL = "Retail"
    TEXTILE_APPAREL = "Textile & Apparel"
    TECHNOLOGY = "Technology"
    WOOD_FURNITURE = "Wood & Furniture"


class Category:
    MANUFACTURER = "manufacturer"
    TRADER = "trader"
    DISTRIBUTOR = "distributor"
    SERVICE_PROVIDER = "service_provider"
    SUPPLIER = "supplier"
    BROKER = "broker"
    WHOLESALER = "wholesaler"


class TradeType:
    DOMESTIC = "domestic"
    INTERNATIONAL = "international"
    BOTH = "both"


class Role:
    BUYER = "buyer"
    SELLER = "seller"
    IMPORTER = "importer"
    EXPORTER = "exporter"
    AGENT = "agent"
    RESELLER = "reseller"


SUB_INDUSTRY_MAP = {
    Industry.AGRICULTURE: ["Crop Production", "Livestock", "Aquaculture", "Agri-inputs", "Farming Equipment"],
    Industry.AUTOMOTIVE: ["OEM Parts", "Aftermarket Parts", "Electric Vehicles", "Commercial Vehicles", "Auto Accessories"],
    Industry.CHEMICALS: ["Specialty Chemicals", "Petrochemicals", "Agrochemicals", "Adhesives", "Industrial Coatings"],
    Industry.CONSTRUCTION: ["Building Materials", "Infrastructure", "Interior Fit-out", "Civil Engineering", "Architecture & Planning"],
    Industry.ELECTRONICS: ["Consumer Electronics", "Industrial Electronics", "Semiconductors", "PCB", "Telecommunications Equipment"],
    Industry.ENERGY: ["Renewables", "Oil & Gas", "Power Equipment", "Energy Storage", "Utilities"],
    Industry.FOOD_BEVERAGE: ["Packaged Foods", "Beverages", "Raw Ingredients", "Frozen Foods", "Food Production"],
    Industry.HEALTHCARE: ["Medical Devices", "Diagnostics", "Hospital Supplies", "Wellness", "Medical Practice"],
    Industry.LOGISTICS: ["Freight Forwarding", "Warehousing", "Cold Chain", "Last-Mile Delivery", "Import & Export"],
    Industry.MANUFACTURING: ["Contract Manufacturing", "CNC Machining", "Injection Moulding", "Assembly", "Industrial Automation"],
    Industry.METALS_MINING: ["Steel", "Aluminum", "Copper", "Precious Metals", "Mining Equipment"],
    Industry.PHARMACEUTICALS: ["Generic Drugs", "Active Ingredients", "Nutraceuticals", "Biologics", "Biotechnology"],
    Industry.RETAIL: ["E-commerce", "FMCG", "Wholesale", "Private Label", "Consumer Goods"],
    Industry.TEXTILE_APPAREL: ["Yarn & Fabric", "Garments", "Technical Textiles", "Leather Goods", "Apparel Manufacturing"],
    Industry.TECHNOLOGY: ["SaaS", "Hardware", "IT Services", "Cybersecurity", "Computer Software"],
    Industry.WOOD_FURNITURE: ["Timber", "Panels", "Furniture", "Flooring", "Building Materials"],
}

# Maps dataset industry strings (lowercase) → our Industry enum values.
# Unmapped industries are skipped during transform.
INDUSTRY_MAPPING = {
    # Technology
    "information technology and services": Industry.TECHNOLOGY,
    "computer software": Industry.TECHNOLOGY,
    "internet": Industry.TECHNOLOGY,
    "computer & network security": Industry.TECHNOLOGY,
    "computer games": Industry.TECHNOLOGY,
    "information services": Industry.TECHNOLOGY,

    # Construction
    "construction": Industry.CONSTRUCTION,
    "architecture & planning": Industry.CONSTRUCTION,
    "building materials": Industry.CONSTRUCTION,
    "civil engineering": Industry.CONSTRUCTION,

    # Automotive
    "automotive": Industry.AUTOMOTIVE,

    # Food & Beverage
    "food & beverages": Industry.FOOD_BEVERAGE,
    "food production": Industry.FOOD_BEVERAGE,
    "wine and spirits": Industry.FOOD_BEVERAGE,

    # Manufacturing
    "mechanical or industrial engineering": Industry.MANUFACTURING,
    "electrical/electronic manufacturing": Industry.MANUFACTURING,
    "machinery": Industry.MANUFACTURING,
    "industrial automation": Industry.MANUFACTURING,
    "plastics": Industry.MANUFACTURING,
    "business supplies and equipment": Industry.MANUFACTURING,
    "printing": Industry.MANUFACTURING,
    "aviation & aerospace": Industry.MANUFACTURING,

    # Energy
    "oil & energy": Industry.ENERGY,
    "renewables & environment": Industry.ENERGY,
    "utilities": Industry.ENERGY,

    # Retail
    "retail": Industry.RETAIL,
    "wholesale": Industry.RETAIL,
    "consumer goods": Industry.RETAIL,
    "import and export": Industry.RETAIL,

    # Healthcare
    "hospital & health care": Industry.HEALTHCARE,
    "health, wellness and fitness": Industry.HEALTHCARE,
    "medical devices": Industry.HEALTHCARE,
    "medical practice": Industry.HEALTHCARE,
    "mental health care": Industry.HEALTHCARE,

    # Electronics / Telecom
    "telecommunications": Industry.ELECTRONICS,
    "consumer electronics": Industry.ELECTRONICS,
    "broadcast media": Industry.ELECTRONICS,

    # Chemicals
    "chemicals": Industry.CHEMICALS,
    "environmental services": Industry.CHEMICALS,

    # Pharmaceuticals
    "pharmaceuticals": Industry.PHARMACEUTICALS,
    "biotechnology": Industry.PHARMACEUTICALS,

    # Metals & Mining
    "mining & metals": Industry.METALS_MINING,

    # Logistics
    "logistics and supply chain": Industry.LOGISTICS,
    "transportation/trucking/railroad": Industry.LOGISTICS,
    "international trade and development": Industry.LOGISTICS,
    "maritime": Industry.LOGISTICS,
    "airlines/aviation": Industry.LOGISTICS,

    # Textile & Apparel
    "textiles": Industry.TEXTILE_APPAREL,
    "apparel & fashion": Industry.TEXTILE_APPAREL,
    "sporting goods": Industry.TEXTILE_APPAREL,

    # Agriculture
    "farming": Industry.AGRICULTURE,

    # Wood & Furniture
    "furniture": Industry.WOOD_FURNITURE,

    # Chemicals (extra)
    "cosmetics": Industry.CHEMICALS,
}

TRADE_REGIONS = [
    "North America", "South America", "Western Europe", "Eastern Europe",
    "Middle East", "Africa", "South Asia", "Southeast Asia",
    "East Asia", "Central Asia", "Oceania", "India", "Global",
]

CERTIFICATES = [
    "ISO 9001", "ISO 14001", "ISO 45001", "CE", "FDA", "GMP",
    "Halal", "Kosher", "Fair Trade", "REACH", "RoHS", "BSCI",
    "SA8000", "GOTS", "FSC", "BRC", "HACCP",
]


class Business(BaseModel):
    id: UUID
    owner_id: UUID
    verified: bool
    reputation: float = Field(ge=0.0, le=5.0)
    name: str
    industry: str
    sub_industry: str
    roles: list[str]
    category: str
    trade_type: str
    location: str
    trade_regions: list[str]
    capacity: str
    certificates: list[str]
    tags: list[str]
    description: str
    partner_goals: str
    profile_embedding: Optional[list[float]] = None

    def to_generation_dict(self) -> dict:
        """Serializes all fields except profile_embedding for LLM generation / storage."""
        return self.model_dump(
            mode="json",
            exclude={"profile_embedding"},
        )
