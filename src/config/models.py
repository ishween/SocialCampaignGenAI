"""Pydantic models for campaign brief validation."""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ProductConfig(BaseModel):
    name: str = Field(..., description="Product name")
    description: str = Field(..., description="Product description for prompt generation")
    existing_asset: Optional[str] = Field(None, description="Path to existing product image")
    style_keywords: list[str] = Field(default_factory=list, description="Visual style hints")
    label_text: Optional[str] = Field(
        None,
        description=(
            "Text to render ON the product label in the generated image. "
            "Multiline supported (use \\n). When set, the image model is instructed "
            "to include this text as part of the product design, not as a post-process overlay. "
            "Example: 'HydraFresh\\nHydrate+\\n610mg Electrolytes'. "
            "Leave null for clean product-only shots or when using existing_asset."
        )
    )
    background_scene: Optional[str] = Field(
        None,
        description=(
            "Scene/environment description for the generated image background. "
            "Drives the lifestyle context around the product. "
            "Example: 'dynamic sports stadium, basketball player dunking, crowd cheering, "
            "dramatic floodlights, motion blur'. "
            "Leave null for clean studio/white background."
        )
    )

    @field_validator("existing_asset")
    @classmethod
    def asset_must_exist_if_provided(cls, v):
        if v is not None and not Path(v).exists():
            raise ValueError(f"existing_asset path does not exist: {v}")
        return v


class RegionConfig(BaseModel):
    code: str = Field(..., description="ISO region code, e.g. US, FR, JP")
    language: str = Field(default="en", description="ISO language code")
    locale_name: str = Field(default="", description="Human-readable locale for prompts")
    message_override: Optional[str] = Field(
        None,
        description="Localized campaign message for this region. Falls back to brief.message if not set."
    )


class CampaignBrief(BaseModel):
    name: str = Field(..., description="Campaign name")
    message: str = Field(..., description="Campaign message to display on creatives")
    products: list[ProductConfig] = Field(..., min_length=1, description="Products to generate assets for")
    target_audience: str = Field(..., description="Target audience description")
    regions: list[RegionConfig] = Field(default_factory=lambda: [RegionConfig(code="US", language="en")])
    prohibited_words: list[str] = Field(default_factory=list, description="Words that must not appear in message")
    aspect_ratios: list[str] = Field(
        default=["1:1", "9:16", "16:9"],
        description="Aspect ratios to generate"
    )
    brand_style: str = Field(default="modern, clean, professional", description="Overall brand style keywords")

    @field_validator("products")
    @classmethod
    def at_least_one_product(cls, v):
        if len(v) < 1:
            raise ValueError("Campaign brief must include at least one product")
        return v

    @field_validator("aspect_ratios")
    @classmethod
    def valid_aspect_ratios(cls, v):
        allowed = {"1:1", "9:16", "16:9", "4:5", "2:3"}
        for ratio in v:
            if ratio not in allowed:
                raise ValueError(f"Unsupported aspect ratio '{ratio}'. Choose from: {allowed}")
        return v


class BrandConfig(BaseModel):
    name: str = Field(..., description="Brand name")
    logo_path: Optional[str] = Field(None, description="Path to logo file (PNG with transparency)")
    primary_colors: list[str] = Field(default_factory=list, description="Brand hex colors, e.g. #FF0000")
    font_path: Optional[str] = Field(None, description="Path to brand font file (.ttf)")
    font_size_base: int = Field(default=48, description="Base font size for text overlay")
    text_color: str = Field(default="#FFFFFF", description="Text overlay color (hex)")
    text_shadow: bool = Field(default=True, description="Add shadow behind text for readability")
