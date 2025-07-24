from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class CryptoPriceInput(BaseModel):
    symbol: str = Field(description="Cryptocurrency symbol to look up (e.g., BTC, ETH)")


class CryptoPriceTool(BaseTool):
    name: str = "crypto_price_finder"
    description: str = "Find current price of cryptocurrencies in USD"
    args_schema: Type[BaseModel] = CryptoPriceInput

    def _run(self, symbol: str) -> str:
        # Mock cryptocurrency price database with recent/obscure coins
        crypto_prices = {
            "btc": "$67,234.50",
            "bitcoin": "$67,234.50",
            "eth": "$3,456.78",
            "ethereum": "$3,456.78",
            "sol": "$198.45",
            "solana": "$198.45",
            "ada": "$0.89",
            "cardano": "$0.89",
            "dot": "$12.34",
            "polkadot": "$12.34",
            "avax": "$45.67",
            "avalanche": "$45.67",
            "link": "$23.89",
            "chainlink": "$23.89",
            "matic": "$1.12",
            "polygon": "$1.12",
            "ftm": "$0.78",
            "fantom": "$0.78",
            "icp": "$15.67",
            "internet computer": "$15.67",
            "near": "$7.89",
            "sui": "$4.23",
            "apt": "$11.45",
            "aptos": "$11.45",
            "arb": "$2.34",
            "arbitrum": "$2.34",
            "op": "$3.21",
            "optimism": "$3.21",
        }

        symbol_lower = symbol.lower().strip()
        for key, price in crypto_prices.items():
            if key.lower() == symbol_lower:
                return f"Current price of {symbol.upper()}: {price} USD"

        return f"Sorry, I couldn't find price information for {symbol}. Please check the symbol and try again."
