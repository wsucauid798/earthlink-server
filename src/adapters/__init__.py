"""Earth connection adapters.

Each adapter connects the virtual world to a real Earth data source.
Adapters are world infrastructure — agents never interact with them directly.
"""

from .base import EarthAdapter, EarthFact

# Key-free adapters — the world's connection to Earth civilisation
from .wikipedia import WikipediaAdapter

# S40: News / current events
from .news import BBCNewsAdapter, ReutersAdapter

# S42: Government / institutional
from .government import (
    LegislationAdapter,
    ONSAdapter,
    DataGovUKAdapter,
    ParliamentAdapter,
    PoliceDataAdapter,
)

# S43: Public discourse
from .discourse import StackExchangeAdapter, MastodonAdapter

# S44: Cultural / social
from .cultural import BankHolidaysAdapter, OverpassAdapter

# S45: Historical / archival
from .historical import (
    InternetArchiveAdapter,
    GutenbergAdapter,
    BritishMuseumAdapter,
    HistoricEnglandAdapter,
    NationalArchivesAdapter,
)

# S46: Search / discovery
from .duckduckgo import DuckDuckGoAdapter

# S47: Social / real-time
from .social import BlueskyAdapter

# S48: LLM substrate
from .ollama import OllamaAdapter

# S49: Wikidata / DBpedia
from .wikidata import WikidataAdapter
from .dbpedia import DBpediaAdapter


def all_key_free_adapters() -> list[EarthAdapter]:
    """Instantiate all key-free adapters with default config.

    Returns a list of adapter instances ready to register with EarthProxy.
    """
    return [
        WikipediaAdapter(),
        # News
        BBCNewsAdapter(),
        ReutersAdapter(),
        # Government
        LegislationAdapter(),
        ONSAdapter(),
        DataGovUKAdapter(),
        ParliamentAdapter(),
        PoliceDataAdapter(),
        # Public discourse
        StackExchangeAdapter(),
        MastodonAdapter(),
        # Cultural
        BankHolidaysAdapter(),
        OverpassAdapter(),
        # Historical
        InternetArchiveAdapter(),
        GutenbergAdapter(),
        BritishMuseumAdapter(),
        HistoricEnglandAdapter(),
        NationalArchivesAdapter(),
        # Search
        DuckDuckGoAdapter(),
        # Social
        BlueskyAdapter(),
        # LLM
        OllamaAdapter(),
        # Structured knowledge
        WikidataAdapter(),
        DBpediaAdapter(),
    ]
