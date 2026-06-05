"""QuantConnect code templates and API reference for using in agent instructions."""

# =============================================================================
# API REFERENCE - Used by Code Agent for accurate API usage
# =============================================================================

QC_API_REFERENCE = """
## QuantConnect API Quick Reference (0DTE Options)

### Time Zone
```python
# Set algorithm to New York time (required for SPX)
self.SetTimeZone("America/New_York")
```

### Option Chain Access
```python
# In OnData - ALWAYS use this pattern
if not data.OptionChains:
    return
chain = data.OptionChains.get(self.option_symbol)  # Use .get(), not []
if chain is None:
    return
```

### Contract Filtering
```python
# Filter for 0DTE
contracts = [c for c in chain if c.Expiry.date() == self.Time.date()]

# Filter by right
puts = [c for c in contracts if c.Right == OptionRight.Put]
calls = [c for c in contracts if c.Right == OptionRight.Call]

# Filter with Greeks (ALWAYS check for None)
with_greeks = [c for c in contracts if c.Greeks and c.Greeks.Delta is not None]
```

### Greeks Properties
```python
contract.Greeks.Delta      # float or None
contract.Greeks.Gamma      # float or None  
contract.Greeks.Theta      # float or None
contract.Greeks.Vega       # float or None
contract.Greeks.ImpliedVolatility  # float or None
```

### Order Methods
```python
self.MarketOrder(symbol, quantity)           # Positive = buy, Negative = sell
self.LimitOrder(symbol, quantity, limit_price)
self.Liquidate(symbol)                       # Close specific position
self.Liquidate()                             # Close all positions
```

### Position Information
```python
self.Portfolio[symbol].Invested              # bool
self.Portfolio[symbol].Quantity              # int (positive = long, negative = short)
self.Portfolio[symbol].UnrealizedProfitPercent  # float
self.Portfolio[symbol].AveragePrice          # float
```

### Scheduling
```python
self.Schedule.On(
    self.DateRules.EveryDay(symbol),
    self.TimeRules.At(hour, minute),
    self.MethodName
)
self.Schedule.On(
    self.DateRules.EveryDay(symbol),
    self.TimeRules.AfterMarketOpen(symbol, minutes_after),
    self.MethodName
)
```

### Index Option Setup
```python
# In Initialize()
self.spx = self.AddIndex("SPX", Resolution.Minute)
self.spx.SetDataNormalizationMode(DataNormalizationMode.Raw)

option = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-30, 30).Expiration(0, 0))
self.option_symbol = option.Symbol
```

### Common Errors to Avoid
- Never use `data.OptionChains[symbol]` - use `.get(symbol)`
- Never access Greeks without null check
- Never use `from QuantConnect import *` - use `from AlgorithmImports import *`
- Never use `datetime.now()` - use `self.Time`
"""

# =============================================================================
# REFERENCE TEMPLATE - Compilable skeleton for 0DTE strategies
# =============================================================================

"""QuantConnect code template - minimal structure only."""

QC_REFERENCE_TEMPLATE = '''from AlgorithmImports import *
from datetime import timedelta

class SPX0DTEStrategy(QCAlgorithm):
    
    def Initialize(self):
        # === CONFIGURATION ===
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)
        self.SetTimeZone("America/New_York")
        
        # Realistic transaction costs
        self.SetSecurityInitializer(self.CustomSecurityInitializer)

        # === SPX AND OPTIONS SETUP ===
        self.spx = self.AddIndex("SPX", Resolution.Minute)
        self.spx.SetDataNormalizationMode(DataNormalizationMode.Raw)

        self.option = self.AddIndexOption(self.spx.Symbol, "SPXW", Resolution.Minute)
        self.option.SetFilter(lambda u: u.Strikes(-50, 50).Expiration(0, 0))
        self.option_symbol = self.option.Symbol

        # === MARGIN (recognize spreads as defined-risk) ===
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0
        
        # === TRACKING ===
        self.daily_entry_done = False
        self.cached_contracts = []
        self.cache_time = self.Time
        
        # === SCHEDULING (modify times as needed) ===
        self.Schedule.On(
            self.DateRules.EveryDay(self.spx.Symbol),
            self.TimeRules.At(10, 0),
            self.TryEntry
        )
        self.Schedule.On(
            self.DateRules.EveryDay(self.spx.Symbol),
            self.TimeRules.At(15, 45),
            self.ExitAllPositions
        )
        self.Schedule.On(
            self.DateRules.EveryDay(self.spx.Symbol),
            self.TimeRules.AfterMarketOpen(self.spx.Symbol, 1),
            self.ResetDaily
        )

    def CustomSecurityInitializer(self, security):
        if security.Type == SecurityType.IndexOption:
            security.SetFeeModel(ConstantFeeModel(0.65))
            
    def ResetDaily(self):
        self.daily_entry_done = False
        self.cached_contracts = []
    
    def OnData(self, data: Slice):
        # Cache option chain for use in scheduled functions
        if data.OptionChains:
            chain = data.OptionChains.get(self.option_symbol)
            if chain:
                contracts = [c for c in chain if c.Expiry.date() == self.Time.date()]
                if contracts:
                    self.cached_contracts = contracts
                    self.cache_time = self.Time
        
        # Monitor positions if needed
        if self.Portfolio.Invested:
            self.MonitorPositions()
    
    def TryEntry(self):
        if self.daily_entry_done:
            return
        
        if self.Portfolio.Invested:
            return
        
        if not self.CanTrade():
            return
        
        contracts = self.cached_contracts
        if not contracts:
            self.Debug(f"{self.Time}: No contracts available")
            return
        
        self.ExecuteEntry(contracts)
    
    def CanTrade(self) -> bool:
        # IMPLEMENT: Your entry conditions
        return True
    
    def ExecuteEntry(self, contracts):
        # IMPLEMENT: Your entry logic
        # MUST: Handle case where Greeks are None
        # MUST: Log decisions for debugging
        # MUST: Set self.daily_entry_done = True after entry
        pass
    
    def MonitorPositions(self):
        # IMPLEMENT: Your exit logic (profit target, stop loss)
        pass
    
    def ExitAllPositions(self):
        self.Liquidate(tag="End of day exit")
        self.daily_entry_done = False
'''