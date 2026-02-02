"""QuantConnect code templates and API reference for using in agent instructions."""

# =============================================================================
# API REFERENCE - Used by Code Agent for accurate API usage
# =============================================================================

QC_API_REFERENCE = """
## QuantConnect API Quick Reference (0DTE Options)

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

QC_REFERENCE_TEMPLATE = '''from AlgorithmImports import *

class SPX0DTEStrategy(QCAlgorithm):
    """
    Base template for 0DTE SPX options strategies.
    This template compiles and runs - extend the marked sections only.
    """
    
    def Initialize(self):
        # === CONFIGURATION (modify values as needed) ===
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        # === SPX AND OPTIONS SETUP (do not modify structure) ===
        self.spx = self.AddIndex("SPX", Resolution.Minute)
        self.spx.SetDataNormalizationMode(DataNormalizationMode.Raw)
        
        option = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-30, 30).Expiration(0, 0))
        self.option_symbol = option.Symbol
        
        # === STRATEGY PARAMETERS (modify as needed) ===
        self.entry_hour = 10
        self.entry_minute = 0
        self.exit_hour = 15
        self.exit_minute = 45
        self.profit_target = 0.50
        self.stop_loss = 2.0
        self.max_risk_percent = 0.02
        
        # === TRACKING VARIABLES ===
        self.daily_entry_done = False
        
        # === SCHEDULING (modify times as needed) ===
        self.Schedule.On(
            self.DateRules.EveryDay(self.spx.Symbol),
            self.TimeRules.At(self.entry_hour, self.entry_minute),
            self.TryEntry
        )
        self.Schedule.On(
            self.DateRules.EveryDay(self.spx.Symbol),
            self.TimeRules.At(self.exit_hour, self.exit_minute),
            self.ExitAllPositions
        )
        self.Schedule.On(
            self.DateRules.EveryDay(self.spx.Symbol),
            self.TimeRules.AfterMarketOpen(self.spx.Symbol, 1),
            self.ResetDaily
        )
    
    def ResetDaily(self):
        """Reset daily tracking variables."""
        self.daily_entry_done = False
    
    def OnData(self, data: Slice):
        """Process incoming data - use for monitoring and mid-day exits."""
        # === OPTION CHAIN ACCESS PATTERN (do not modify) ===
        if not data.OptionChains:
            return
        
        chain = data.OptionChains.get(self.option_symbol)
        if chain is None:
            return
        
        # === FILTER FOR 0DTE (do not modify) ===
        contracts = [c for c in chain if c.Expiry.date() == self.Time.date()]
        if not contracts:
            return
        
        # === POSITION MONITORING (implement your logic) ===
        self.MonitorPositions(contracts)
    
    def TryEntry(self):
        """Scheduled entry attempt."""
        if self.daily_entry_done:
            return
        
        if not self.CanTrade():
            return
        
        # Get current option chain
        chain = self.CurrentSlice.OptionChains.get(self.option_symbol)
        if chain is None:
            self.Debug(f"{self.Time}: No option chain available")
            return
        
        contracts = [c for c in chain if c.Expiry.date() == self.Time.date()]
        if not contracts:
            self.Debug(f"{self.Time}: No 0DTE contracts available")
            return
        
        # === IMPLEMENT ENTRY LOGIC HERE ===
        self.ExecuteEntry(contracts)
    
    def CanTrade(self) -> bool:
        """Check if trading conditions are met. Override as needed."""
        return True
    
    def ExecuteEntry(self, contracts):
        """Execute entry logic. IMPLEMENT BASED ON SPECIFICATION."""
        # Filter by right
        puts = [c for c in contracts if c.Right == OptionRight.Put]
        calls = [c for c in contracts if c.Right == OptionRight.Call]
        
        # Filter for valid Greeks
        puts = [c for c in puts if c.Greeks and c.Greeks.Delta]
        calls = [c for c in calls if c.Greeks and c.Greeks.Delta]
        
        # === IMPLEMENT STRATEGY-SPECIFIC LOGIC HERE ===
        
        self.daily_entry_done = True

    
    def MonitorPositions(self, contracts):
        """Monitor open positions for exit conditions. IMPLEMENT THIS METHOD."""
        for kvp in self.Portfolio:
            holding = kvp.Value
            if not holding.Invested:
                continue
            if holding.Type != SecurityType.IndexOption:
                continue
            
            # === YOUR EXIT LOGIC HERE ===
            pass
    
    def ExitAllPositions(self):
        """Exit all positions at scheduled time."""
        self.Liquidate(tag="End of day exit")
        self.daily_entry_done = False

    # === HELPER METHODS ===
    
    def GetContractsByDelta(self, contracts, option_right, target_delta: float, tolerance: float = 0.05):
        """
        Find contracts near a target delta.
        
        Args:
            contracts: List of option contracts
            option_right: OptionRight.Put or OptionRight.Call
            target_delta: Target delta (use positive value)
            tolerance: Acceptable deviation from target
            
        Returns:
            List of contracts within tolerance, sorted by proximity to target
        """
        filtered = [
            c for c in contracts 
            if c.Right == option_right 
            and c.Greeks 
            and c.Greeks.Delta is not None
        ]
        
        if option_right == OptionRight.Put:
            filtered = [c for c in filtered if abs(abs(c.Greeks.Delta) - target_delta) <= tolerance]
            filtered.sort(key=lambda c: abs(abs(c.Greeks.Delta) - target_delta))
        else:
            filtered = [c for c in filtered if abs(c.Greeks.Delta - target_delta) <= tolerance]
            filtered.sort(key=lambda c: abs(c.Greeks.Delta - target_delta))
        
        return filtered
'''

# =============================================================================
# FEW-SHOT EXAMPLE - Working strategy for pattern matching >>> Currently not used, may add later
# =============================================================================

FEW_SHOT_EXAMPLE = '''
## WORKING EXAMPLE: Put Credit Spread

**Specification:** Sell 0DTE put spread on SPX at 10AM ET, 0.10 delta short put, 
5-strike-wide spread, exit at 50% profit or 3:45PM.

**Implementation:**
```python
from AlgorithmImports import *

class PutCreditSpread(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        self.spx = self.AddIndex("SPX", Resolution.Minute)
        self.spx.SetDataNormalizationMode(DataNormalizationMode.Raw)
        
        option = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        option.SetFilter(lambda u: u.IncludeWeeklys().Strikes(-30, 30).Expiration(0, 0))
        self.option_symbol = option.Symbol
        
        self.target_delta = 0.10
        self.spread_width = 5
        self.profit_target = 0.50
        self.daily_entry_done = False
        
        self.Schedule.On(self.DateRules.EveryDay(self.spx.Symbol), 
                         self.TimeRules.At(10, 0), self.TryEntry)
        self.Schedule.On(self.DateRules.EveryDay(self.spx.Symbol),
                         self.TimeRules.At(15, 45), self.ExitAll)
        self.Schedule.On(self.DateRules.EveryDay(self.spx.Symbol),
                         self.TimeRules.AfterMarketOpen(self.spx.Symbol, 1), self.Reset)
    
    def Reset(self):
        self.daily_entry_done = False
    
    def OnData(self, data: Slice):
        if not data.OptionChains:
            return
        chain = data.OptionChains.get(self.option_symbol)
        if chain is None:
            return
        
        for kvp in self.Portfolio:
            h = kvp.Value
            if h.Invested and h.Type == SecurityType.IndexOption:
                if h.UnrealizedProfitPercent >= self.profit_target:
                    self.Liquidate(h.Symbol, "Profit target")
    
    def TryEntry(self):
        if self.daily_entry_done:
            return
        
        chain = self.CurrentSlice.OptionChains.get(self.option_symbol)
        if chain is None:
            return
        
        contracts = [c for c in chain if c.Expiry.date() == self.Time.date()]
        puts = [c for c in contracts if c.Right == OptionRight.Put and c.Greeks and c.Greeks.Delta]
        
        if not puts:
            return
        
        short_put = min(puts, key=lambda c: abs(abs(c.Greeks.Delta) - self.target_delta))
        
        target_strike = short_put.Strike - self.spread_width
        long_puts = [c for c in puts if c.Strike == target_strike]
        
        if not long_puts:
            lower_puts = [c for c in puts if c.Strike < short_put.Strike]
            if not lower_puts:
                return
            long_put = max(lower_puts, key=lambda c: c.Strike)
        else:
            long_put = long_puts[0]
        
        self.MarketOrder(short_put.Symbol, -1)
        self.MarketOrder(long_put.Symbol, 1)
        self.daily_entry_done = True
        self.Debug(f"{self.Time}: Spread opened - Short {short_put.Strike}, Long {long_put.Strike}")
    
    def ExitAll(self):
        self.Liquidate(tag="EOD exit")
        self.daily_entry_done = False
```
'''