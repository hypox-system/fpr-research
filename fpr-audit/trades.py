"""
Canonical trade return computation.

Delta 33: This is the SINGLE source of truth for trade return calculation.
ALL pipeline code (backtest, metrics, permutation test, DSR) MUST use these functions.
No inline return calculations allowed.

Delta 4, 20: Fill model is side-aware. Slippage ALWAYS makes fill worse.
"""

from typing import Literal


def compute_fill_price(
    open_price: float,
    side: Literal['long', 'short'],
    is_entry: bool,
    slippage_rate: float
) -> float:
    """
    Compute fill price with side-aware slippage.

    Delta 4, 20: Slippage ALWAYS makes fill worse.
    - Long entry: open * (1 + slip) - slippage makes buy more expensive
    - Long exit: open * (1 - slip) - slippage makes sell cheaper
    - Short entry: open * (1 - slip) - slippage makes sell cheaper
    - Short exit: open * (1 + slip) - slippage makes buy more expensive

    Args:
        open_price: Next bar's open price
        side: 'long' or 'short'
        is_entry: True for entry, False for exit
        slippage_rate: Slippage rate in decimal (e.g., 0.0001 for 0.01%)

    Returns:
        Fill price after slippage
    """
    if side == 'long':
        if is_entry:
            # Long entry: buying, slippage makes it more expensive
            return open_price * (1 + slippage_rate)
        else:
            # Long exit: selling, slippage makes it cheaper
            return open_price * (1 - slippage_rate)
    else:  # short
        if is_entry:
            # Short entry: selling, slippage makes it cheaper
            return open_price * (1 - slippage_rate)
        else:
            # Short exit: buying, slippage makes it more expensive
            return open_price * (1 + slippage_rate)


def compute_trade_return(
    entry_price: float,
    exit_price: float,
    side: Literal['long', 'short'],
    fee_rate: float,
    slippage_rate: float
) -> float:
    """
    Compute NET trade return after fees and slippage.

    Delta 33: This is the canonical trade return formula.
    ALL trade return calculations MUST use this function.

    Formulas:
    - Long: exit_fill/entry_fill - 1 - 2*fee_rate
      where entry_fill = open*(1+slip), exit_fill = open*(1-slip)
    - Short: entry_fill/exit_fill - 1 - 2*fee_rate
      where entry_fill = open*(1-slip), exit_fill = open*(1+slip)

    Args:
        entry_price: Entry bar's open price (before slippage)
        exit_price: Exit bar's open price (before slippage)
        side: 'long' or 'short'
        fee_rate: Taker fee rate in decimal (e.g., 0.0006 for 0.06%)
        slippage_rate: Slippage rate in decimal (e.g., 0.0001 for 0.01%)

    Returns:
        Net return as decimal (e.g., 0.01 for 1%)
    """
    # Compute fill prices with slippage
    entry_fill = compute_fill_price(entry_price, side, is_entry=True, slippage_rate=slippage_rate)
    exit_fill = compute_fill_price(exit_price, side, is_entry=False, slippage_rate=slippage_rate)

    # Compute gross return
    if side == 'long':
        gross_return = exit_fill / entry_fill - 1
    else:  # short
        gross_return = entry_fill / exit_fill - 1

    # Subtract fees (both entry and exit)
    net_return = gross_return - 2 * fee_rate

    return net_return


def compute_trade_return_from_fills(
    entry_fill: float,
    exit_fill: float,
    side: Literal['long', 'short'],
    fee_rate: float
) -> float:
    """
    Compute NET trade return from already-computed fill prices.

    Use this when fill prices have already been computed.

    Args:
        entry_fill: Entry fill price (after slippage)
        exit_fill: Exit fill price (after slippage)
        side: 'long' or 'short'
        fee_rate: Taker fee rate in decimal

    Returns:
        Net return as decimal
    """
    if side == 'long':
        gross_return = exit_fill / entry_fill - 1
    else:  # short
        gross_return = entry_fill / exit_fill - 1

    return gross_return - 2 * fee_rate


def validate_trade_params(
    fee_rate: float,
    slippage_rate: float
) -> None:
    """
    Validate trade parameters.

    Delta 1: Fee/slippage in decimal (rate), not percent.

    Raises:
        ValueError: If params are invalid
    """
    if fee_rate < 0:
        raise ValueError(f"fee_rate must be >= 0, got {fee_rate}")

    if fee_rate > 0.01:  # More than 1%
        raise ValueError(
            f"fee_rate {fee_rate} looks like percent, not decimal. "
            f"Use 0.0006 for 0.06%, not 0.06"
        )

    if slippage_rate < 0:
        raise ValueError(f"slippage_rate must be >= 0, got {slippage_rate}")

    if slippage_rate > 0.01:  # More than 1%
        raise ValueError(
            f"slippage_rate {slippage_rate} looks like percent, not decimal. "
            f"Use 0.0001 for 0.01%, not 0.01"
        )


# Tests to verify the canonical formulas
if __name__ == "__main__":
    # Test: trivial trade (entry=exit) should give negative return due to fees
    entry = 100.0
    exit_price = 100.0
    fee = 0.0006
    slip = 0.0001

    long_ret = compute_trade_return(entry, exit_price, 'long', fee, slip)
    short_ret = compute_trade_return(entry, exit_price, 'short', fee, slip)

    print(f"Trivial long trade return: {long_ret:.6f} (expected negative)")
    print(f"Trivial short trade return: {short_ret:.6f} (expected negative)")

    assert long_ret < 0, "Trivial long trade should have negative return"
    assert short_ret < 0, "Trivial short trade should have negative return"

    # Test: long trade with 1% price increase
    entry = 100.0
    exit_price = 101.0

    long_ret = compute_trade_return(entry, exit_price, 'long', fee, slip)
    print(f"Long trade +1%: {long_ret:.6f}")

    # Test: short trade with 1% price decrease
    entry = 100.0
    exit_price = 99.0

    short_ret = compute_trade_return(entry, exit_price, 'short', fee, slip)
    print(f"Short trade -1%: {short_ret:.6f}")

    # Test: slippage makes fills worse
    entry_fill_long = compute_fill_price(100.0, 'long', is_entry=True, slippage_rate=slip)
    exit_fill_long = compute_fill_price(100.0, 'long', is_entry=False, slippage_rate=slip)
    print(f"Long entry fill: {entry_fill_long:.4f} (should be > 100)")
    print(f"Long exit fill: {exit_fill_long:.4f} (should be < 100)")

    assert entry_fill_long > 100.0, "Long entry should be more expensive"
    assert exit_fill_long < 100.0, "Long exit should be cheaper"

    entry_fill_short = compute_fill_price(100.0, 'short', is_entry=True, slippage_rate=slip)
    exit_fill_short = compute_fill_price(100.0, 'short', is_entry=False, slippage_rate=slip)
    print(f"Short entry fill: {entry_fill_short:.4f} (should be < 100)")
    print(f"Short exit fill: {exit_fill_short:.4f} (should be > 100)")

    assert entry_fill_short < 100.0, "Short entry should be cheaper"
    assert exit_fill_short > 100.0, "Short exit should be more expensive"

    print("\nAll trade.py tests passed!")
