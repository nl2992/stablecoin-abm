"""Primary redemption channel — the issuer's on-demand mint/redeem facility.

Tracks the full balance sheet:
  - reserve_usd : USD held in the redemption reserve
  - total_supply : stablecoins in circulation (minted − redeemed)

Gating controls:
  - flat redemption fee (fee_bps)
  - queue cap + settlement delay
  - circuit breaker (halts redemptions when |depeg| ≥ threshold)

Accounting invariant (zero-fee, no exhaustion):
  After any mint(x) followed by a full redeem(x):
      reserve_usd returns to initial and total_supply returns to initial.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class RedemptionOrder:
    agent_id: str
    amount: float          # stablecoins to redeem
    submitted_step: int


class RedemptionChannel:
    """Issuer redemption facility with configurable gating and balance-sheet accounting.

    Parameters
    ----------
    initial_reserve_usd : float
        USD held in reserve at t=0.
    initial_supply : float
        Stablecoins outstanding at t=0.
    fee_bps : float
        Flat redemption fee in basis points (applied to USD paid out).
    max_queue : int
        Maximum pending redemption orders (0 = unlimited).
    delay_steps : int
        Steps between submission and settlement.
    cb_threshold : float
        |price − 1| that triggers circuit breaker.
    cb_duration : int
        Steps the circuit breaker stays active once triggered.
    """

    def __init__(
        self,
        initial_reserve_usd: float = 1_000_000.0,
        initial_supply: float = 1_000_000.0,
        fee_bps: float = 0.0,
        max_queue: int = 0,
        delay_steps: int = 0,
        cb_threshold: float = 0.10,
        cb_duration: int = 12,
    ) -> None:
        self.reserve_usd = float(initial_reserve_usd)
        self.total_supply = float(initial_supply)
        self.fee_bps = float(fee_bps)
        self.max_queue = max_queue
        self.delay_steps = delay_steps
        self.cb_threshold = float(cb_threshold)
        self.cb_duration = int(cb_duration)

        self._queue: deque[RedemptionOrder] = deque()
        self._cb_active_until: int = 0
        self._settled: list[dict] = []

        # Accounting ledger
        self._total_minted: float = 0.0
        self._total_redeemed: float = 0.0
        self._total_fees_collected: float = 0.0

    # ------------------------------------------------------------------
    # Balance sheet

    @property
    def backing_ratio(self) -> float:
        """USD reserve / stablecoins outstanding.  1.0 = fully backed."""
        if self.total_supply <= 0:
            return float("inf")
        return self.reserve_usd / self.total_supply

    @property
    def is_exhausted(self) -> bool:
        """True when the reserve has no USD to honour further redemptions."""
        return self.reserve_usd <= 0.0

    # ------------------------------------------------------------------
    # Mint

    def mint(self, usd_in: float) -> float:
        """Mint stablecoins 1:1 against USD deposit.  Returns stablecoins issued.

        Accounting: reserve_usd += usd_in; total_supply += usd_in.
        """
        if usd_in <= 0:
            return 0.0
        stablecoins = usd_in  # 1:1 peg on issuance
        self.reserve_usd += usd_in
        self.total_supply += stablecoins
        self._total_minted += stablecoins
        return stablecoins

    # ------------------------------------------------------------------
    # Redemption pipeline

    def submit(self, agent_id: str, amount: float, current_step: int) -> bool:
        """Attempt to submit a redemption order.  Returns True if accepted.

        Rejected if: queue full, circuit breaker active, or reserve exhausted.
        """
        if amount <= 0:
            return False
        if self.max_queue > 0 and len(self._queue) >= self.max_queue:
            return False
        if current_step < self._cb_active_until:
            return False
        if self.is_exhausted:
            return False
        self._queue.append(RedemptionOrder(agent_id, amount, current_step))
        return True

    def settle(self, current_step: int) -> list[dict]:
        """Process all orders whose delay has elapsed.  Returns settled records.

        Accounting per order:
            usd_gross = stablecoins redeemed (1:1)
            fee_usd   = usd_gross × fee
            usd_net   = usd_gross − fee_usd   → paid to redeemer
            reserve_usd  −= usd_gross
            total_supply −= stablecoins
            (fee stays in reserve: reserve_usd += fee_usd — so net reserve change = −usd_net)
        """
        fee_rate = self.fee_bps / 10_000.0
        ready: list[RedemptionOrder] = []
        remaining: deque[RedemptionOrder] = deque()

        for order in self._queue:
            if current_step - order.submitted_step >= self.delay_steps:
                ready.append(order)
            else:
                remaining.append(order)
        self._queue = remaining

        settled = []
        for order in ready:
            # Cap at available reserve
            redeemable = min(order.amount, max(self.reserve_usd, 0.0))
            if redeemable <= 0:
                continue

            usd_gross = redeemable
            fee_usd = usd_gross * fee_rate
            usd_net = usd_gross - fee_usd

            # Update balance sheet
            self.reserve_usd -= usd_net          # net USD leaves reserve
            self.total_supply -= redeemable       # stablecoins burned
            self._total_redeemed += redeemable
            self._total_fees_collected += fee_usd

            record = {
                "agent_id": order.agent_id,
                "stablecoins_burned": redeemable,
                "usd_gross": usd_gross,
                "fee_usd": fee_usd,
                "usd_net": usd_net,
                "settled_step": current_step,
                "partial": redeemable < order.amount,
            }
            settled.append(record)

        self._settled.extend(settled)
        return settled

    # ------------------------------------------------------------------
    # Circuit breaker

    def trigger_circuit_breaker(self, current_step: int) -> None:
        self._cb_active_until = current_step + self.cb_duration

    def check_and_trigger(self, price: float, current_step: int) -> bool:
        """Trigger circuit breaker if |depeg| ≥ threshold.  Returns True if triggered."""
        if abs(price - 1.0) >= self.cb_threshold:
            self.trigger_circuit_breaker(current_step)
            return True
        return False

    def is_halted(self, current_step: int) -> bool:
        return current_step < self._cb_active_until

    # ------------------------------------------------------------------
    # Accessors

    def queue_depth(self) -> int:
        return len(self._queue)

    def accounting_summary(self) -> dict:
        """Full balance-sheet snapshot for validation."""
        return {
            "reserve_usd": self.reserve_usd,
            "total_supply": self.total_supply,
            "backing_ratio": self.backing_ratio,
            "total_minted": self._total_minted,
            "total_redeemed": self._total_redeemed,
            "total_fees_collected": self._total_fees_collected,
            "is_exhausted": self.is_exhausted,
        }

    def state(self) -> dict:
        return {
            "queue_depth": self.queue_depth(),
            "fee_bps": self.fee_bps,
            "backing_ratio": self.backing_ratio,
            "cb_active_until": self._cb_active_until,
            "reserve_usd": self.reserve_usd,
        }
