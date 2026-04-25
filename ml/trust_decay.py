import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from backend.models.mesa_schema import ProcessEvent, Deviation

class TrustDecayEngine:
    def __init__(self, alpha=0.85, initial_trust=1.0, threshold=0.30):
        self.alpha = alpha
        self.current_trust = initial_trust
        self.nominal_trust = initial_trust
        self.anomaly_threshold = threshold
        
    def process_telemetry(self, recon_error: float) -> float:
        # Trust decays when reconstruction error is high
        # Γ(t+1) = α * Γ(t) + (1-α) * (Nominal - Penalty)
        penalty = min(1.0, recon_error / 100.0) # normalizer
        target = max(0.0, self.nominal_trust - penalty)
        self.current_trust = self.alpha * self.current_trust + (1.0 - self.alpha) * target
        return self.current_trust
        
    def is_compromised(self):
        return self.current_trust <= self.anomaly_threshold

async def record_process_event(session: AsyncSession, serial_id: str, recon_error: float, trust: float):
    stmt = insert(ProcessEvent).values(
        serial_id=serial_id,
        recon_error=recon_error,
        current_trust=trust,
        timestamp=datetime.datetime.utcnow()
    )
    res = await session.execute(stmt)
    event_id = res.inserted_primary_key[0]
    return event_id
    
async def trigger_deviation(session: AsyncSession, event_id: int, action: str):
    stmt = insert(Deviation).values(
        event_id=event_id,
        system_action=action,
        clearance_id=None
    )
    await session.execute(stmt)
