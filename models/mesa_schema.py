from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, BLOB, Date
from sqlalchemy.orm import declarative_base, relationship
import datetime

Base = declarative_base()

# ==========================================
# F2 & F4: Master Data (Products & Documents)
# ==========================================
class Product(Base):
    __tablename__ = 'products'
    product_id = Column(String(64), primary_key=True)
    routing_path = Column(String(128))
    cycle_time = Column(Float)
    bill_of_materials = Column(String(255))
    approval_status = Column(String(32))

class Document(Base):
    __tablename__ = 'documents'
    doc_id = Column(String(64), primary_key=True)
    product_id = Column(String(64), ForeignKey('products.product_id'))
    file_blob = Column(BLOB, nullable=True)
    current_rev = Column(String(16))
    document_type = Column(String(32))

# ==========================================
# F6: Labour Management 
# ==========================================
class Worker(Base):
    __tablename__ = 'workers'
    worker_id = Column(String(64), primary_key=True)
    name = Column(String(128))
    role = Column(String(64))
    active_status = Column(Boolean)
    clearances = Column(String(256))

class Shift(Base):
    __tablename__ = 'shifts'
    shift_id = Column(Integer, primary_key=True, autoincrement=True)
    worker_id = Column(String(64), ForeignKey('workers.worker_id'))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    total_hours = Column(Float)

class TimeEvent(Base):
    __tablename__ = 'time_events'
    event_id = Column(Integer, primary_key=True, autoincrement=True)
    worker_id = Column(String(64), ForeignKey('workers.worker_id'))
    action = Column(String(64))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# ==========================================
# F1 & F9: Resources & Maintenance
# ==========================================
class Resource(Base):
    __tablename__ = 'resources'
    resource_id = Column(String(64), primary_key=True)
    resource_type = Column(String(64))
    capacity_limit = Column(Integer)
    current_status = Column(String(32))
    last_maintenance = Column(DateTime)

class SkillMatrix(Base):
    __tablename__ = 'skill_matrix'
    matrix_id = Column(Integer, primary_key=True, autoincrement=True)
    worker_id = Column(String(64), ForeignKey('workers.worker_id'))
    resource_type = Column(String(64)) # Maps logically to Resource.resource_type
    certification_date = Column(DateTime)
    expiration_date = Column(DateTime)

class Equipment(Base):
    __tablename__ = 'equipment'
    equip_id = Column(String(64), primary_key=True)
    asset_class = Column(String(64))
    mtbf_hrs = Column(Float)
    operating_hrs = Column(Float)

class MaintenanceLog(Base):
    __tablename__ = 'maintenance_log'
    maint_id = Column(Integer, primary_key=True, autoincrement=True)
    equip_id = Column(String(64), ForeignKey('equipment.equip_id'))
    worker_id = Column(String(64), ForeignKey('workers.worker_id'))
    action_type = Column(String(64))
    duration_mins = Column(Float)

class FailureLog(Base):
    __tablename__ = 'failure_log'
    fail_id = Column(Integer, primary_key=True, autoincrement=True)
    equip_id = Column(String(64), ForeignKey('equipment.equip_id'))
    cause_mode = Column(String(128))
    time_down = Column(DateTime)
    time_up = Column(DateTime, nullable=True)

# ==========================================
# F2 & F3: Operations Scheduling & Dispatching
# ==========================================
class WorkOrder(Base):
    __tablename__ = 'work_orders'
    order_id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(64), ForeignKey('products.product_id'))
    requesting_unit = Column(String(128))
    due_date = Column(DateTime)
    status = Column(String(32))

class Schedule(Base):
    __tablename__ = 'schedule'
    schedule_id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey('work_orders.order_id'))
    planned_start = Column(DateTime)
    planned_end = Column(DateTime)
    sequence_rank = Column(Integer)

class ResourceAssignment(Base):
    __tablename__ = 'resource_assignment'
    assignment_id = Column(Integer, primary_key=True, autoincrement=True)
    resource_id = Column(String(64), ForeignKey('resources.resource_id'))
    task_id = Column(Integer, ForeignKey('work_orders.order_id'))
    assigned_by = Column(String(64), ForeignKey('workers.worker_id'))
    expected_duration = Column(Float)

class ResourceStatus(Base):
    __tablename__ = 'resource_status'
    status_id = Column(Integer, primary_key=True, autoincrement=True)
    resource_id = Column(String(64), ForeignKey('resources.resource_id'))
    state = Column(String(32))
    job_in_progress = Column(Integer, ForeignKey('work_orders.order_id'), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class DispatchQueue(Base):
    __tablename__ = 'dispatch_queue'
    queue_id = Column(Integer, primary_key=True, autoincrement=True)
    resource_id = Column(String(64), ForeignKey('resources.resource_id'))
    order_id = Column(Integer, ForeignKey('work_orders.order_id'))
    local_priority = Column(Float)
    time_entered = Column(DateTime, default=datetime.datetime.utcnow)

class DispatchLog(Base):
    __tablename__ = 'dispatch_log'
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    queue_id = Column(Integer, ForeignKey('dispatch_queue.queue_id'))
    dispatched_at = Column(DateTime, default=datetime.datetime.utcnow)
    authorized_by = Column(String(64), ForeignKey('workers.worker_id'), nullable=True)
    overridden = Column(Boolean, default=False)

# ==========================================
# F4 Continued: Revisions & Acks
# ==========================================
class Revision(Base):
    __tablename__ = 'revisions'
    rev_id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(64), ForeignKey('documents.doc_id'))
    hash_key = Column(String(256))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    created_by = Column(String(64), ForeignKey('workers.worker_id'))

class Acknowledgment(Base):
    __tablename__ = 'acknowledgments'
    ack_id = Column(Integer, primary_key=True, autoincrement=True)
    rev_id = Column(Integer, ForeignKey('revisions.rev_id'))
    worker_id = Column(String(64), ForeignKey('workers.worker_id'))
    signature_time = Column(DateTime, default=datetime.datetime.utcnow)
    valid = Column(Boolean)

# ==========================================
# F10: Product Tracking (Lots & Genealogy)
# ==========================================
class Lot(Base):
    __tablename__ = 'lots'
    lot_id = Column(String(128), primary_key=True)
    product_id = Column(String(64), ForeignKey('products.product_id'))
    original_qty = Column(Integer)
    current_qty = Column(Integer)

class Genealogy(Base):
    __tablename__ = 'genealogy'
    serial_id = Column(String(128), primary_key=True)
    lot_id = Column(String(128), ForeignKey('lots.lot_id'))
    material_id = Column(String(64)) # Could fk to raw inventory
    work_order_id = Column(Integer, ForeignKey('work_orders.order_id'))
    status = Column(String(32))

class LotEvent(Base):
    __tablename__ = 'lot_events'
    event_id = Column(Integer, primary_key=True, autoincrement=True)
    serial_id = Column(String(128), ForeignKey('genealogy.serial_id'))
    station_point = Column(String(64), ForeignKey('resources.resource_id'))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# ==========================================
# F7: Quality Management
# ==========================================
class QualitySpec(Base):
    __tablename__ = 'quality_specs'
    spec_id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(64), ForeignKey('products.product_id'))
    dimension_name = Column(String(64))
    target_value = Column(Float)
    tolerance = Column(Float)

class InspResult(Base):
    __tablename__ = 'insp_results'
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    serial_id = Column(String(128), ForeignKey('genealogy.serial_id'))
    spec_id = Column(Integer, ForeignKey('quality_specs.spec_id'))
    measured_value = Column(Float)
    passed = Column(Boolean)

class NCRecord(Base):
    __tablename__ = 'nc_records'
    ncr_id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(Integer, ForeignKey('insp_results.result_id'))
    severity = Column(String(32))
    capa_action = Column(String(128))
    rework_route = Column(String(128))

# ==========================================
# F5 & F8: Data Collection & Process Management
# ==========================================
class Sensor(Base):
    __tablename__ = 'sensors'
    sensor_id = Column(String(64), primary_key=True)
    node_id = Column(String(64), ForeignKey('resources.resource_id'))
    polling_hz = Column(Integer)
    calibration_date = Column(DateTime)
    status = Column(String(16))

class SensorReading(Base):
    __tablename__ = 'sensor_readings'
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    sensor_id = Column(String(64), ForeignKey('sensors.sensor_id'))
    accel_x = Column(Float)
    gyro_x = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class Event(Base):
    __tablename__ = 'events'
    event_id = Column(Integer, primary_key=True, autoincrement=True)
    sensor_id = Column(String(64), ForeignKey('sensors.sensor_id'))
    event_type = Column(String(64))
    alert_severity = Column(Integer)
    acknowledged = Column(Boolean, default=False)

class RecipeParam(Base):
    __tablename__ = 'recipe_params'
    param_id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(64), ForeignKey('documents.doc_id'))
    nominal_trust = Column(Float)
    anomaly_thresh = Column(Float)

class ProcessEvent(Base):
    __tablename__ = 'process_events'
    event_id = Column(Integer, primary_key=True, autoincrement=True)
    serial_id = Column(String(128), ForeignKey('genealogy.serial_id'))
    recon_error = Column(Float)
    current_trust = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class Deviation(Base):
    __tablename__ = 'deviations'
    dev_id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey('process_events.event_id'))
    system_action = Column(String(128))
    clearance_id = Column(String(64), ForeignKey('workers.worker_id'), nullable=True)

# ==========================================
# F11: Performance Analysis
# ==========================================
class KPILog(Base):
    __tablename__ = 'kpi_log'
    kpi_id = Column(Integer, primary_key=True, autoincrement=True)
    date_index = Column(Date)
    value_a = Column(Float)
    value_p = Column(Float)
    value_q = Column(Float)

class OEELog(Base):
    __tablename__ = 'oee_log'
    oee_id = Column(Integer, primary_key=True, autoincrement=True)
    work_center = Column(String(64), ForeignKey('resources.resource_id'))
    final_oee = Column(Float)

class ShiftReport(Base):
    __tablename__ = 'shift_reports'
    report_id = Column(Integer, primary_key=True, autoincrement=True)
    shift_id = Column(Integer, ForeignKey('shifts.shift_id'))
    defect_rate = Column(Float)
    report_time = Column(DateTime, default=datetime.datetime.utcnow)
