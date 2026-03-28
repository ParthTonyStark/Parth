from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.event import listen as event_listen
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from datetime import datetime, timedelta
import hashlib
import hmac
import base64
import uuid
import json
import os

SECRET_KEY = "change-this-secret-key-in-production"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440
DATABASE_URL = "sqlite:///./clinical_doc.db"

app = FastAPI(title="Clinical Documentation EMR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

def enable_wal_mode(db_engine):
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    event_listen(db_engine, "connect", set_sqlite_pragma)

enable_wal_mode(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class LoginRequest(BaseModel):
    username: str
    password: str

class DoctorBase(BaseModel):
    username: str
    full_name: str

class DoctorCreate(DoctorBase):
    password: str

class DoctorOut(DoctorBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    doctor: DoctorOut

class PatientBase(BaseModel):
    name: str
    age: int
    sex: str
    phone: Optional[str] = None
    locality: Optional[str] = None
    hospital_mrn: Optional[str] = None

class PatientCreate(PatientBase):
    pass

class PatientOut(PatientBase):
    id: int
    uuid: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class VisitCreate(BaseModel):
    patient_uuid: str
    department: str
    doctor_name: str
    visit_type: str
    content: Dict[str, Any]
    created_by: Optional[str] = None

class VisitOut(BaseModel):
    id: int
    patient_uuid: str
    department: str
    doctor_name: str
    visit_type: str
    content: Dict[str, Any]
    created_by: str
    created_at: Optional[datetime] = None
    patient_name: str

class DoctorDB(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)
    password = Column(String, nullable=True)
    full_name = Column(String, nullable=False, default="Doctor")
    is_active = Column(Boolean, default=True, nullable=False)

class PatientDB(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    sex = Column(String, nullable=False)
    phone = Column(String, nullable=True)
    locality = Column(String, nullable=True)
    hospital_mrn = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=True)

class VisitDB(Base):
    __tablename__ = "visits"

    id = Column(Integer, primary_key=True, index=True)
    patient_uuid = Column(String, index=True, nullable=False)
    department = Column(String, nullable=False)
    doctor_name = Column(String, nullable=False)
    visit_type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_by = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, doctor_row: DoctorDB) -> bool:
    hashed_input = get_password_hash(plain_password)
    if doctor_row.hashed_password:
        return doctor_row.hashed_password == hashed_input or doctor_row.hashed_password == plain_password
    if doctor_row.password:
        return doctor_row.password == plain_password
    return False

def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")

def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode())

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = data.copy()
    payload["exp"] = int(expire.timestamp())
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(SECRET_KEY.encode(), payload_bytes, hashlib.sha256).digest()
    return _b64encode(payload_bytes) + "." + _b64encode(sig)

def decode_access_token(token: str) -> dict:
    try:
        payload_part, sig_part = token.split(".", 1)
        payload_bytes = _b64decode(payload_part)
        sent_sig = _b64decode(sig_part)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    expected_sig = hmac.new(SECRET_KEY.encode(), payload_bytes, hashlib.sha256).digest()
    if not hmac.compare_digest(sent_sig, expected_sig):
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        payload = json.loads(payload_bytes.decode())
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    if int(payload.get("exp", 0)) < int(datetime.utcnow().timestamp()):
        raise HTTPException(status_code=401, detail="Token expired")

    return payload

def get_current_doctor(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> DoctorDB:
    payload = decode_access_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")

    doctor = db.query(DoctorDB).filter(DoctorDB.username == username).first()
    if doctor is None or not doctor.is_active:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    return doctor

def visit_to_out(visit: VisitDB, patient_name: str) -> VisitOut:
    try:
        parsed_content = json.loads(visit.content) if isinstance(visit.content, str) else visit.content
    except json.JSONDecodeError:
        parsed_content = {}
    return VisitOut(
        id=visit.id,
        patient_uuid=visit.patient_uuid,
        department=visit.department,
        doctor_name=visit.doctor_name,
        visit_type=visit.visit_type,
        content=parsed_content,
        created_by=visit.created_by,
        created_at=visit.created_at,
        patient_name=patient_name,
    )

def add_column_if_missing(table_name: str, column_name: str, column_sql: str):
    inspector = inspect(engine)
    cols = {c['name'] for c in inspector.get_columns(table_name)}
    if column_name not in cols:
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"))

def ensure_schema():
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    if 'doctors' in tables:
        add_column_if_missing('doctors', 'hashed_password', 'VARCHAR')
        add_column_if_missing('doctors', 'password', 'VARCHAR')
        add_column_if_missing('doctors', 'full_name', "VARCHAR DEFAULT 'Doctor'")
        add_column_if_missing('doctors', 'is_active', 'BOOLEAN DEFAULT 1')
        with engine.begin() as conn:
            conn.execute(text("UPDATE doctors SET full_name = COALESCE(full_name, username, 'Doctor')"))
            conn.execute(text("UPDATE doctors SET is_active = COALESCE(is_active, 1)"))

    if 'patients' in tables:
        add_column_if_missing('patients', 'uuid', 'VARCHAR')
        add_column_if_missing('patients', 'phone', 'VARCHAR')
        add_column_if_missing('patients', 'locality', 'VARCHAR')
        add_column_if_missing('patients', 'hospital_mrn', 'VARCHAR')
        add_column_if_missing('patients', 'created_at', 'DATETIME')
        with engine.begin() as conn:
            conn.execute(text("UPDATE patients SET uuid = lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))), 2) || '-' || substr('89ab', abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))), 2) || '-' || lower(hex(randomblob(6))) WHERE uuid IS NULL OR uuid = ''"))
            conn.execute(text("UPDATE patients SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP)"))

    if 'visits' in tables:
        add_column_if_missing('visits', 'created_by', 'VARCHAR')
        add_column_if_missing('visits', 'created_at', 'DATETIME')
        with engine.begin() as conn:
            conn.execute(text("UPDATE visits SET created_by = COALESCE(created_by, 'system')"))
            conn.execute(text("UPDATE visits SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP)"))

def ensure_demo_doctor():
    db = SessionLocal()
    try:
        doctor = db.query(DoctorDB).filter(DoctorDB.username == 'drparth').first()
        if not doctor:
            doctor = DoctorDB(
                username='drparth',
                hashed_password=get_password_hash('demo123'),
                full_name='Dr Parth',
                is_active=True,
            )
            db.add(doctor)
            db.commit()
            db.refresh(doctor)
        else:
            changed = False
            if not doctor.full_name:
                doctor.full_name = 'Dr Parth'
                changed = True
            if doctor.is_active is None:
                doctor.is_active = True
                changed = True
            if not doctor.hashed_password and not doctor.password:
                doctor.hashed_password = get_password_hash('demo123')
                changed = True
            elif doctor.hashed_password == 'demo123':
                doctor.hashed_password = get_password_hash('demo123')
                changed = True
            if changed:
                db.commit()
    finally:
        db.close()

@app.post('/doctors/register', response_model=DoctorOut)
def register_doctor(doctor: DoctorCreate, db: Session = Depends(get_db)):
    existing = db.query(DoctorDB).filter(DoctorDB.username == doctor.username).first()
    if existing:
        raise HTTPException(status_code=400, detail='Username already registered')

    db_doctor = DoctorDB(
        username=doctor.username.strip(),
        hashed_password=get_password_hash(doctor.password),
        full_name=doctor.full_name.strip(),
        is_active=True,
    )
    db.add(db_doctor)
    db.commit()
    db.refresh(db_doctor)
    return db_doctor

@app.post('/login', response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    doctor = db.query(DoctorDB).filter(DoctorDB.username == payload.username).first()
    if not doctor or not verify_password(payload.password, doctor):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Incorrect username or password',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    if doctor.password and doctor.password == payload.password and not doctor.hashed_password:
        doctor.hashed_password = get_password_hash(payload.password)
        db.commit()
        db.refresh(doctor)

    if doctor.hashed_password == payload.password:
        doctor.hashed_password = get_password_hash(payload.password)
        db.commit()
        db.refresh(doctor)

    access_token = create_access_token(
        data={'sub': doctor.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {
        'access_token': access_token,
        'token_type': 'bearer',
        'doctor': doctor,
    }

@app.get('/patients', response_model=List[PatientOut])
def get_patients(
    name: Optional[str] = None,
    phone: Optional[str] = None,
    hospital_mrn: Optional[str] = None,
    db: Session = Depends(get_db),
    current_doctor: DoctorDB = Depends(get_current_doctor),
):
    query = db.query(PatientDB)
    if name:
        query = query.filter(PatientDB.name.contains(name))
    if phone:
        query = query.filter(PatientDB.phone == phone)
    if hospital_mrn:
        query = query.filter(PatientDB.hospital_mrn == hospital_mrn)
    return query.order_by(PatientDB.id.desc()).all()

@app.post('/patients', response_model=PatientOut)
def create_patient(
    patient: PatientCreate,
    db: Session = Depends(get_db),
    current_doctor: DoctorDB = Depends(get_current_doctor),
):
    patient_db = PatientDB(
        uuid=str(uuid.uuid4()),
        name=patient.name,
        age=patient.age,
        sex=patient.sex,
        phone=patient.phone,
        locality=patient.locality,
        hospital_mrn=patient.hospital_mrn,
        created_at=datetime.utcnow(),
    )
    db.add(patient_db)
    db.commit()
    db.refresh(patient_db)
    return patient_db

@app.get('/patients/{patient_uuid}/visits', response_model=List[VisitOut])
def get_patient_visits(
    patient_uuid: str,
    db: Session = Depends(get_db),
    current_doctor: DoctorDB = Depends(get_current_doctor),
):
    patient = db.query(PatientDB).filter(PatientDB.uuid == patient_uuid).first()
    patient_name = patient.name if patient else 'Unknown'
    visits = (
        db.query(VisitDB)
        .filter(VisitDB.patient_uuid == patient_uuid)
        .order_by(VisitDB.id.desc())
        .all()
    )
    return [visit_to_out(v, patient_name) for v in visits]

@app.post('/visits')
def create_visit(
    visit: VisitCreate,
    db: Session = Depends(get_db),
    current_doctor: DoctorDB = Depends(get_current_doctor),
):
    patient = db.query(PatientDB).filter(PatientDB.uuid == visit.patient_uuid).first()
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    visit_db = VisitDB(
        patient_uuid=visit.patient_uuid,
        department=visit.department,
        doctor_name=visit.doctor_name,
        visit_type=visit.visit_type,
        content=json.dumps(visit.content),
        created_by=current_doctor.username,
        created_at=datetime.utcnow(),
    )
    db.add(visit_db)
    db.commit()
    db.refresh(visit_db)
    return {'visit': {'id': visit_db.id}}

@app.get('/compare_visits')
def compare_visits(
    visit_a_id: int,
    visit_b_id: int,
    db: Session = Depends(get_db),
    current_doctor: DoctorDB = Depends(get_current_doctor),
):
    visit_a = db.query(VisitDB).filter(VisitDB.id == visit_a_id).first()
    visit_b = db.query(VisitDB).filter(VisitDB.id == visit_b_id).first()
    if not visit_a or not visit_b:
        raise HTTPException(status_code=404, detail='Visit not found')

    try:
        content_a = json.loads(visit_a.content) if visit_a.content else {}
    except json.JSONDecodeError:
        content_a = {}
    try:
        content_b = json.loads(visit_b.content) if visit_b.content else {}
    except json.JSONDecodeError:
        content_b = {}

    all_fields = set(content_a.keys()) | set(content_b.keys())
    diffs = []
    for field in sorted(all_fields):
        old_val = content_a.get(field)
        new_val = content_b.get(field)
        old_str = None if old_val in [None, ''] else str(old_val)
        new_str = None if new_val in [None, ''] else str(new_val)

        if old_str is None and new_str is not None:
            state = 'NEW'
        elif old_str is not None and new_str is None:
            state = 'RESOLVED'
        elif old_str != new_str:
            state = 'CHANGED'
        else:
            state = 'SAME'

        diffs.append({
            'field': field,
            'old': old_str,
            'new': new_str,
            'status': state,
        })

    return {'diffs': diffs}

@app.get('/health')
def health():
    return {'status': 'ok'}

if not os.path.isdir('web'):
    os.makedirs('web', exist_ok=True)

app.mount('/web', StaticFiles(directory='web', html=True), name='web')

@app.get('/')
async def root():
    return FileResponse('web/index.html')

ensure_schema()
ensure_demo_doctor()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)