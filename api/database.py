import os
import sys
from sqlalchemy import create_engine, Column, Integer, String, Date, Text, DateTime, ForeignKey, Float, func
from sqlalchemy.orm import sessionmaker, relationship, declarative_base


# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/hitaishi") # Updated default URL to include user/password and port

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Patient(Base):
    __tablename__ = "customers"

    id = Column("customerid", Integer, primary_key=True, index=True)
    name = Column("fullname", String, index=True)
    # Note: Customers table uses DateOfBirth, but existing code uses age.
    # We'll map 'age' to a new column if it doesn't exist, or just use it.
    # The SQL schema has Weight, Height, etc.
    age = Column(Integer) 
    gender = Column(String)
    activity_level = Column(String)
    primary_condition = Column(String)
    created_at = Column("createdat", DateTime(timezone=True), default=func.now())

    contacts = relationship("EmergencyContact", back_populates="patient", cascade="all, delete-orphan")

class EmergencyContact(Base):
    __tablename__ = "guardians"

    id = Column("guardianid", Integer, primary_key=True, index=True)
    patient_id = Column("customerid", Integer, ForeignKey("customers.customerid"))
    name = Column("guardianname", String)
    phone = Column("contactinfo", String)
    verified = Column(Integer, default=0) # 0: False, 1: True

    patient = relationship("Patient", back_populates="contacts")

def init_db():
    print("Attempting to create database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")

def get_db():
    print(f"DEBUG: Connecting to database using URL: {DATABASE_URL}")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
