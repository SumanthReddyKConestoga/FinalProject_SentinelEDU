"""SQLAlchemy ORM models for SentinelEDU."""
from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from src.config import SETTINGS

Base = declarative_base()


class Student(Base):
    __tablename__ = "students"
    id = Column(String, primary_key=True)  # student_id like S0047
    age = Column(Integer)
    sex = Column(String(2))
    school = Column(String(8))
    program = Column(String(32), default="General")
    enrollment_date = Column(DateTime, default=datetime.utcnow)
    segment = Column(String(64), default=None)  # assigned by clustering
    current_risk_class = Column(String(16), default="Low")
    current_predicted_g3 = Column(Float, default=None)
    static_features = Column(JSON, default=dict)  # for API hydration

    predictions = relationship("Prediction", back_populates="student")
    alerts = relationship("Alert", back_populates="student")
    recommendations = relationship("Recommendation", back_populates="student")


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String, ForeignKey("students.id"))
    model_name = Column(String(64))
    prediction_value = Column(Float)
    predicted_class = Column(String(16), default=None)
    confidence = Column(Float, default=None)
    week = Column(Integer, default=None)
    created_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="predictions")


class WeeklyRecord(Base):
    __tablename__ = "weekly_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String, ForeignKey("students.id"))
    week = Column(Integer)
    attendance_pct = Column(Float)
    quiz_score = Column(Float)
    submission_rate = Column(Float)
    lms_logins = Column(Integer)
    late_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String, ForeignKey("students.id"))
    rule_id = Column(String(64))
    severity = Column(String(16))
    description = Column(String(256))
    evidence = Column(JSON, default=dict)
    acknowledged = Column(Boolean, default=False)
    triggered_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="alerts")


class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String, ForeignKey("students.id"))
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=True)
    action = Column(String(64))
    rationale = Column(String(256))
    priority = Column(Float)
    status = Column(String(16), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="recommendations")


class AdvisorAction(Base):
    __tablename__ = "advisor_actions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String, ForeignKey("students.id"))
    recommendation_id = Column(Integer, ForeignKey("recommendations.id"), nullable=True)
    advisor_name = Column(String(64), default="Advisor")
    action_taken = Column(String(64))
    notes = Column(String(512), default="")
    created_at = Column(DateTime, default=datetime.utcnow)


# --- Engine + session ---
engine = create_engine(SETTINGS["database"]["url"], echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db():
    Base.metadata.create_all(engine)
