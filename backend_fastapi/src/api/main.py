import os
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float, Table
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
import httpx

# Load environment variables from .env
load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./career_nav.db")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001/recommend")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# -----------------------------------------------------------------------------
# Database Setup (SQLAlchemy)
# -----------------------------------------------------------------------------

# SQLite needs check_same_thread=False for standard synchronous engine
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Association table for many-to-many Role-Skill
role_skills_table = Table(
    "role_skills",
    Base.metadata,
    Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
    Column("skill_id", Integer, ForeignKey("skills.id"), primary_key=True),
    Column("required_level", Float, nullable=True),
)

class Role(Base):
    """Role model representing job roles."""
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)

    skills = relationship("Skill", secondary=role_skills_table, back_populates="roles")


class Skill(Base):
    """Skill model representing skills aligned to frameworks (e.g., SFIA)."""
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), unique=True, nullable=False, index=True)
    category = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)

    roles = relationship("Role", secondary=role_skills_table, back_populates="skills")


class User(Base):
    """Minimal user table to track progress updates. Expand as needed."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=True)


class UserSkillProgress(Base):
    """Per-user progress on a skill (0-1 scale or percentage)."""
    __tablename__ = "user_skill_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    skill_id = Column(Integer, ForeignKey("skills.id"), nullable=False)
    progress = Column(Float, nullable=False, default=0.0)  # 0.0 to 1.0

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------

class SkillBase(BaseModel):
    id: int = Field(..., description="Unique identifier for the skill")
    name: str = Field(..., description="Name of the skill")
    category: Optional[str] = Field(None, description="Category or family of the skill")
    description: Optional[str] = Field(None, description="Description of the skill")

    class Config:
        from_attributes = True


class RoleBase(BaseModel):
    id: int = Field(..., description="Unique identifier for the role")
    name: str = Field(..., description="Name of the role")
    description: Optional[str] = Field(None, description="Description of the role")

    class Config:
        from_attributes = True


class RoleDetail(RoleBase):
    skills: List[SkillBase] = Field(default_factory=list, description="Skills associated with this role")


class GapAnalysisRequest(BaseModel):
    current_skills: List[str] = Field(..., description="List of user's current skill names")
    target_role_id: int = Field(..., description="Target role ID for gap analysis")


class GapItem(BaseModel):
    skill: str = Field(..., description="Skill name")
    required: bool = Field(..., description="Whether the skill is required by the role")
    gap: bool = Field(..., description="True if missing from user's current skills")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions to close the gap")


class GapAnalysisResponse(BaseModel):
    target_role_id: int = Field(..., description="Analyzed role ID")
    target_role_name: str = Field(..., description="Analyzed role name")
    gaps: List[GapItem] = Field(default_factory=list, description="List of skills and gaps")


class RoadmapRequest(BaseModel):
    target_role_id: int = Field(..., description="Target role ID to build a roadmap for")
    current_skills: List[str] = Field(default_factory=list, description="User's current skill names")


class CytoscapeElement(BaseModel):
    data: Dict[str, Any] = Field(..., description="Cytoscape element data dictionary")
    group: Optional[str] = Field(None, description="Optional group (nodes/edges)")
    selectable: Optional[bool] = Field(None, description="Whether selectable")
    grabbable: Optional[bool] = Field(None, description="Whether can be grabbed")


class RoadmapResponse(BaseModel):
    elements: List[CytoscapeElement] = Field(..., description="Cytoscape.js elements array for rendering a roadmap graph")


class ProgressUpdateRequest(BaseModel):
    user_email: str = Field(..., description="User email to identify the user")
    skill_name: str = Field(..., description="Skill name")
    progress: float = Field(..., ge=0.0, le=1.0, description="Progress value between 0 and 1")


class RecommendRequest(BaseModel):
    user_profile: Dict[str, Any] = Field(..., description="Arbitrary profile data including goals, interests, background")
    current_skills: List[str] = Field(default_factory=list, description="User's current skills")
    target_role_id: Optional[int] = Field(None, description="Optional target role ID")


class RecommendResponse(BaseModel):
    recommendations: Dict[str, Any] = Field(..., description="Recommendations payload from LLM service")


# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------

openapi_tags = [
    {"name": "Health", "description": "Health and diagnostics"},
    {"name": "Roles", "description": "Role catalog and details"},
    {"name": "Analysis", "description": "Gap analysis and roadmap generation"},
    {"name": "Progress", "description": "User progress tracking"},
    {"name": "LLM", "description": "LLM-driven recommendations"},
]

app = FastAPI(
    title="Career Navigator Backend",
    description="Backend API for the Career Navigator Platform MVP. Provides role data, gap analysis, roadmap generation, progress tracking, and LLM recommendations.",
    version="0.1.0",
    openapi_tags=openapi_tags,
)

# Configure CORS from env (comma-separated list or '*')
origins: List[str]
if CORS_ORIGINS.strip() == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

def get_db():
    """Yield a database session per request and ensure it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def get_role_by_id(db: Session, role_id: int) -> Role:
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
    return role


def ensure_user(db: Session, email: str) -> User:
    user = db.query(User).filter(User.email == email).first()
    if user:
        return user
    user = User(email=email, name=email.split("@")[0])
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def link_required_level(db: Session, role_id: int, skill_id: int) -> Optional[float]:
    # Utility to fetch required level if present in association table
    # For SQLite, reflect using direct query
    result = db.execute(
        role_skills_table.select()
        .where(role_skills_table.c.role_id == role_id)
        .where(role_skills_table.c.skill_id == skill_id)
    ).first()
    if result and "required_level" in result._mapping:
        return result._mapping["required_level"]
    return None

# -----------------------------------------------------------------------------
# Seed minimal data if tables are empty (for MVP local run)
# -----------------------------------------------------------------------------

def seed_minimal(db: Session) -> None:
    if db.query(Role).count() > 0:
        return
    # Create a minimal set of skills and roles
    skill_python = Skill(name="Python", category="Engineering", description="Programming in Python")
    skill_sql = Skill(name="SQL", category="Data", description="Querying relational databases")
    skill_communication = Skill(name="Communication", category="Leadership", description="Clear communication")
    db.add_all([skill_python, skill_sql, skill_communication])
    db.commit()
    db.refresh(skill_python)
    db.refresh(skill_sql)
    db.refresh(skill_communication)

    role_ds = Role(name="Data Scientist", description="Builds models and analyzes data")
    role_ds.skills = [skill_python, skill_sql, skill_communication]
    db.add(role_ds)

    role_se = Role(name="Software Engineer", description="Builds software systems")
    role_se.skills = [skill_python, skill_communication]
    db.add(role_se)

    db.commit()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

# PUBLIC_INTERFACE
@app.get("/", tags=["Health"], summary="Health Check")
def health_check():
    """Health check endpoint.
    Returns:
        dict: Simple health message for monitoring.
    """
    return {"message": "Healthy"}

# PUBLIC_INTERFACE
@app.on_event("startup")
def on_startup():
    """Startup hook to ensure DB schema and seed minimal data."""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_minimal(db)

# PUBLIC_INTERFACE
@app.get("/roles", response_model=List[RoleBase], tags=["Roles"], summary="List roles")
def list_roles(db: Session = Depends(get_db)):
    """List all roles with basic details.
    Returns:
        List[RoleBase]: Array of roles.
    """
    roles = db.query(Role).order_by(Role.name.asc()).all()
    return roles

# PUBLIC_INTERFACE
@app.get("/roles/{role_id}", response_model=RoleDetail, tags=["Roles"], summary="Get role by ID")
def get_role(role_id: int, db: Session = Depends(get_db)):
    """Get role details including associated skills.
    Args:
        role_id (int): Role ID
    Returns:
        RoleDetail: Role with skills
    """
    role = get_role_by_id(db, role_id)
    return RoleDetail(
        id=role.id,
        name=role.name,
        description=role.description,
        skills=[SkillBase.model_validate(s) for s in role.skills],
    )

# PUBLIC_INTERFACE
@app.post("/gap-analysis", response_model=GapAnalysisResponse, tags=["Analysis"], summary="Perform gap analysis")
def post_gap_analysis(payload: GapAnalysisRequest, db: Session = Depends(get_db)):
    """Perform gap analysis between user's current skills and a target role's required skills.
    Args:
        payload (GapAnalysisRequest): current_skills list and target_role_id
    Returns:
        GapAnalysisResponse: Gaps and suggested actions
    """
    role = get_role_by_id(db, payload.target_role_id)
    current_set = {s.lower() for s in payload.current_skills}
    gaps: List[GapItem] = []

    for skill in role.skills:
        has_skill = skill.name.lower() in current_set
        recs = []
        if not has_skill:
            recs = [
                f"Take an introductory course on {skill.name}",
                f"Complete a small project demonstrating {skill.name}",
            ]
        gaps.append(
            GapItem(
                skill=skill.name,
                required=True,
                gap=not has_skill,
                recommended_actions=recs,
            )
        )

    return GapAnalysisResponse(
        target_role_id=role.id,
        target_role_name=role.name,
        gaps=gaps,
    )

# PUBLIC_INTERFACE
@app.post("/roadmap", response_model=RoadmapResponse, tags=["Analysis"], summary="Generate roadmap graph")
def post_roadmap(payload: RoadmapRequest, db: Session = Depends(get_db)):
    """Generate a simple Cytoscape.js graph representing steps from current skills to a target role.
    Args:
        payload (RoadmapRequest): target_role_id and current_skills
    Returns:
        RoadmapResponse: Cytoscape.js elements
    """
    role = get_role_by_id(db, payload.target_role_id)
    current_set = {s.lower() for s in payload.current_skills}

    elements: List[CytoscapeElement] = []

    # Node for role
    role_node_id = f"role-{role.id}"
    elements.append(
        CytoscapeElement(
            data={"id": role_node_id, "label": role.name, "type": "role"},
            group="nodes",
        )
    )

    # Nodes for each required skill
    for skill in role.skills:
        skill_node_id = f"skill-{skill.id}"
        have_it = skill.name.lower() in current_set
        elements.append(
            CytoscapeElement(
                data={
                    "id": skill_node_id,
                    "label": skill.name,
                    "type": "skill",
                    "status": "have" if have_it else "need",
                },
                group="nodes",
            )
        )
        # Edge from skill to role (indicating prerequisite)
        elements.append(
            CytoscapeElement(
                data={"id": f"edge-{skill_node_id}-{role_node_id}", "source": skill_node_id, "target": role_node_id, "type": "requires"},
                group="edges",
            )
        )

        if not have_it:
            # Add a learning task node for missing skills
            task_node_id = f"task-{skill.id}"
            elements.append(
                CytoscapeElement(
                    data={
                        "id": task_node_id,
                        "label": f"Learn {skill.name}",
                        "type": "task",
                    },
                    group="nodes",
                )
            )
            elements.append(
                CytoscapeElement(
                    data={"id": f"edge-{task_node_id}-{skill_node_id}", "source": task_node_id, "target": skill_node_id, "type": "enables"},
                    group="edges",
                )
            )

    return RoadmapResponse(elements=elements)

# PUBLIC_INTERFACE
@app.post("/progress/update", tags=["Progress"], summary="Update user skill progress")
def post_progress_update(payload: ProgressUpdateRequest, db: Session = Depends(get_db)):
    """Update or create the user's progress for a given skill.
    Args:
        payload (ProgressUpdateRequest): user_email, skill_name, progress [0,1]
    Returns:
        dict: status message and current record
    """
    user = ensure_user(db, payload.user_email)
    skill = db.query(Skill).filter(Skill.name.ilike(payload.skill_name)).first()
    if not skill:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found")

    record = (
        db.query(UserSkillProgress)
        .filter(UserSkillProgress.user_id == user.id, UserSkillProgress.skill_id == skill.id)
        .first()
    )

    if record:
        record.progress = payload.progress
    else:
        record = UserSkillProgress(user_id=user.id, skill_id=skill.id, progress=payload.progress)
        db.add(record)

    db.commit()
    db.refresh(record)

    return {
        "message": "Progress updated",
        "user_email": payload.user_email,
        "skill_name": skill.name,
        "progress": record.progress,
    }

# PUBLIC_INTERFACE
@app.post("/recommend", response_model=RecommendResponse, tags=["LLM"], summary="Proxy to LLM recommend service")
async def post_recommend(payload: RecommendRequest, db: Session = Depends(get_db)):
    """Proxy request to an LLM recommendation microservice.
    Args:
        payload (RecommendRequest): user profile, current skills, optional target role
    Returns:
        RecommendResponse: LLM service response payload
    """
    # Optionally enrich with role details
    role_info = None
    if payload.target_role_id is not None:
        try:
            role = get_role_by_id(db, payload.target_role_id)
            role_info = {"id": role.id, "name": role.name, "skills": [s.name for s in role.skills]}
        except HTTPException:
            role_info = None

    proxy_body = {
        "user_profile": payload.user_profile,
        "current_skills": payload.current_skills,
        "target_role": role_info,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(LLM_SERVICE_URL, json=proxy_body)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"LLM service error: {str(e)}") from e

    if not isinstance(data, dict) or "recommendations" not in data:
        # Normalize minimal structure
        data = {"recommendations": data}

    return RecommendResponse(**data)
